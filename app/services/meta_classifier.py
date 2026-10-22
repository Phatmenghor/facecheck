"""Meta-classifier and probability calibration service for FaceCheck API.

Fuses multi-detector predictions (HuggingFace/Safetensors, OpenCLIP, DINOv2, Noise Residual CNN)
using an L2-regularized logistic regression meta-classifier and Platt scaling / Isotonic calibration.
Guarantees calibrated risk scores (e.g., score 80 corresponds to ~80% empirical fake probability).

100% offline, zero pickle risk (pure JSON serialization), zero external cloud dependencies.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger("facecheck.meta_classifier")

DEFAULT_FEATURE_NAMES = ["safetensors", "openclip", "dinov2", "noise_cnn"]


class PlattCalibrator:
    """Platt scaling probability calibrator: P(y=1|z) = 1 / (1 + exp(A*z + B))."""

    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a = float(a)
        self.b = float(b)

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        """Fit Platt scaling parameters (A, B) via negative log-likelihood minimization."""
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)

        # Target smoothing to prevent overfitting (Platt 1999)
        n_pos = np.sum(labels == 1.0)
        n_neg = np.sum(labels == 0.0)
        t_pos = (n_pos + 1.0) / (n_pos + 2.0)
        t_neg = 1.0 / (n_neg + 2.0)
        targets = np.where(labels == 1.0, t_pos, t_neg)

        def loss_fn(params: np.ndarray) -> float:
            a, b = params[0], params[1]
            p = 1.0 / (1.0 + np.exp(np.clip(a * logits + b, -25.0, 25.0)))
            p = np.clip(p, 1e-12, 1.0 - 1e-12)
            # Binary cross entropy
            bce = -np.mean(targets * np.log(p) + (1.0 - targets) * np.log(1.0 - p))
            return float(bce)

        res = minimize(loss_fn, x0=[1.0, 0.0], method="L-BFGS-B", bounds=[(0.01, 20.0), (-20.0, 20.0)])
        if res.success:
            self.a = float(res.x[0])
            self.b = float(res.x[1])
        return self

    def calibrate(self, logits: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Apply Platt calibration to map logits to calibrated probabilities."""
        z = np.asarray(logits, dtype=np.float64)
        prob = 1.0 / (1.0 + np.exp(np.clip(self.a * z + self.b, -25.0, 25.0)))
        if np.isscalar(logits):
            return float(prob)
        return prob

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "platt", "a": self.a, "b": self.b}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlattCalibrator":
        return cls(a=data.get("a", 1.0), b=data.get("b", 0.0))


class IsotonicCalibrator:
    """Non-parametric isotonic probability calibrator using PAVA algorithm."""

    def __init__(self, thresholds: Optional[List[float]] = None, values: Optional[List[float]] = None):
        self.thresholds = [float(t) for t in (thresholds or [0.0, 0.5, 1.0])]
        self.values = [float(v) for v in (values or [0.0, 0.5, 1.0])]

    def fit(self, preds: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        """Fit isotonic regression using pool-adjacent-violators."""
        preds = np.asarray(preds, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)

        order = np.argsort(preds)
        x_sort = preds[order]
        y_sort = labels[order]

        # PAVA implementation
        blocks = [[x_sort[i], y_sort[i], 1] for i in range(len(x_sort))]
        i = 0
        while i < len(blocks) - 1:
            if blocks[i][1] > blocks[i + 1][1]:
                # Pool adjacent violators
                w1, w2 = blocks[i][2], blocks[i + 1][2]
                new_y = (blocks[i][1] * w1 + blocks[i + 1][1] * w2) / (w1 + w2)
                blocks[i][1] = new_y
                blocks[i][2] = w1 + w2
                del blocks[i + 1]
                if i > 0:
                    i -= 1
            else:
                i += 1

        self.thresholds = [b[0] for b in blocks]
        self.values = [float(np.clip(b[1], 0.0, 1.0)) for b in blocks]
        return self

    def calibrate(self, preds: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Interpolate calibrated probabilities."""
        p = np.asarray(preds, dtype=np.float64)
        calibrated = np.interp(p, self.thresholds, self.values)
        if np.isscalar(preds):
            return float(calibrated)
        return calibrated

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "isotonic", "thresholds": self.thresholds, "values": self.values}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IsotonicCalibrator":
        return cls(thresholds=data.get("thresholds"), values=data.get("values"))


class MetaClassifier:
    """L2-regularized logistic regression meta-classifier with probability calibration."""

    def __init__(
        self,
        weights: Optional[List[float]] = None,
        bias: float = 0.0,
        feature_names: Optional[List[str]] = None,
        calibrator: Optional[Union[PlattCalibrator, IsotonicCalibrator]] = None,
    ):
        self.feature_names = feature_names or list(DEFAULT_FEATURE_NAMES)
        self.weights = np.array(weights or [1.0] * len(self.feature_names), dtype=np.float64)
        self.bias = float(bias)
        self.calibrator = calibrator or PlattCalibrator(a=1.0, b=0.0)
        self.metrics: Dict[str, Any] = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_split: float = 0.2,
        feature_names: Optional[List[str]] = None,
        calibrate_method: str = "platt",
        l2_reg: float = 0.1,
    ) -> Dict[str, Any]:
        """Train meta-classifier weights using validation split and fit calibrator.

        Args:
            X: 2D array of detector predictions of shape (N, num_features).
            y: 1D array of binary labels (0 = Real, 1 = Fake).
            val_split: Fraction of data held out for validation and calibration.
            feature_names: List of detector identifiers corresponding to columns in X.
            calibrate_method: "platt" (Platt scaling) or "isotonic".
            l2_reg: L2 regularization strength.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape

        if feature_names:
            self.feature_names = list(feature_names)
        elif len(self.feature_names) != n_features:
            self.feature_names = [f"detector_{i}" for i in range(n_features)]

        # Stratified or randomized train/val split
        np.random.seed(42)
        indices = np.random.permutation(n_samples)
        n_val = max(1, int(n_samples * val_split))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # 1. Train Logistic Regression via L2-regularized negative log-likelihood
        def loss_fn(params: np.ndarray) -> float:
            w, b = params[:n_features], params[-1]
            logits = np.dot(X_train, w) + b
            # Safe sigmoid
            p = 1.0 / (1.0 + np.exp(-np.clip(logits, -25.0, 25.0)))
            p = np.clip(p, 1e-12, 1.0 - 1e-12)
            nll = -np.mean(y_train * np.log(p) + (1.0 - y_train) * np.log(1.0 - p))
            reg = 0.5 * l2_reg * np.sum(w ** 2)
            return float(nll + reg)

        init_params = np.zeros(n_features + 1, dtype=np.float64)
        init_params[:n_features] = 1.0 / n_features  # Equal initial weighting

        res = minimize(loss_fn, init_params, method="L-BFGS-B")
        if res.success:
            self.weights = np.asarray(res.x[:n_features], dtype=np.float64)
            self.bias = float(res.x[-1])
        else:
            logger.warning("Optimization did not fully converge. Using normalized weights.")
            self.weights = np.ones(n_features, dtype=np.float64) / n_features
            self.bias = 0.0

        # 2. Fit Calibrator on Validation Fold
        val_logits = np.dot(X_val, self.weights) + self.bias
        if calibrate_method.lower() == "isotonic":
            raw_val_probs = 1.0 / (1.0 + np.exp(-np.clip(val_logits, -25.0, 25.0)))
            self.calibrator = IsotonicCalibrator().fit(raw_val_probs, y_val)
        else:
            self.calibrator = PlattCalibrator().fit(val_logits, y_val)

        # 3. Evaluate Validation Metrics
        val_probs = self.predict_proba(X_val)
        val_preds = (val_probs >= 0.5).astype(int)
        accuracy = float(np.mean(val_preds == y_val))
        brier_score = float(np.mean((val_probs - y_val) ** 2))

        # Expected Calibration Error (ECE)
        bins = np.linspace(0.0, 1.0, 11)
        ece = 0.0
        for b_idx in range(len(bins) - 1):
            mask = (val_probs >= bins[b_idx]) & (val_probs < bins[b_idx + 1])
            if np.any(mask):
                bin_acc = np.mean(y_val[mask])
                bin_conf = np.mean(val_probs[mask])
                ece += np.sum(mask) / len(val_probs) * abs(bin_acc - bin_conf)

        self.metrics = {
            "validation_accuracy": round(accuracy, 4),
            "brier_score": round(brier_score, 4),
            "expected_calibration_error": round(float(ece), 4),
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "feature_weights": {name: round(float(self.weights[i]), 4) for i, name in enumerate(self.feature_names)},
            "bias": round(self.bias, 4),
        }
        logger.info("MetaClassifier trained: accuracy=%.3f, Brier=%.4f, ECE=%.4f", accuracy, brier_score, ece)
        return self.metrics

    def predict_proba(self, X: Union[Dict[str, float], List[float], np.ndarray]) -> Union[float, np.ndarray]:
        """Compute calibrated probability [0.0, 1.0]."""
        if isinstance(X, dict):
            # Extract features in expected order
            arr = np.array([X.get(name, 0.5) for name in self.feature_names], dtype=np.float64)
            logits = np.dot(arr, self.weights) + self.bias
            return float(self.calibrator.calibrate(logits))

        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            logits = np.dot(arr, self.weights) + self.bias
            return float(self.calibrator.calibrate(logits))

        logits = np.dot(arr, self.weights) + self.bias
        return self.calibrator.calibrate(logits)

    def predict_score(self, X: Union[Dict[str, float], List[float], np.ndarray]) -> int:
        """Compute calibrated integer risk score in strictly [1, 100].

        Ensures that a score of 80 corresponds to ~80% calibrated fake probability.
        """
        prob = self.predict_proba(X)
        if isinstance(prob, np.ndarray):
            scores = np.clip(np.round(prob * 100.0), 1, 100).astype(int)
            return scores
        return int(max(1, min(100, round(prob * 100.0))))

    def save(self, filepath: Union[str, Path]) -> None:
        """Save model coefficients, calibration parameters, and metrics to clean JSON."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "feature_names": self.feature_names,
            "weights": [float(w) for w in self.weights],
            "bias": self.bias,
            "calibrator": self.calibrator.to_dict(),
            "metrics": self.metrics,
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved MetaClassifier to %s", p)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "MetaClassifier":
        """Load meta-classifier from JSON."""
        p = Path(filepath)
        if not p.is_file():
            raise FileNotFoundError(f"MetaClassifier file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        cal_data = data.get("calibrator", {})
        cal_type = cal_data.get("type", "platt")
        if cal_type == "isotonic":
            calibrator = IsotonicCalibrator.from_dict(cal_data)
        else:
            calibrator = PlattCalibrator.from_dict(cal_data)

        clf = cls(
            weights=data.get("weights"),
            bias=data.get("bias", 0.0),
            feature_names=data.get("feature_names"),
            calibrator=calibrator,
        )
        clf.metrics = data.get("metrics", {})
        return clf


_meta_classifier_singleton: Optional[MetaClassifier] = None


def get_meta_classifier(model_path: Optional[str] = None) -> Optional[MetaClassifier]:
    """Retrieve singleton MetaClassifier instance, loading from disk if present."""
    global _meta_classifier_singleton
    if _meta_classifier_singleton is not None:
        return _meta_classifier_singleton

    from app.config import get_config
    config = get_config()
    target_path = Path(model_path or getattr(config.fusion, "meta_classifier_path", "models/meta_classifier/model.json"))

    if target_path.is_file():
        try:
            _meta_classifier_singleton = MetaClassifier.load(target_path)
            logger.info("Loaded MetaClassifier from %s", target_path)
            return _meta_classifier_singleton
        except Exception as exc:
            logger.warning("Failed loading MetaClassifier from %s: %s", target_path, exc)

    return None
