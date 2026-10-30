"""Configuration management for FaceCheck API.

Loads settings from config.yaml and applies environment overrides.
Guarantees 100% offline runtime settings.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import dotenv
import yaml
from pydantic import BaseModel, Field

# Load .env file at the project root if present
PROJECT_ROOT = Path(__file__).resolve().parent.parent
dotenv.load_dotenv(PROJECT_ROOT / ".env")


class DatabaseConfig(BaseModel):
    engine: str = "postgresql"
    host: str = "127.0.0.1"
    port: int = 5432
    dbname: str = "llm_face"
    username: str = "postgres"
    password: str = ""
    connect_timeout: int = 5
    auto_migrate: bool = True
    pool_size: int = 10


class LLMConfig(BaseModel):
    provider: str = "gemini"  # "gemini", "openai", "custom_openai_compatible"
    api_key: str = ""
    model: str = "gemini-1.5-flash"
    base_url: Optional[str] = None
    temperature: float = 0.2
    auto_review_enabled: bool = True
    auto_feedback_enabled: bool = True
    online_training_url: Optional[str] = None
    openai_fine_tuning_enabled: bool = True
    online_stream_enabled: bool = True
    online_stream_batch_size: int = 2


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    environment: str = "production"
    offline_mode: bool = True
    enable_docs_offline: bool = True


class ApiKeyEntry(BaseModel):
    name: str
    key: str


class SecurityConfig(BaseModel):
    api_keys: List[ApiKeyEntry] = Field(default_factory=list)
    rate_limit_per_minute: int = 120


class StorageConfig(BaseModel):
    max_upload_size_mb: int = 10
    allowed_extensions: List[str] = Field(default_factory=lambda: ["jpg", "jpeg", "png", "webp"])
    save_encrypted_images: bool = False
    encrypted_dir: str = "storage/encrypted_images"
    encryption_key_env: str = "FACECHECK_ENCRYPTION_KEY"
    encryption_key_file: str = "storage/master.key"


class SyslogConfig(BaseModel):
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 514
    protocol: str = "udp"


class AuditConfig(BaseModel):
    log_file: str = "logs/audit.log"
    enabled: bool = True
    use_hash_chain: bool = True
    syslog: SyslogConfig = Field(default_factory=SyslogConfig)


class FaceCheckConfig(BaseModel):
    enabled: bool = True
    min_face_size: int = 48
    max_yaw_degrees: float = 35.0
    max_pitch_degrees: float = 25.0
    max_roll_degrees: float = 25.0
    boundary_margin_px: int = 5


class FaceDetectionConfig(BaseModel):
    model_path: str = "models/face_detection_yunet_2023mar.onnx"
    score_threshold: float = 0.6
    nms_threshold: float = 0.3
    top_k: int = 50
    crop_margin_ratio: float = 0.25


class AiDetectorConfig(BaseModel):
    active_detector: str = "huggingface"  # "huggingface", "onnx", "mock"
    model_dir: str = "models/deepfake_detector"
    onnx_model_path: str = "models/deepfake_detector.onnx"
    device: str = "cpu"
    strict_local_files_only: bool = True
    allow_fallback_to_mock: bool = True


class FusionWeights(BaseModel):
    metadata: float = 0.25
    detector_face: float = 0.50
    detector_full_image: float = 0.25


class FusionThresholds(BaseModel):
    likely_real_max: int = 30  # 1 to 30: LIKELY_REAL
    uncertain_max: int = 70    # 31 to 70: UNCERTAIN
    # 71 to 100: LIKELY_FAKE


class FusionConfig(BaseModel):
    weights: FusionWeights = Field(default_factory=FusionWeights)
    thresholds: FusionThresholds = Field(default_factory=FusionThresholds)


class QualityConfig(BaseModel):
    min_width: int = 128
    min_height: int = 128
    min_face_size: int = 48
    blur_threshold: float = 50.0
    dark_luminance_threshold: float = 25.0
    bright_luminance_threshold: float = 235.0
    heavy_compression_ratio: float = 0.012


class HardeningConfig(BaseModel):
    max_image_pixels: int = 25_000_000
    max_dimension: int = 8192
    sandbox_timeout_seconds: float = 15.0
    reject_polyglots: bool = True
    reject_trailing_data: bool = True


class EvalMinAcceptableConfig(BaseModel):
    max_false_positive_rate: float = 0.05
    max_false_negative_rate: float = 0.05
    min_accuracy: float = 0.90


class EvalConfig(BaseModel):
    data_dir: str = "eval/data"
    reports_dir: str = "eval/reports"
    min_acceptable: EvalMinAcceptableConfig = Field(default_factory=EvalMinAcceptableConfig)


class MonitoringConfig(BaseModel):
    metrics_enabled: bool = True


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    face_check: FaceCheckConfig = Field(default_factory=FaceCheckConfig)
    face_detection: FaceDetectionConfig = Field(default_factory=FaceDetectionConfig)
    ai_detector: AiDetectorConfig = Field(default_factory=AiDetectorConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    hardening: HardeningConfig = Field(default_factory=HardeningConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_project_path(path_str: str) -> str:
    """Resolve a relative path against the facecheck_api project root."""
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load config.yaml with environment variable overrides and enforce offline invariants."""
    candidates = [
        config_path,
        os.getenv("FACECHECK_CONFIG"),
        PROJECT_ROOT / "config.yaml",
        "config.yaml",
    ]

    selected: Optional[Path] = None
    for cand in candidates:
        if cand:
            p = Path(cand)
            if p.is_file():
                selected = p
                break

    data = {}
    if selected:
        with open(selected, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    config = AppConfig(**data)

    # Resolve all relative paths strictly to facecheck_api directory
    config.storage.encrypted_dir = _resolve_project_path(config.storage.encrypted_dir)
    config.storage.encryption_key_file = _resolve_project_path(config.storage.encryption_key_file)
    config.audit.log_file = _resolve_project_path(config.audit.log_file)
    config.face_detection.model_path = _resolve_project_path(config.face_detection.model_path)
    config.ai_detector.model_dir = _resolve_project_path(config.ai_detector.model_dir)
    config.ai_detector.onnx_model_path = _resolve_project_path(config.ai_detector.onnx_model_path)
    config.eval.data_dir = _resolve_project_path(config.eval.data_dir)
    config.eval.reports_dir = _resolve_project_path(config.eval.reports_dir)

    # Environment overrides
    if os.getenv("FACECHECK_SERVER_HOST"):
        config.server.host = os.environ["FACECHECK_SERVER_HOST"]
    if os.getenv("FACECHECK_SERVER_PORT"):
        config.server.port = int(os.environ["FACECHECK_SERVER_PORT"])
    if os.getenv("FACECHECK_DEVICE"):
        config.ai_detector.device = os.environ["FACECHECK_DEVICE"]
    if os.getenv("FACECHECK_ACTIVE_DETECTOR"):
        config.ai_detector.active_detector = os.environ["FACECHECK_ACTIVE_DETECTOR"]
    if os.getenv("FACECHECK_SAVE_ENCRYPTED_IMAGES"):
        config.storage.save_encrypted_images = os.environ["FACECHECK_SAVE_ENCRYPTED_IMAGES"].lower() in ("true", "1")

    # Database environment overrides
    if os.getenv("FACECHECK_DB_ENGINE"):
        config.database.engine = os.environ["FACECHECK_DB_ENGINE"]
    if os.getenv("FACECHECK_DB_HOST"):
        config.database.host = os.environ["FACECHECK_DB_HOST"]
    if os.getenv("FACECHECK_DB_PORT"):
        config.database.port = int(os.environ["FACECHECK_DB_PORT"])
    if os.getenv("FACECHECK_DB_NAME"):
        config.database.dbname = os.environ["FACECHECK_DB_NAME"]
    if os.getenv("FACECHECK_DB_USER"):
        config.database.username = os.environ["FACECHECK_DB_USER"]
    if os.getenv("FACECHECK_DB_PASSWORD"):
        config.database.password = os.environ["FACECHECK_DB_PASSWORD"]
    if os.getenv("FACECHECK_DB_CONNECT_TIMEOUT"):
        config.database.connect_timeout = int(os.environ["FACECHECK_DB_CONNECT_TIMEOUT"])

    # API keys from environment variables
    single_key = os.getenv("FACECHECK_API_KEY")
    if single_key and not any(k.key == single_key for k in config.security.api_keys):
        config.security.api_keys.append(ApiKeyEntry(name="env_api_key", key=single_key))

    env_keys = os.getenv("FACECHECK_API_KEYS")
    if env_keys:
        parsed_keys: List[ApiKeyEntry] = []
        for item in env_keys.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" in item:
                name, key_val = item.split(":", 1)
                parsed_keys.append(ApiKeyEntry(name=name.strip(), key=key_val.strip()))
            else:
                parsed_keys.append(ApiKeyEntry(name=f"key_{len(parsed_keys)+1}", key=item))
        if parsed_keys:
            config.security.api_keys = parsed_keys

    # LLM & Online AI Model environment overrides
    if os.getenv("LLM_PROVIDER"):
        config.llm.provider = os.environ["LLM_PROVIDER"].lower()
    if os.getenv("GEMINI_API_KEY"):
        config.llm.api_key = os.environ["GEMINI_API_KEY"]
        config.llm.provider = "gemini"
    elif os.getenv("OPENAI_API_KEY"):
        config.llm.api_key = os.environ["OPENAI_API_KEY"]
        config.llm.provider = "openai"
    elif os.getenv("LLM_API_KEY"):
        config.llm.api_key = os.environ["LLM_API_KEY"]
    if os.getenv("LLM_MODEL"):
        config.llm.model = os.environ["LLM_MODEL"]
    if os.getenv("LLM_BASE_URL"):
        config.llm.base_url = os.environ["LLM_BASE_URL"]
    if os.getenv("LLM_TEMPERATURE"):
        config.llm.temperature = float(os.environ["LLM_TEMPERATURE"])
    if os.getenv("LLM_AUTO_REVIEW_ENABLED"):
        config.llm.auto_review_enabled = os.environ["LLM_AUTO_REVIEW_ENABLED"].lower() in ("true", "1")
    if os.getenv("LLM_AUTO_FEEDBACK_ENABLED"):
        config.llm.auto_feedback_enabled = os.environ["LLM_AUTO_FEEDBACK_ENABLED"].lower() in ("true", "1")
    if os.getenv("LLM_OPENAI_FINE_TUNING_ENABLED"):
        config.llm.openai_fine_tuning_enabled = os.environ["LLM_OPENAI_FINE_TUNING_ENABLED"].lower() in ("true", "1")
    if os.getenv("LLM_ONLINE_TRAINING_URL"):
        config.llm.online_training_url = os.environ["LLM_ONLINE_TRAINING_URL"]
    if os.getenv("FACECHECK_ONLINE_STREAM_ENABLED"):
        config.llm.online_stream_enabled = os.environ["FACECHECK_ONLINE_STREAM_ENABLED"].lower() in ("true", "1")
    if os.getenv("FACECHECK_ONLINE_STREAM_BATCH_SIZE"):
        config.llm.online_stream_batch_size = int(os.environ["FACECHECK_ONLINE_STREAM_BATCH_SIZE"])

    # Enforce offline variables
    if config.server.offline_mode:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["NO_PROXY"] = "*"

    return config


@lru_cache()
def get_config() -> AppConfig:
    """Cached accessor for application configuration."""
    return load_config()
