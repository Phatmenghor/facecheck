"""Thread-safe Prometheus metrics collector for FaceCheck API.

Exposes metrics in Prometheus exposition format without external dependencies:
- facecheck_requests_total{status, api_key}
- facecheck_errors_total{error_type}
- facecheck_processing_duration_seconds (count, sum, buckets)
- facecheck_scores_total{range}
- facecheck_low_quality_total
- facecheck_no_face_found_total
"""

from __future__ import annotations

import time
from threading import Lock
from typing import Dict, List, Optional


class MetricsCollector:
    """Thread-safe metric store generating standard Prometheus plaintext output."""

    def __init__(self):
        self._lock = Lock()
        # Requests counter: key = (status_code, api_key_name) -> count
        self._requests: Dict[tuple[int, str], int] = {}
        # Errors counter: key = error_type -> count
        self._errors: Dict[str, int] = {}
        # Latency summary
        self._duration_count: int = 0
        self._duration_sum: float = 0.0
        # Latency buckets (in seconds): 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
        self._duration_buckets: Dict[float, int] = {
            0.05: 0,
            0.1: 0,
            0.25: 0,
            0.5: 0,
            1.0: 0,
            2.5: 0,
            5.0: 0,
            10.0: 0,
        }
        # Score distribution buckets: 1-30 (likely real), 31-70 (uncertain), 71-100 (likely fake)
        self._scores: Dict[str, int] = {
            "1-30": 0,
            "31-70": 0,
            "71-100": 0,
        }
        self._low_quality_count: int = 0
        self._no_face_count: int = 0

    def record_request(self, status_code: int, api_key: str, duration_sec: float) -> None:
        with self._lock:
            key = (status_code, api_key)
            self._requests[key] = self._requests.get(key, 0) + 1

            self._duration_count += 1
            self._duration_sum += duration_sec

            for bucket in sorted(self._duration_buckets.keys()):
                if duration_sec <= bucket:
                    self._duration_buckets[bucket] += 1

    def record_error(self, error_type: str) -> None:
        with self._lock:
            self._errors[error_type] = self._errors.get(error_type, 0) + 1

    def record_score(self, score: Optional[int]) -> None:
        with self._lock:
            if score is None:
                self._no_face_count += 1
            elif score <= 30:
                self._scores["1-30"] += 1
            elif score <= 70:
                self._scores["31-70"] += 1
            else:
                self._scores["71-100"] += 1

    def record_low_quality(self) -> None:
        with self._lock:
            self._low_quality_count += 1

    def record_no_face(self) -> None:
        with self._lock:
            self._no_face_count += 1

    def export_metrics_text(self) -> str:
        """Render metrics in Prometheus text exposition format."""
        lines: List[str] = []

        with self._lock:
            # 1. Total Requests
            lines.append("# HELP facecheck_requests_total Total number of API requests processed")
            lines.append("# TYPE facecheck_requests_total counter")
            if not self._requests:
                lines.append('facecheck_requests_total{status="200",api_key="none"} 0')
            for (status_code, key_name), count in sorted(self._requests.items()):
                lines.append(f'facecheck_requests_total{{status="{status_code}",api_key="{key_name}"}} {count}')

            # 2. Total Errors
            lines.append("# HELP facecheck_errors_total Total number of application errors")
            lines.append("# TYPE facecheck_errors_total counter")
            if not self._errors:
                lines.append('facecheck_errors_total{error_type="none"} 0')
            for err_type, count in sorted(self._errors.items()):
                lines.append(f'facecheck_errors_total{{error_type="{err_type}"}} {count}')

            # 3. Request Processing Duration
            lines.append("# HELP facecheck_processing_duration_seconds Processing latency histogram")
            lines.append("# TYPE facecheck_processing_duration_seconds histogram")
            cum_count = 0
            for le, count in sorted(self._duration_buckets.items()):
                cum_count += count
                lines.append(f'facecheck_processing_duration_seconds_bucket{{le="{le}"}} {cum_count}')
            lines.append(f'facecheck_processing_duration_seconds_bucket{{le="+Inf"}} {self._duration_count}')
            lines.append(f"facecheck_processing_duration_seconds_sum {self._duration_sum:.4f}")
            lines.append(f"facecheck_processing_duration_seconds_count {self._duration_count}")

            # 4. Score Buckets
            lines.append("# HELP facecheck_scores_total Distribution of returned risk scores")
            lines.append("# TYPE facecheck_scores_total counter")
            for rng, count in sorted(self._scores.items()):
                lines.append(f'facecheck_scores_total{{range="{rng}"}} {count}')

            # 5. Low Quality Images
            lines.append("# HELP facecheck_low_quality_total Images flagged with low quality warnings")
            lines.append("# TYPE facecheck_low_quality_total counter")
            lines.append(f"facecheck_low_quality_total {self._low_quality_count}")

            # 6. No Face Found
            lines.append("# HELP facecheck_no_face_found_total Images where no human face was detected")
            lines.append("# TYPE facecheck_no_face_found_total counter")
            lines.append(f"facecheck_no_face_found_total {self._no_face_count}")

        return "\n".join(lines) + "\n"


_metrics_singleton: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    global _metrics_singleton
    if _metrics_singleton is None:
        _metrics_singleton = MetricsCollector()
    return _metrics_singleton
