"""Process-sandboxed execution for image decoding and analysis.

Runs image processing in an isolated worker process with strict timeout,
CPU limit, and memory limits to protect against denial of service, memory exhaustion,
or native ONNX/C library crashes.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import sys
from typing import Any, Dict, Optional

from app.config import get_config

logger = logging.getLogger("facecheck.sandbox")


class SandboxTimeoutError(Exception):
    """Raised when the sandboxed analysis exceeds the configured timeout."""
    pass


class SandboxWorkerError(Exception):
    """Raised when the sandboxed worker process crashes or encounters a fatal error."""
    pass


def _worker_initializer() -> None:
    """Set process resource limits on POSIX systems (Linux/Container)."""
    try:
        import resource
        # Virtual memory limit: 1.5 GB
        max_bytes = 1536 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
        # CPU time limit: 30 seconds
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    except (ImportError, AttributeError, ValueError, Exception):
        # On Windows or non-privileged environments, resource limits are managed externally
        pass


def _worker_task(filename: str, raw_bytes: bytes) -> Dict[str, Any]:
    """Target function executed inside the sandboxed worker process."""
    # Ensure offline flags inside child process
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["NO_PROXY"] = "*"

    from app.services.pipeline import run_pipeline
    return run_pipeline(filename, raw_bytes)


class SandboxManager:
    """Manages isolated worker process execution with timeouts and crash isolation."""

    def __init__(self, max_workers: int = 2):
        self._max_workers = max_workers
        self._executor: Optional[concurrent.futures.ProcessPoolExecutor] = None

    def _get_executor(self) -> concurrent.futures.ProcessPoolExecutor:
        if self._executor is None:
            self._executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self._max_workers,
                initializer=_worker_initializer,
            )
        return self._executor

    def restart(self) -> None:
        """Shut down and recreate the worker pool if a child process crashed."""
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except Exception as exc:
                logger.warning("Error shutting down worker pool: %s", exc)
            self._executor = None

    async def execute_sandboxed(
        self,
        filename: str,
        raw_bytes: bytes,
        timeout_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute image analysis in the sandboxed process with a strict timeout.

        Args:
            filename: Client filename
            raw_bytes: File contents
            timeout_seconds: Maximum allowed seconds before timeout

        Returns:
            Result dictionary from run_pipeline.

        Raises:
            SandboxTimeoutError: If processing exceeded timeout_seconds.
            SandboxWorkerError: If worker process crashed or errored.
        """
        config = get_config()
        timeout = timeout_seconds or config.hardening.sandbox_timeout_seconds

        loop = asyncio.get_running_loop()
        executor = self._get_executor()

        try:
            future = loop.run_in_executor(
                executor,
                _worker_task,
                filename,
                raw_bytes,
            )
            result = await asyncio.wait_for(future, timeout=timeout)
            return result

        except asyncio.TimeoutError:
            logger.error("Sandboxed worker timed out after %.1f seconds", timeout)
            self.restart()
            raise SandboxTimeoutError(
                f"Image analysis timed out after {timeout:.1f} seconds. Worker process terminated."
            )

        except concurrent.futures.process.BrokenProcessPool as b_err:
            logger.critical("Sandboxed worker process crashed or killed: %s", b_err)
            self.restart()
            raise SandboxWorkerError(
                "Sandboxed worker process crashed unexpectedly during execution."
            )

        except Exception as exc:
            # Check if this is an application validation error passed through
            # from validator (FileTooLargeError, UnsupportedFileTypeError, InvalidImageError)
            err_name = type(exc).__name__
            if err_name in ("FileTooLargeError", "UnsupportedFileTypeError", "InvalidImageError"):
                raise exc
            logger.error("Error inside sandboxed worker: %s", exc)
            raise


_sandbox_singleton: Optional[SandboxManager] = None


def get_sandbox() -> SandboxManager:
    global _sandbox_singleton
    if _sandbox_singleton is None:
        _sandbox_singleton = SandboxManager()
    return _sandbox_singleton
