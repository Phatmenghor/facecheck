"""Standalone CLI runner for FaceCheck Continuous Autonomous Learning Daemon.

Usage:
    py scripts/run_continuous_daemon.py

Reads configuration directly from .env (Database: PostgreSQL 192.168.103.106:5432/llm_face).
Runs forever until Ctrl+C.
"""

from __future__ import annotations

import logging
import signal
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.continuous_daemon import get_continuous_daemon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_continuous_daemon")


def main():
    daemon = get_continuous_daemon()

    def handle_sigint(signum, frame):
        logger.info("Termination signal received. Stopping continuous daemon...")
        daemon.stop()

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    daemon.run_forever()


if __name__ == "__main__":
    main()
