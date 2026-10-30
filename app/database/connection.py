"""PostgreSQL database connection provider for FaceCheck."""

from __future__ import annotations

import logging
import psycopg2
import psycopg2.extras
from typing import Any

from app.config import get_config

logger = logging.getLogger("facecheck.database.connection")


class DatabaseConnectionManager:
    """Manages connections directly to PostgreSQL."""

    def __init__(self) -> None:
        self.config = get_config()

    def get_connection(self) -> psycopg2.extensions.connection:
        """Create a fresh PostgreSQL connection using configured environment credentials."""
        return psycopg2.connect(
            host=self.config.database.host,
            port=self.config.database.port,
            dbname=self.config.database.dbname,
            user=self.config.database.username,
            password=self.config.database.password,
            connect_timeout=self.config.database.connect_timeout,
        )

    def test_connection(self) -> bool:
        """Test whether PostgreSQL is reachable and authenticated."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                    return cur.fetchone()[0] == 1
        except Exception as exc:
            logger.error("PostgreSQL connection check failed: %s", exc)
            return False
