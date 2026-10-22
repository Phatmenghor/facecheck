"""Convenience runner script for FaceCheck API.

Ensures the project directory is on sys.path regardless of where it is invoked from.
"""

import sys
from pathlib import Path

# Add project root to sys.path so 'app' can always be imported
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    import uvicorn
    from app.config import get_config

    cfg = get_config()
    print("=" * 60)
    print("Starting FaceCheck API (Bank Biometric Risk Service)...")
    print(f"Server URL:    http://{cfg.server.host}:{cfg.server.port}")
    print(f"Swagger Docs:  http://127.0.0.1:{cfg.server.port}/docs")
    print(f"Default Key:   fc-prod-secret-key-bank-core-9872")
    print("=" * 60)

    uvicorn.run(
        "app.main:app",
        host=cfg.server.host,
        port=cfg.server.port,
        reload=True,
    )
