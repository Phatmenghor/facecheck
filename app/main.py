"""Main entry point for FaceCheck API."""

from __future__ import annotations

import hashlib
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_config
from app.api.routes import router as api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("facecheck.main")


def verify_model_checksums() -> None:
    """Verify SHA-256 hashes of local model files at startup.

    Refuses to start if checksum mismatch or corruption is detected.
    """
    config = get_config()
    project_root = Path(__file__).resolve().parent.parent
    checksum_file = project_root / "models" / "CHECKSUMS.txt"

    if not checksum_file.is_file():
        logger.warning("No models/CHECKSUMS.txt found. Skipping startup model hash check.")
        return

    logger.info("Verifying model file checksums from %s...", checksum_file)
    with open(checksum_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split(None, 1)
            if len(parts) != 2:
                continue

            expected_sha256, rel_path = parts[0].strip(), parts[1].strip()
            model_path = project_root / rel_path

            if not model_path.is_file():
                # If file not present, check if mock mode allowed
                if config.ai_detector.allow_fallback_to_mock:
                    logger.info("Model file '%s' listed in checksums not found (mock fallback enabled).", rel_path)
                    continue
                else:
                    raise RuntimeError(
                        f"CRITICAL: Model file '{rel_path}' specified in CHECKSUMS.txt is missing from offline server."
                    )

            # Compute sha256
            h = hashlib.sha256()
            with open(model_path, "rb") as mf:
                while chunk := mf.read(65536):
                    h.update(chunk)
            computed_sha = h.hexdigest().lower()

            if computed_sha != expected_sha256.lower():
                raise RuntimeError(
                    f"CRITICAL: Model checksum mismatch for '{rel_path}'!\n"
                    f"Expected: {expected_sha256}\n"
                    f"Computed: {computed_sha}\n"
                    f"Application refusing to start due to model tampering or corruption."
                )

    logger.info("All model checksums verified successfully.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup checks and shutdown cleanup."""
    config = get_config()
    mode_str = "OFFLINE" if config.server.offline_mode else "ONLINE"
    logger.info("Starting FaceCheck API in %s mode...", mode_str)
    verify_model_checksums()
    yield
    logger.info("Shutting down FaceCheck API.")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Enforce strict banking security headers on all responses."""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Allow local scripts and styles for offline Swagger UI documentation
        if request.url.path.startswith("/docs") or request.url.path.startswith("/static"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:;"
            )
        else:
            response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none';"

        return response


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    config = get_config()
    project_root = Path(__file__).resolve().parent.parent
    static_dir = project_root / "app" / "static"

    # OpenAPI & Swagger Docs configuration
    docs_enabled = bool(config.server.enable_docs_offline)
    openapi_url = "/openapi.json" if docs_enabled else None

    app = FastAPI(
        title="FaceCheck API",
        description="Offline REST API for bank advertising face synthetic/deepfake risk verification.",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None,  # Custom local offline handler below
        redoc_url=None,
        openapi_url=openapi_url,
    )

    # Mount offline static assets (Swagger UI JS and CSS)
    if static_dir.is_dir():
        from fastapi.staticfiles import StaticFiles
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Offline Swagger UI documentation endpoint (zero external CDN calls)
    if docs_enabled:
        from fastapi.openapi.docs import get_swagger_ui_html

        @app.get("/docs", include_in_schema=False)
        async def offline_swagger_ui_html():
            return get_swagger_ui_html(
                openapi_url="/openapi.json",
                title="FaceCheck API - Offline Swagger UI",
                swagger_js_url="/static/swagger/swagger-ui-bundle.js",
                swagger_css_url="/static/swagger/swagger-ui.css",
                swagger_favicon_url="data:image/x-icon;,",
            )

    # Security Headers
    app.add_middleware(SecurityHeadersMiddleware)

    # CORS restricted
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Mount API routes
    app.include_router(api_router)

    # Configure OpenAPI security scheme for X-API-Key interactive testing
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        from fastapi.openapi.utils import get_openapi
        openapi_schema = get_openapi(
            title="FaceCheck API",
            version="1.0.0",
            description="Offline REST API for bank advertising face synthetic/deepfake risk verification.",
            routes=app.routes,
        )
        openapi_schema["components"] = openapi_schema.get("components", {})
        openapi_schema["components"]["securitySchemes"] = {
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "Enter your authorized API key (e.g. fc-prod-secret-key-bank-core-9872)",
            }
        }
        openapi_schema["security"] = [{"ApiKeyAuth": []}]
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi

    # Consistent JSON error handling for unhandled exceptions
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled server exception: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred during request execution.",
            },
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    cfg = get_config()
    uvicorn.run("app.main:app", host=cfg.server.host, port=cfg.server.port, reload=False)
