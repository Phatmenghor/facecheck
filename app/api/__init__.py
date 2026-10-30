"""API package export with component routers."""

from app.api.routes import router
from app.api.check_router import router as check_router
from app.api.match_router import router as match_router
from app.api.pad_router import router as pad_router
from app.api.learning_router import router as learning_router

__all__ = [
    "router",
    "check_router",
    "match_router",
    "pad_router",
    "learning_router",
]
