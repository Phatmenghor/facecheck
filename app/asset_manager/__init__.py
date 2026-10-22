"""Asset and dataset management package for FaceCheck API."""

from app.asset_manager.manager import AssetManager, get_asset_manager
from app.asset_manager.manifest import MasterAssetManifest, AssetEntry, APPROVED_ASSET_REGISTRY
from app.asset_manager.security import calculate_sha256, check_free_disk_space, safe_extract_zip, safe_extract_tar

__all__ = [
    "AssetManager",
    "get_asset_manager",
    "MasterAssetManifest",
    "AssetEntry",
    "APPROVED_ASSET_REGISTRY",
    "calculate_sha256",
    "check_free_disk_space",
    "safe_extract_zip",
    "safe_extract_tar",
]
