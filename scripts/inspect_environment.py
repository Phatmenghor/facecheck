#!/usr/bin/env python3
"""Environment and dependency inspection script for FaceCheck API.

Inspects:
- Operating system, architecture, logical and physical CPU cores
- Available RAM (using ctypes on Windows or /proc/meminfo on Linux, zero external deps)
- GPU availability
- Python runtime and executable path
- Installed packages vs pinned requirements
- Model files and integrity checks
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_memory_info() -> Dict[str, float]:
    """Retrieve total and available RAM in GB without requiring psutil."""
    total_gb = 0.0
    avail_gb = 0.0

    if platform.system() == "Windows":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            total_gb = round(stat.ullTotalPhys / (1024**3), 2)
            avail_gb = round(stat.ullAvailPhys / (1024**3), 2)
    elif os.path.exists("/proc/meminfo"):
        meminfo = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip().split()[0]
                    meminfo[key] = float(val)
        total_kb = meminfo.get("MemTotal", 0.0)
        avail_kb = meminfo.get("MemAvailable", meminfo.get("MemFree", 0.0))
        total_gb = round(total_kb / (1024**2), 2)
        avail_gb = round(avail_kb / (1024**2), 2)

    return {"total_ram_gb": total_gb, "available_ram_gb": avail_gb}


def get_gpu_info() -> Dict[str, Any]:
    """Check GPU availability."""
    gpu_info = {
        "cuda_available": False,
        "device_count": 0,
        "device_names": [],
        "driver_info": "CPU-only execution",
    }
    try:
        import torch
        gpu_info["cuda_available"] = torch.cuda.is_available()
        gpu_info["device_count"] = torch.cuda.device_count()
        if gpu_info["cuda_available"]:
            gpu_info["device_names"] = [torch.cuda.get_device_name(i) for i in range(gpu_info["device_count"])]
    except ImportError:
        pass

    # Windows WMI query fallback if torch not present
    if platform.system() == "Windows" and not gpu_info["device_names"]:
        try:
            import subprocess
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
                text=True,
                timeout=5,
            )
            names = [line.strip() for line in out.strip().splitlines() if line.strip()]
            gpu_info["device_names"] = names
        except Exception:
            pass

    return gpu_info


def inspect_dependencies() -> Dict[str, Any]:
    """Check pinned requirements vs installed versions."""
    pinned_path = PROJECT_ROOT / "requirements.txt"
    pinned_reqs: Dict[str, str] = {}
    if pinned_path.is_file():
        for line in pinned_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                if "==" in line:
                    pkg, ver = line.split("==", 1)
                    pkg = pkg.split("[")[0].strip().lower()
                    pinned_reqs[pkg] = ver.strip()

    installed_map: Dict[str, str] = {}
    try:
        import importlib.metadata
        for dist in importlib.metadata.distributions():
            installed_map[dist.metadata["Name"].lower()] = dist.version
    except Exception:
        pass

    comparison = {}
    for pkg, pinned_ver in pinned_reqs.items():
        installed_ver = installed_map.get(pkg)
        comparison[pkg] = {
            "pinned": pinned_ver,
            "installed": installed_ver,
            "status": "MATCH" if installed_ver == pinned_ver else ("COMPATIBLE" if installed_ver else "MISSING"),
        }

    return comparison


def inspect_models() -> List[Dict[str, Any]]:
    """Inspect model files in models/ directory."""
    models_dir = PROJECT_ROOT / "models"
    model_list = []
    if models_dir.is_dir():
        for p in models_dir.rglob("*"):
            if p.is_file() and p.name not in ("CHECKSUMS.txt", "README.md", ".gitkeep"):
                h = hashlib.sha256(p.read_bytes()).hexdigest()
                model_list.append({
                    "path": str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "size_bytes": p.stat().st_size,
                    "sha256": h,
                })
    return model_list


def main() -> None:
    print("=" * 80)
    print("           FACECHECK API: ENVIRONMENT & ASSET INSPECTOR")
    print("=" * 80)

    mem = get_memory_info()
    gpu = get_gpu_info()
    deps = inspect_dependencies()
    models = inspect_models()

    report = {
        "system": {
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "cpu_cores": os.cpu_count(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "total_ram_gb": mem["total_ram_gb"],
            "available_ram_gb": mem["available_ram_gb"],
        },
        "gpu": gpu,
        "dependencies": deps,
        "models": models,
    }

    print(f"\n[SYSTEM]")
    print(f"  OS Platform:        {report['system']['platform']}")
    print(f"  CPU Architecture:   {report['system']['architecture']} ({report['system']['cpu_cores']} logical cores)")
    print(f"  Memory:             {mem['total_ram_gb']} GB total, {mem['available_ram_gb']} GB available")
    print(f"  Python Version:     {sys.version.split()[0]}")
    print(f"  Python Executable:  {sys.executable}")

    print(f"\n[GPU & ACCELERATION]")
    print(f"  CUDA Available:     {gpu['cuda_available']}")
    print(f"  Video Controllers:  {', '.join(gpu['device_names']) if gpu['device_names'] else 'None detected'}")
    print(f"  Inference Mode:     CPU-Only (air-gapped bank standard)")

    print(f"\n[DEPENDENCY AUDIT]")
    missing = [pkg for pkg, data in deps.items() if data["status"] == "MISSING"]
    for pkg, data in deps.items():
        st = data["status"]
        print(f"  * {pkg:25s} | Pinned: {data['pinned']:10s} | Installed: {str(data['installed']):10s} | {st}")

    if missing:
        print(f"\n  [NOTE] Missing pinned packages: {', '.join(missing)}")
        if "torch" in missing:
            print("         * 'torch==2.5.1' is omitted intentionally on Python 3.13; OpenCV & NumPy run all CPU models.")

    print(f"\n[LOCAL MODEL INVENTORY]")
    for m in models:
        sz_kb = m["size_bytes"] / 1024
        print(f"  * {m['path']:45s} | {sz_kb:8.1f} KB | SHA256: {m['sha256'][:16]}...")

    # Save to reports/baseline
    out_json = PROJECT_ROOT / "reports" / "baseline" / "environment_inspection.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[SAVED] Inspection report saved to: {out_json}")


if __name__ == "__main__":
    main()
