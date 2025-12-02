"""Helpers for capturing reproducibility metadata inside science runs."""

from __future__ import annotations

import importlib
import logging
import multiprocessing
import os
import platform
import shlex
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

KEY_PACKAGES = [
    "reporting_system",
    "cosmos2_science_runner",
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "astropy",
    "pydantic",
    "psutil",
]


def gather_run_environment() -> Dict[str, Any]:
    """Capture git/python/package/cpu/gpu details plus the invoking CLI."""

    env: Dict[str, Any] = {}
    git_info = _get_git_info()
    if git_info:
        env["git"] = git_info
    python_info = _get_python_info()
    if python_info:
        env["python"] = python_info
    packages = _get_package_versions(KEY_PACKAGES)
    if packages:
        env["packages"] = packages
    blas_backend = _detect_blas_backend()
    if blas_backend:
        env["blas_backend"] = blas_backend
    cpu_info = _collect_cpu_info()
    if cpu_info:
        env["cpu"] = cpu_info
    gpu_info = _detect_gpus()
    if gpu_info:
        env["gpu"] = gpu_info
    cli_command = _format_cli_command()
    if cli_command:
        env["cli_command"] = cli_command
    return env


def _get_python_info() -> Dict[str, Any]:
    return {
        "version": platform.python_version(),
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
    }


def _get_git_info() -> Dict[str, Any] | None:
    repo = _repo_root()
    if repo is None:
        return None
    commit = None
    dirty_files: List[str] = []
    dirty = False
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        commit = result.stdout.strip()
    except Exception:
        return None
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        dirty_files = [line.strip() for line in status.stdout.splitlines() if line.strip()]
        dirty = bool(dirty_files)
    except Exception:
        dirty = False
    info: Dict[str, Any] = {"commit": commit, "dirty": dirty}
    if dirty_files:
        info["dirty_files"] = dirty_files
    return info


def _repo_root() -> Path | None:
    try:
        return Path(__file__).resolve().parents[2]
    except Exception:
        return None


def _format_cli_command() -> str | None:
    if not sys.argv:
        return None
    try:
        return shlex.join(sys.argv)
    except Exception:
        return " ".join(sys.argv)


def _get_package_versions(package_names: List[str]) -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for name in package_names:
        version = _resolve_package_version(name)
        if version:
            versions[name] = version
    return versions


def _resolve_package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        pass
    except Exception:
        pass
    try:
        module = importlib.import_module(name)
    except ImportError:
        return None
    version = getattr(module, "__version__", None)
    if isinstance(version, str) and version:
        return version
    return None


def _detect_blas_backend() -> str | None:
    try:
        import numpy as np
    except ImportError:
        return None
    infos = [
        np.__config__.get_info("blas_opt"),
        np.__config__.get_info("openblas_info"),
        np.__config__.get_info("mkl_info"),
    ]
    backend_info = next((info for info in infos if info), {})
    libs = backend_info.get("libraries") or []
    lib_list = ", ".join(sorted(set(libs))) if libs else None
    vendor = "unknown"
    lib_words = " ".join(libs).lower()
    if "openblas" in lib_words:
        vendor = "OpenBLAS"
    elif "mkl" in lib_words or "intel" in lib_words:
        vendor = "Intel MKL"
    elif "atlas" in lib_words:
        vendor = "ATLAS"
    description = vendor
    if lib_list:
        description = f"{description} (libs: {lib_list})"
    return description


def _collect_cpu_info() -> Dict[str, Any]:
    uname = platform.uname()
    logical = os.cpu_count()
    physical = multiprocessing.cpu_count()
    info: Dict[str, Any] = {
        "node": uname.node,
        "platform": f"{uname.system} {uname.release}",
        "processor": uname.processor or uname.machine,
        "cores_physical": physical,
        "cores_logical": logical,
    }
    return info


def _detect_gpus() -> Dict[str, Any] | None:
    devices = _query_nvidia_smi()
    if devices:
        return {"available": True, "source": "nvidia-smi", "devices": devices}
    devices = _query_torch_cuda()
    if devices:
        return {"available": True, "source": "torch.cuda", "devices": devices}
    return {"available": False, "reason": "nvidia-smi/torch.cuda not available"}


def _query_nvidia_smi() -> List[str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    devices: List[str] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [segment.strip() for segment in line.split(",")]
        if len(parts) >= 4:
            idx, name, memory, driver = parts[:4]
            devices.append(f"GPU {idx}: {name} (driver {driver}, mem {memory} MiB)")
        else:
            devices.append(line.strip())
    return devices


def _query_torch_cuda() -> List[str]:
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    devices: List[str] = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        devices.append(f"GPU {idx}: {props.name} ({props.total_memory // (1024 ** 2)} MiB)")
    return devices
