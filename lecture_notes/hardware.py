"""GPU detection and NVIDIA runtime setup."""

from __future__ import annotations

import functools
import importlib.util
import os
import platform
import subprocess
from pathlib import Path


@functools.cache
def gpu_names() -> tuple[str, ...]:
    """Names of the display adapters on this machine (best effort)."""
    try:
        if platform.system() == "Windows":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_VideoController).Name"],
                capture_output=True, text=True, timeout=20,
            ).stdout
        else:
            out = subprocess.run(["lspci"], capture_output=True, text=True, timeout=20).stdout
            out = "\n".join(l for l in out.splitlines() if "VGA" in l or "3D" in l or "Display" in l)
    except (OSError, subprocess.SubprocessError):
        return ()
    return tuple(l.strip() for l in out.splitlines() if l.strip())


def has_nvidia() -> bool:
    return any("nvidia" in n.lower() for n in gpu_names())


def has_amd() -> bool:
    return any(k in n.lower() for n in gpu_names() for k in ("amd", "radeon"))


_dll_dirs_added = False


def add_nvidia_dll_dirs() -> None:
    """Make pip-installed NVIDIA runtime DLLs (nvidia-cublas-cu12, ...) loadable on Windows."""
    global _dll_dirs_added
    if _dll_dirs_added or platform.system() != "Windows":
        return
    _dll_dirs_added = True
    for pkg in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_runtime", "nvidia.cuda_nvrtc"):
        try:
            spec = importlib.util.find_spec(pkg)
        except ModuleNotFoundError:
            continue
        for loc in (spec.submodule_search_locations or []) if spec else []:
            bin_dir = Path(loc) / "bin"
            if bin_dir.is_dir():
                os.add_dll_directory(str(bin_dir))
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


@functools.cache
def cuda_status() -> tuple[bool, str]:
    """(usable, reason) for faster-whisper on CUDA."""
    try:
        import ctranslate2
    except ImportError as e:
        return False, f"ctranslate2 not installed ({e})"
    if ctranslate2.get_cuda_device_count() == 0:
        return False, "no CUDA device visible to CTranslate2"
    if platform.system() == "Windows":
        add_nvidia_dll_dirs()
        import ctypes
        try:
            ctypes.WinDLL("cublas64_12.dll")
        except OSError:
            return False, ("cuBLAS 12 not found - run: pip install nvidia-cublas-cu12 "
                           "(or install.bat nvidia)")
    return True, "ok"
