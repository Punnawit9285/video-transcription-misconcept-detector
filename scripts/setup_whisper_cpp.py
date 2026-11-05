"""Download a prebuilt whisper.cpp (AMD Vulkan / ROCm / CPU) into tools/whisper-cpp/<variant>/.

Builds come from https://github.com/lemonade-sdk/whisper.cpp-rocm (AMD's Lemonade project),
which publishes Windows and Linux binaries for Vulkan and for ROCm per GPU family:

  vulkan        any GPU with a Vulkan driver (AMD Radeon RX 9070 XT, also NVIDIA)  ~20 MB
  rocm-gfx120X  ROCm build for RDNA4 (RX 9070 XT / 9070 / 9060 XT)                 ~500 MB
  rocm-gfx110X  ROCm build for RDNA3 (RX 7900 / 7800 / 7600 ...)                    ~180 MB
  cpu           CPU only                                                            ~4 MB

Usage:
  python scripts/setup_whisper_cpp.py --variant vulkan --model large-v3
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO = "lemonade-sdk/whisper.cpp-rocm"
ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "tools" / "whisper-cpp"
VARIANTS = ["vulkan", "rocm-gfx120X", "rocm-gfx110X", "cpu"]


def asset_suffix(variant: str) -> str:
    if platform.system() == "Windows":
        return {"vulkan": "-windows-vulkan-x64.zip", "cpu": "-windows-cpu-x64.zip"}.get(
            variant, f"-windows-{variant}.zip")
    return {"vulkan": "-linux-vulkan-x86_64.tar.gz", "cpu": "-linux-cpu-x86_64.tar.gz"}.get(
        variant, f"-linux-{variant}.tar.gz")


def find_asset(variant: str, tag: str | None) -> tuple[str, str, int]:
    url = f"https://api.github.com/repos/{REPO}/releases" + (f"/tags/{tag}" if tag else "?per_page=10")
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                               "User-Agent": "lecture-notes-setup"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    releases = [data] if tag else data
    suffix = asset_suffix(variant)
    for rel in releases:
        for a in rel.get("assets", []):
            if a["name"].endswith(suffix) and a["name"].startswith("whisper-"):
                return rel["tag_name"], a["browser_download_url"], a["size"]
    raise SystemExit(f"No '{suffix}' asset found in {REPO} releases - check {url}")


def download(url: str, dst: Path, size: int) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "lecture-notes-setup"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as f:
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            print(f"\r  {done / 1e6:7.1f} / {size / 1e6:.1f} MB", end="", flush=True)
    print()


def extract(archive: Path, target: Path) -> None:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)
    else:
        with tarfile.open(archive) as t:
            t.extractall(target, filter="data")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=VARIANTS, default="vulkan")
    p.add_argument("--tag", help="release tag (default: newest release that has the build)")
    p.add_argument("--model", help="also pre-download this ggml model, e.g. large-v3 or large-v3-q5_0")
    a = p.parse_args()

    tag, url, size = find_asset(a.variant, a.tag)
    target = DEST / a.variant
    print(f"whisper.cpp {tag} [{a.variant}]\n  {url}\n  -> {target}")
    with tempfile.TemporaryDirectory() as td:
        archive = Path(td) / url.rsplit("/", 1)[1]
        download(url, archive, size)
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        extract(archive, target)

    exe = "whisper-cli.exe" if platform.system() == "Windows" else "whisper-cli"
    cli = next(iter(sorted(target.rglob(exe))), None)
    if not cli:
        print(f"Warning: {exe} not found in the archive - see {target}", file=sys.stderr)
        return 1
    if platform.system() != "Windows":
        cli.chmod(0o755)
    print(f"Installed: {cli}")

    if a.model:
        sys.path.insert(0, str(ROOT))
        from huggingface_hub import hf_hub_download

        from lecture_notes.models import VAD_MODEL, WHISPER_CPP

        for spec in (WHISPER_CPP[a.model].source, VAD_MODEL):
            repo, name = spec.split("::", 1)
            print(f"Downloading {name} ...")
            print("  ->", hf_hub_download(repo, name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
