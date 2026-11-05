"""Speech recognition backends.

* faster-whisper (CTranslate2): NVIDIA CUDA or CPU. Fastest on NVIDIA (batched, int8/fp16).
* whisper.cpp: Vulkan or ROCm builds for AMD Radeon (e.g. RX 9070 XT), also CPU.
  Binaries are installed by `scripts/setup_whisper_cpp.py` into tools/whisper-cpp/<variant>/.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import hardware
from .audio import SAMPLE_RATE, load_wav
from .models import DEFAULT_PROMPT, FASTER_WHISPER, VAD_MODEL, WHISPER_CPP

ProgressFn = Callable[[float, str], None]

REPO_ROOT = Path(__file__).resolve().parent.parent
WHISPER_CPP_DIR = Path(os.environ.get("WHISPER_CPP_DIR", REPO_ROOT / "tools" / "whisper-cpp"))

BACKENDS = {
    "auto": "Auto-detect (NVIDIA -> faster-whisper CUDA, AMD -> whisper.cpp Vulkan/ROCm, else CPU)",
    "faster-whisper-cuda": "faster-whisper on NVIDIA CUDA (RTX 3070 Ti etc.)",
    "faster-whisper-cpu": "faster-whisper on CPU (slow: ~2-4x the audio length for large-v3)",
    "whisper.cpp-vulkan": "whisper.cpp on Vulkan (AMD Radeon e.g. RX 9070 XT; also works on NVIDIA)",
    "whisper.cpp-rocm": "whisper.cpp on ROCm (AMD Radeon, RDNA3/RDNA4 builds)",
    "whisper.cpp-cpu": "whisper.cpp on CPU",
}


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class AsrOptions:
    backend: str = "auto"
    model: str = "large-v3"
    language: str = "th"  # "auto" to detect
    prompt: str = DEFAULT_PROMPT
    beam_size: int = 5
    batch_size: int = 8  # faster-whisper CUDA only
    # Batched mode cuts speech into chunks of up to this many seconds. Dense Thai speech
    # overflows Whisper's 448-token window in 30 s chunks and loses sentences; 15 s does not.
    chunk_length: int = 15
    compute_type: str = ""  # faster-whisper; "" = pick per device
    vad: bool = True
    threads: int = 0  # 0 = auto
    gpu_device: int = 0


def whisper_cpp_variants() -> list[str]:
    """whisper.cpp builds installed under tools/whisper-cpp/ (e.g. ['vulkan', 'rocm-gfx120X'])."""
    if not WHISPER_CPP_DIR.is_dir():
        return []
    return sorted(d.name for d in WHISPER_CPP_DIR.iterdir() if d.is_dir() and _find_cli(d))


def _find_cli(root: Path) -> Path | None:
    name = "whisper-cli.exe" if platform.system() == "Windows" else "whisper-cli"
    return next(iter(sorted(root.rglob(name))), None)


def _whisper_cpp_cli(kind: str) -> Path:
    """kind: 'vulkan', 'rocm' or 'cpu'."""
    if env := os.environ.get("WHISPER_CPP_CLI"):
        return Path(env)
    for variant in whisper_cpp_variants():
        if variant.startswith(kind):
            return _find_cli(WHISPER_CPP_DIR / variant)  # type: ignore[return-value]
    raise RuntimeError(
        f"whisper.cpp ({kind}) is not installed. Run:\n"
        f"  python scripts/setup_whisper_cpp.py --variant {kind if kind != 'rocm' else 'rocm-gfx120X'}"
    )


def resolve_backend(backend: str) -> str:
    if backend != "auto":
        return backend
    if hardware.has_nvidia() and hardware.cuda_status()[0]:
        return "faster-whisper-cuda"
    installed = whisper_cpp_variants()
    if hardware.has_amd() or hardware.has_nvidia():
        for kind in ("rocm", "vulkan"):
            if any(v.startswith(kind) for v in installed):
                return f"whisper.cpp-{kind}"
    return "faster-whisper-cpu"


def transcribe(wav_path: str | Path, opts: AsrOptions, progress: ProgressFn) -> tuple[list[Segment], str]:
    """Transcribe a 16 kHz mono WAV. Returns (segments, resolved backend name)."""
    backend = resolve_backend(opts.backend)
    if backend.startswith("faster-whisper"):
        segs = _faster_whisper(wav_path, opts, "cuda" if backend.endswith("cuda") else "cpu", progress)
    elif backend.startswith("whisper.cpp"):
        segs = _whisper_cpp(wav_path, opts, backend.split("-", 1)[1], progress)
    else:
        raise ValueError(f"Unknown backend {backend!r}. Choices: {', '.join(BACKENDS)}")
    return segs, backend


# --------------------------------------------------------------------------- faster-whisper

def _faster_whisper(wav_path, opts: AsrOptions, device: str, progress: ProgressFn) -> list[Segment]:
    if device == "cuda":
        ok, why = hardware.cuda_status()
        if not ok:
            raise RuntimeError(f"CUDA is not usable: {why}")
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    model_id = FASTER_WHISPER[opts.model].source if opts.model in FASTER_WHISPER else opts.model
    compute = opts.compute_type or ("int8_float16" if device == "cuda" else "int8")
    threads = opts.threads or (os.cpu_count() or 4)
    progress(0.0, f"Loading {model_id} on {device} ({compute}) - first run downloads the model")
    model = WhisperModel(model_id, device=device, device_index=opts.gpu_device,
                         compute_type=compute, cpu_threads=threads)

    audio = load_wav(wav_path)
    duration = len(audio) / SAMPLE_RATE
    common = dict(
        language=None if opts.language == "auto" else opts.language,
        beam_size=opts.beam_size,
        initial_prompt=opts.prompt or None,
        vad_filter=opts.vad,
    )
    batch = opts.batch_size if device == "cuda" else 1

    while True:
        try:
            if batch > 1:
                pipe = BatchedInferencePipeline(model)
                it, _ = pipe.transcribe(audio, batch_size=batch, chunk_length=opts.chunk_length,
                                        without_timestamps=False, **common)
            else:
                it, _ = model.transcribe(audio, condition_on_previous_text=False, **common)
            out: list[Segment] = []
            for s in it:
                text = s.text.strip()
                if text:
                    out.append(Segment(round(s.start, 2), round(s.end, 2), text))
                progress(min(s.end / duration, 1.0), f"Transcribed {_hms(s.end)} / {_hms(duration)}")
            return out
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and batch > 1:
                batch //= 2
                progress(0.0, f"GPU out of memory - retrying with batch size {batch}")
                continue
            raise


# --------------------------------------------------------------------------- whisper.cpp

_PROGRESS_RE = re.compile(r"progress\s*=\s*(\d+)%")


def _hf_file(spec: str) -> str:
    from huggingface_hub import hf_hub_download

    repo, filename = spec.split("::", 1)
    return hf_hub_download(repo, filename)


def _whisper_cpp(wav_path, opts: AsrOptions, kind: str, progress: ProgressFn) -> list[Segment]:
    cli = _whisper_cpp_cli(kind)
    if opts.model in WHISPER_CPP:
        progress(0.0, f"Fetching ggml model {opts.model} (first run downloads it)")
        model_path = _hf_file(WHISPER_CPP[opts.model].source)
    elif Path(opts.model).is_file():
        model_path = opts.model
    else:
        raise ValueError(f"whisper.cpp model must be one of {list(WHISPER_CPP)} or a path to a ggml .bin file")

    cpus = os.cpu_count() or 4
    threads = opts.threads or (cpus if kind == "cpu" else min(8, cpus))
    # whisper-cli on Windows reads argv in the ANSI code page, so keep the prompt ASCII-only.
    prompt = re.sub(r"\s+", " ", "".join(c for c in opts.prompt if ord(c) < 128)).strip(" ,")

    with tempfile.TemporaryDirectory(prefix="lecture-notes-") as td:
        out_base = Path(td) / "out"
        cmd = [str(cli), "-m", model_path, "-f", str(wav_path), "-l", opts.language,
               "-t", str(threads), "-bs", str(opts.beam_size), "-mc", "0",
               "-oj", "-of", str(out_base), "-pp"]
        if prompt:
            cmd += ["--prompt", prompt]
        if kind == "cpu":
            cmd += ["-ng"]
        else:
            cmd += ["-dev", str(opts.gpu_device)]
        if opts.vad:
            cmd += ["--vad", "-vm", _hf_file(VAD_MODEL)]

        progress(0.0, f"Running whisper.cpp ({kind})")
        tail: list[str] = []
        proc = subprocess.Popen(cmd, cwd=cli.parent, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        assert proc.stdout is not None
        for line in proc.stdout:
            tail = (tail + [line.rstrip()])[-30:]
            if m := _PROGRESS_RE.search(line):
                progress(int(m.group(1)) / 100, f"whisper.cpp {m.group(1)}%")
        if proc.wait() != 0:
            raise RuntimeError("whisper.cpp failed:\n" + "\n".join(tail))

        data = json.loads(out_base.with_suffix(".json").read_text(encoding="utf-8", errors="replace"))

    out = []
    for item in data.get("transcription", []):
        text = item.get("text", "").strip()
        if text:
            off = item["offsets"]
            out.append(Segment(off["from"] / 1000, off["to"] / 1000, text))
    return out


def _hms(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"
