"""Command line: python -m lecture_notes <video|audio|.srt|.json> [options]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import hardware
from .analyze import CLAUDE_MODELS, EFFORTS, AnalysisError
from .asr import BACKENDS, AsrOptions, resolve_backend, whisper_cpp_variants
from .models import DEFAULT_PROMPT, FASTER_WHISPER, WHISPER_CPP
from .pipeline import DEFAULT_OUT, RunOptions, run


def _doctor() -> None:
    print("GPUs:", ", ".join(hardware.gpu_names()) or "(none detected)")
    ok, why = hardware.cuda_status()
    print(f"faster-whisper CUDA: {'OK' if ok else 'not available - ' + why}")
    print("whisper.cpp builds installed:", ", ".join(whisper_cpp_variants()) or "(none)")
    print("auto backend ->", resolve_backend("auto"))


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(prog="lecture-notes", description=__doc__)
    p.add_argument("input", nargs="?", help="video/audio file, or an existing .srt/.vtt/.json transcript")
    p.add_argument("--doctor", action="store_true", help="show detected GPUs and usable backends, then exit")
    p.add_argument("--backend", default="auto", choices=list(BACKENDS))
    p.add_argument("--model", default="large-v3",
                   help=f"faster-whisper: {', '.join(FASTER_WHISPER)} (or HF id/path); "
                        f"whisper.cpp: {', '.join(WHISPER_CPP)} (or .bin path)")
    p.add_argument("--language", default="th", help="Whisper language code, or 'auto'")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="hint text: subject and key terms")
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8, help="faster-whisper CUDA batch size (lower if out of VRAM)")
    p.add_argument("--chunk-length", type=int, default=15, help="faster-whisper batched chunk seconds")
    p.add_argument("--compute-type", default="", help="faster-whisper: int8_float16, float16, int8, ...")
    p.add_argument("--no-vad", action="store_true", help="disable voice-activity detection")
    p.add_argument("--no-analysis", action="store_true", help="transcribe only, skip Claude")
    p.add_argument("--claude-model", default="claude-opus-5-5", choices=list(CLAUDE_MODELS))
    p.add_argument("--effort", default="high", choices=EFFORTS)
    p.add_argument("--context", default="", help="e.g. 'Physiology lecture for 2nd-year med students'")
    p.add_argument("--retranscribe", action="store_true", help="ignore a cached transcript")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = p.parse_args(argv)

    if a.doctor:
        _doctor()
        return 0
    if not a.input:
        p.error("input file is required (or use --doctor)")

    opts = RunOptions(
        asr=AsrOptions(backend=a.backend, model=a.model, language=a.language, prompt=a.prompt,
                       beam_size=a.beam_size, batch_size=a.batch_size, chunk_length=a.chunk_length,
                       compute_type=a.compute_type, vad=not a.no_vad),
        analyze=not a.no_analysis, claude_model=a.claude_model, effort=a.effort,
        context=a.context, out_dir=a.out, reuse_transcript=not a.retranscribe,
    )
    last = [-1.0]

    def progress(frac: float, msg: str) -> None:
        if frac - last[0] >= 0.02 or frac >= 1.0 or "Claude" in msg:
            last[0] = frac
            print(f"[{frac:6.1%}] {msg}", flush=True)

    try:
        result = run(a.input, opts, progress)
    except (AnalysisError, RuntimeError, ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    for name, path in result.files.items():
        print(f"  {name:16} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
