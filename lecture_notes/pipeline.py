"""Input file -> transcript -> analysis -> report files."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .analyze import analyze
from .asr import AsrOptions, Segment, transcribe
from .audio import extract_wav
from .report import to_markdown
from .transcript import TRANSCRIPT_SUFFIXES, load_transcript, to_json, to_srt, to_txt

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "outputs"

ProgressFn = Callable[[float, str], None]


@dataclass
class RunOptions:
    asr: AsrOptions = field(default_factory=AsrOptions)
    analyze: bool = True
    claude_model: str = "claude-opus-5-5"
    effort: str = "high"
    context: str = ""
    api_key: str | None = None
    out_dir: Path = DEFAULT_OUT
    reuse_transcript: bool = True


@dataclass
class RunResult:
    out_dir: Path
    files: dict[str, Path]
    markdown: str
    segments: list[Segment]


def _cache_key(src: Path, asr: AsrOptions) -> dict:
    st = src.stat()
    keep = ("model", "language", "prompt", "beam_size", "chunk_length", "vad")
    return {"source": src.name, "size": st.st_size, "mtime": int(st.st_mtime),
            **{k: v for k, v in asdict(asr).items() if k in keep}}


def run(src: str | Path, opts: RunOptions, progress: ProgressFn = lambda f, m: None) -> RunResult:
    src = Path(src)
    out = Path(opts.out_dir) / src.stem
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}

    # 1. Transcript: given directly, cached from an earlier run, or transcribed now.
    tjson = out / "transcript.json"
    if src.suffix.lower() in TRANSCRIPT_SUFFIXES:
        segments = load_transcript(src)
        asr_info = f"ไฟล์ทรานสคริปต์ที่อัปโหลด ({src.suffix})"
        if src.resolve() != tjson.resolve():
            tjson.write_text(to_json(segments, {"source": src.name, "backend": "uploaded transcript"}),
                             encoding="utf-8")
        progress(0.6, f"Loaded {len(segments)} transcript lines from {src.name}")
    else:
        key = _cache_key(src, opts.asr)
        cached = json.loads(tjson.read_text(encoding="utf-8")) if tjson.exists() else None
        if opts.reuse_transcript and cached and {k: cached["meta"].get(k) for k in key} == key:
            segments = load_transcript(tjson)
            backend = cached["meta"].get("backend", "?")
            progress(0.6, f"Reusing transcript from an earlier run ({backend})")
        else:
            with tempfile.TemporaryDirectory(prefix="lecture-notes-") as td:
                wav = Path(td) / "audio.wav"
                progress(0.0, "Extracting audio")
                extract_wav(src, wav, lambda f: progress(0.05 * f, f"Extracting audio {f:.0%}"))
                segments, backend = transcribe(
                    wav, opts.asr, lambda f, m: progress(0.05 + 0.55 * f, m))
            tjson.write_text(to_json(segments, {**key, "backend": backend}), encoding="utf-8")
        asr_info = f"{backend} / {opts.asr.model}"
    files["transcript.json"] = tjson
    (out / "transcript.srt").write_text(to_srt(segments), encoding="utf-8")
    (out / "transcript.txt").write_text(to_txt(segments), encoding="utf-8")
    files["transcript.srt"] = out / "transcript.srt"
    files["transcript.txt"] = out / "transcript.txt"

    # 2. Topics + issues via Claude.
    markdown = ""
    if opts.analyze:
        progress(0.62, f"Sending transcript to {opts.claude_model}")
        result = analyze(segments, source_name=src.name, context=opts.context,
                         model=opts.claude_model, effort=opts.effort, api_key=opts.api_key,
                         progress=lambda m: progress(0.8, m))
        (out / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        markdown = to_markdown(result, source_name=src.name, asr_info=asr_info)
        (out / "report.md").write_text(markdown, encoding="utf-8")
        files["analysis.json"] = out / "analysis.json"
        files["report.md"] = out / "report.md"
    progress(1.0, f"Done - files in {out}")
    return RunResult(out, files, markdown, segments)
