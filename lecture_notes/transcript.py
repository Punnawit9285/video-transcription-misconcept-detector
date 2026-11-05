"""Transcript formats: our JSON, SRT, and plain timestamped text."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .asr import Segment

TRANSCRIPT_SUFFIXES = {".srt", ".vtt", ".json"}


def hms(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def _srt_time(sec: float) -> str:
    ms = int(round(sec * 1000))
    return f"{ms // 3600000:02d}:{ms % 3600000 // 60000:02d}:{ms % 60000 // 1000:02d},{ms % 1000:03d}"


def to_srt(segments: list[Segment]) -> str:
    return "\n".join(
        f"{i}\n{_srt_time(s.start)} --> {_srt_time(s.end)}\n{s.text}\n"
        for i, s in enumerate(segments, 1)
    )


def to_txt(segments: list[Segment]) -> str:
    return "\n".join(f"[{hms(s.start)}] {s.text}" for s in segments) + "\n"


def to_json(segments: list[Segment], meta: dict) -> str:
    return json.dumps({"meta": meta, "segments": [s.__dict__ for s in segments]},
                      ensure_ascii=False, indent=1)


_TS = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{1,3})")


def _secs(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def load_transcript(path: str | Path) -> list[Segment]:
    """Read a transcript we wrote (.json) or a subtitle file (.srt / .vtt)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        items = data["segments"] if isinstance(data, dict) else data
        return [Segment(float(i["start"]), float(i["end"]), str(i["text"])) for i in items]

    segments: list[Segment] = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        lines = block.strip().splitlines()
        for idx, line in enumerate(lines):
            if m := _TS.search(line):
                body = " ".join(l.strip() for l in lines[idx + 1:] if l.strip())
                body = re.sub(r"<[^>]+>", "", body)
                if body:
                    segments.append(Segment(_secs(*m.groups()[:4]), _secs(*m.groups()[4:]), body))
                break
    if not segments:
        raise ValueError(f"No subtitle cues found in {path.name}")
    return segments
