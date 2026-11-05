"""Topic map + review notes from a transcript, via the Claude API (output in Thai)."""

from __future__ import annotations

import json
from typing import Callable

import anthropic

from .asr import Segment
from .transcript import hms

CLAUDE_MODELS = {
    "claude-opus-5-5": "Claude Opus 5.5 - best review quality (default)",
    "claude-sonnet-5-5": "Claude Sonnet 5.5 - about half the cost, still strong",
    "claude-haiku-5-5": "Claude Haiku 5.5 - cheapest, fine for topic maps, weaker at catching errors",
}
EFFORTS = ["low", "medium", "high", "xhigh", "max"]

ISSUE_TYPES = ["factual_error", "misspoken", "unclear", "oversimplified", "inconsistent", "possible_asr_error"]

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "overview", "topics", "issues"],
    "properties": {
        "title": {"type": "string", "description": "Lecture title in Thai"},
        "overview": {"type": "string", "description": "3-6 sentence Thai summary of the whole lecture"},
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["start", "end", "title", "summary", "subtopics"],
                "properties": {
                    "start": {"type": "string", "description": "HH:MM:SS"},
                    "end": {"type": "string", "description": "HH:MM:SS"},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "subtopics": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["time", "title"],
                            "properties": {
                                "time": {"type": "string", "description": "HH:MM:SS"},
                                "title": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["time", "type", "severity", "quote", "problem", "correction"],
                "properties": {
                    "time": {"type": "string", "description": "HH:MM:SS"},
                    "type": {"type": "string", "enum": ISSUE_TYPES},
                    "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                    "quote": {"type": "string", "description": "Short quote of what was said (from the transcript)"},
                    "problem": {"type": "string", "description": "What is wrong or unclear, in Thai"},
                    "correction": {"type": "string", "description": "Correct statement / clearer wording, in Thai"},
                },
            },
        },
    },
}

SYSTEM = """\
You review lecture recordings for the lecturer who recorded them. You get an automatic \
speech-recognition (ASR) transcript; each line starts with [HH:MM:SS]. Lectures are usually in \
Thai with English technical terms. Write everything you produce in Thai, keeping technical terms \
in English where a Thai student would expect them.

Produce:
1. A topic map: the lecture's main topics in order, each with start/end time, a short title, a \
1-3 sentence summary, and subtopics with the time each one starts. Topics should cover the whole \
lecture without gaps; an intro or off-topic segment is its own topic.
2. Issues: statements a careful subject-matter expert would flag -
   - factual_error: scientifically/medically wrong
   - misspoken: the speaker clearly said the wrong word (e.g. left for right, increase for decrease)
   - unclear: a student would likely misunderstand (undefined term, skipped step, ambiguous "this/that")
   - oversimplified: simplification that leads to a wrong mental model or wrong exam answer
   - inconsistent: contradicts something said elsewhere in the lecture (give both times)
   - possible_asr_error: the transcript is garbled or implausible at a point that matters, so the \
lecturer should re-listen; use this instead of blaming the speaker when the cause is likely ASR

Rules:
- The transcript is machine-generated. Misheard words, wrong homophones, and odd spellings of \
English terms are ASR noise, not the speaker's mistakes; read through them to what was most \
likely said. Never report pure transcription noise as factual_error or misspoken.
- Use only timestamps that appear in the transcript; pick the line where the point begins.
- Precision over volume: report real problems only, each once. An empty issue list is fine.
- In each issue, quote a few words as transcribed, explain the problem, and give the correct \
version a lecturer could say instead.
- Sort issues by time.
"""


class AnalysisError(RuntimeError):
    pass


def transcript_for_prompt(segments: list[Segment]) -> str:
    return "\n".join(f"[{hms(s.start)}] {s.text}" for s in segments)


def analyze(
    segments: list[Segment],
    *,
    source_name: str,
    context: str = "",
    model: str = "claude-opus-5-5",
    effort: str = "high",
    api_key: str | None = None,
    progress: Callable[[str], None] = lambda m: None,
) -> dict:
    """Return the analysis dict matching SCHEMA."""
    if not segments:
        raise AnalysisError("Transcript is empty - nothing to analyze.")
    client = anthropic.Anthropic(api_key=api_key or None, timeout=1800.0)
    duration = hms(segments[-1].end)
    user = (
        f"File: {source_name}\nDuration: {duration}\n"
        + (f"Context from the lecturer: {context}\n" if context.strip() else "")
        + f"\n<transcript>\n{transcript_for_prompt(segments)}\n</transcript>"
    )

    extra: dict = {}
    if not model.startswith("claude-haiku"):
        # Server-side fallback: if a safety classifier declines (e.g. a medical topic tripping the
        # bio classifier), the API re-runs the request on Anthropic's recommended fallback model.
        extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}

    chars = 0
    try:
        with client.beta.messages.stream(
            model=model,
            max_tokens=64000,
            system=SYSTEM,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
            **extra,
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta" and event.delta.type == "text_delta":
                    chars += len(event.delta.text)
                    if chars % 2000 < len(event.delta.text):
                        progress(f"Claude is writing the notes... ({chars:,} characters)")
            message = stream.get_final_message()
    except anthropic.AuthenticationError as e:
        raise AnalysisError("Invalid or missing Anthropic API key (set ANTHROPIC_API_KEY).") from e
    except anthropic.RateLimitError as e:
        raise AnalysisError("Rate limited by the Anthropic API - wait a minute and retry.") from e
    except anthropic.APIStatusError as e:
        raise AnalysisError(f"Anthropic API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise AnalysisError("Could not reach the Anthropic API - check your internet connection.") from e

    if message.stop_reason == "refusal":
        detail = getattr(message.stop_details, "explanation", "") if message.stop_details else ""
        raise AnalysisError(f"Claude declined this request. {detail}".strip())
    if message.stop_reason == "max_tokens":
        raise AnalysisError("The notes were cut off (output limit reached). Try a shorter recording.")

    # Keep only text written after the last model switch, if a fallback happened mid-stream.
    text = ""
    for block in message.content:
        if block.type == "fallback":
            text = ""
        elif block.type == "text":
            text += block.text
    try:
        result = json.loads(text)
    except json.JSONDecodeError as e:
        raise AnalysisError(f"Claude returned invalid JSON: {e}") from e
    result["_meta"] = {
        "model": message.model,
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
    }
    return result
