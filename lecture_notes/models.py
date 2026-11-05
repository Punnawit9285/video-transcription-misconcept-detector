"""Speech-recognition model presets per backend.

Any value not listed here is passed through: a Hugging Face repo id or a local
path for faster-whisper, a local ggml .bin path for whisper.cpp.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    source: str  # HF repo id (faster-whisper) or "repo::filename" (whisper.cpp)
    note: str


FASTER_WHISPER: dict[str, Preset] = {
    "large-v3": Preset("large-v3", "Best accuracy for Thai mixed with English terms (default)."),
    "large-v3-turbo": Preset("large-v3-turbo", "~3-5x faster; slightly weaker on Thai."),
    "thonburian-th-large-v3": Preset(
        "Avocaduu14/whisper-th-large-v3-ct2",
        "Community CTranslate2 conversion of biodatlab's Thai fine-tune; strong on pure Thai, "
        "may handle English terms less well.",
    ),
    "medium": Preset("medium", "Smaller/faster, noticeably less accurate on Thai."),
    "small": Preset("small", "Quick tests only."),
}

_GGML = "ggerganov/whisper.cpp"
WHISPER_CPP: dict[str, Preset] = {
    "large-v3": Preset(f"{_GGML}::ggml-large-v3.bin", "Full precision, 3.1 GB. Best accuracy (default)."),
    "large-v3-q5_0": Preset(f"{_GGML}::ggml-large-v3-q5_0.bin", "Quantized, 1.1 GB, faster, near-identical accuracy."),
    "large-v3-turbo": Preset(f"{_GGML}::ggml-large-v3-turbo.bin", "~3-5x faster; slightly weaker on Thai."),
    "large-v3-turbo-q8_0": Preset(f"{_GGML}::ggml-large-v3-turbo-q8_0.bin", "Turbo, quantized 8-bit."),
    "medium": Preset(f"{_GGML}::ggml-medium.bin", "Smaller/faster, less accurate on Thai."),
}

VAD_MODEL = "ggml-org/whisper-vad::ggml-silero-v5.1.2.bin"

DEFAULT_PROMPT = (
    "บรรยายภาษาไทยปนศัพท์ภาษาอังกฤษ เช่น physiology, homeostasis, negative feedback, "
    "cell membrane, action potential, receptor"
)
