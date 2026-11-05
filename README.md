# Lecture Notes

Upload a lecture video and get:

1. **Transcript** with timestamps (`.txt`, `.srt`, `.json`): Thai speech with English terms
2. **Topic map**: which part of the video covers which topic (start/end + subtopics)
3. **Review notes**: things said wrong, misspoken, unclear or oversimplified, each with a timestamp, a quote and a suggested correction

The notes are written in Thai. Transcription runs locally on your GPU (or CPU). The topic map and review notes come from the Claude API.

## Hardware: which backend to use

| Machine | Backend | Install | Notes |
|---|---|---|---|
| **NVIDIA** (e.g. RTX 3070 Ti 8 GB) | `faster-whisper-cuda` | `install.bat nvidia` | Batched, `int8_float16`. Measured: 98-min lecture in ~7 min with `large-v3`. |
| **AMD** (e.g. RX 9070 XT 16 GB) | `whisper.cpp-vulkan` | `install.bat amd` | Vulkan build, ~20 MB; works with the standard Adrenalin driver. |
| **AMD**, ROCm alternative | `whisper.cpp-rocm` | `install.bat amd-rocm` | ROCm build for RDNA4 (gfx120X), ~500 MB. Try it if Vulkan is slow. |
| No GPU | `faster-whisper-cpu` | `install.bat cpu` | Works, but slow: 2–4× the video length for `large-v3`. |

`auto` (the default) picks NVIDIA CUDA if it works, else an installed whisper.cpp GPU build, else CPU.
Run `python -m lecture_notes --doctor` to see what was detected.

> The AMD builds come from AMD's [Lemonade whisper.cpp builds](https://github.com/lemonade-sdk/whisper.cpp-rocm). They are not tested on an RX 9070 XT yet. If whisper.cpp fails, run `--doctor` and try the other AMD build.

## Install (Windows)

Requires Python 3.10+ and Git.

```bat
git clone <this repo> lecture-notes
cd lecture-notes
install.bat nvidia        :: or: amd / amd-rocm / cpu
setx ANTHROPIC_API_KEY "sk-ant-..."
```

Open a new terminal after `setx`, then double-click **`start.bat`**. A browser tab opens with the upload page.

## Use

**Web UI** (`start.bat`): upload a video, audio file or existing transcript (`.srt` / `.vtt` / `.json`), or paste a local path (faster for multi-GB videos), then press **เริ่ม**.

**Command line**:

```bat
.venv\Scripts\activate
python -m lecture_notes "C:\Users\me\Desktop\Intro physio.mp4" --context "ติวสรีรวิทยา CVS"
python -m lecture_notes lecture.mp4 --backend whisper.cpp-vulkan --model large-v3-q5_0
python -m lecture_notes lecture.mp4 --no-analysis            :: transcript only, no API cost
python -m lecture_notes outputs\lecture\transcript.json      :: re-run analysis on a saved transcript
```

Results go to `outputs/<video name>/`: `transcript.txt`, `transcript.srt`, `transcript.json`, `analysis.json`, `report.md`.
Re-running on the same file reuses the saved transcript (use `--retranscribe` to redo it).

## Options

**Whisper models** (`--model`):

| faster-whisper (NVIDIA / CPU) | whisper.cpp (AMD) | |
|---|---|---|
| `large-v3` | `large-v3` | Best for Thai + English terms (default) |
| `large-v3-turbo` | `large-v3-turbo`, `large-v3-turbo-q8_0` | 3–5× faster, slightly weaker Thai |
| | `large-v3-q5_0` | Quantized, 1.1 GB, near-identical accuracy, faster |
| `thonburian-th-large-v3` | | Thai fine-tune ([community CT2 conversion](https://huggingface.co/Avocaduu14/whisper-th-large-v3-ct2)): better pure Thai, may garble English terms |
| `medium`, `small` | `medium` | Faster, less accurate |

You can also pass any Hugging Face CTranslate2 repo id or local folder (faster-whisper), or a local ggml `.bin` (whisper.cpp).

**Speed and quality settings**:
- `--prompt`: subject plus key terms. Helps Whisper spell English terms correctly. Without it, the first sentences come out noticeably worse.
- `--batch-size` (NVIDIA): 8 fits in 8 GB VRAM with `large-v3`. It halves automatically on out-of-memory.
- `--chunk-length` (NVIDIA): 15 s by default. 30 s chunks lose sentences in dense Thai speech, because each chunk overflows Whisper's token window.
- `--beam-size`: 5 by default. Use 1–2 for speed.

**Claude** (`--claude-model`, `--effort`): `claude-opus-5-5` (default, best at catching errors), `claude-sonnet-5-5` (about half the cost), `claude-haiku-5-5` (cheapest). Effort `high` is the default. A 1.5-hour lecture is roughly 60–100k input tokens.

## How it works

```
video ──PyAV──► 16 kHz WAV ──► faster-whisper (CUDA/CPU) ─┐
                           └──► whisper.cpp (Vulkan/ROCm) ─┴─► segments ──► Claude (structured JSON) ──► report.md
```

- `lecture_notes/audio.py`: streams the audio track out of any video (no ffmpeg install needed)
- `lecture_notes/asr.py`: the backends; `models.py`: model presets.
  After batched decoding, a repair pass re-decodes any segment that is a Whisper repetition loop (e.g. "แคลเซียมแคลเซียมแคลเซียม…") or an echo of the hint prompt over silence. The thresholds are tuned for Thai, which compresses better than English.
- `lecture_notes/analyze.py`: prompt and JSON schema for topics and issues; server-side refusal fallback is enabled
- `lecture_notes/report.py`: Thai Markdown report
- `lecture_notes/app.py`: Gradio UI; `cli.py`: command line

## วิธีใช้แบบย่อ (ภาษาไทย)

1. เครื่อง NVIDIA: รัน `install.bat nvidia` / เครื่อง AMD: รัน `install.bat amd`
2. ตั้ง API key: `setx ANTHROPIC_API_KEY "sk-ant-..."` แล้วเปิด terminal ใหม่
3. ดับเบิลคลิก `start.bat` แล้วอัปโหลดคลิป (หรือวาง path ไฟล์) กด **เริ่ม**
4. ได้ทรานสคริปต์พร้อมเวลา หัวข้อแต่ละช่วง และจุดที่พูดผิดหรือไม่ชัด ในโฟลเดอร์ `outputs/`
