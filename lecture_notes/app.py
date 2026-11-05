"""Local web UI: upload a lecture, get the transcript, topic map and review notes."""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from . import hardware
from .analyze import CLAUDE_MODELS, EFFORTS, AnalysisError
from .asr import BACKENDS, AsrOptions, resolve_backend, whisper_cpp_variants
from .models import DEFAULT_PROMPT, FASTER_WHISPER, WHISPER_CPP
from .pipeline import RunOptions, run


def _model_choices(backend: str) -> list[tuple[str, str]]:
    presets = WHISPER_CPP if resolve_backend(backend).startswith("whisper.cpp") else FASTER_WHISPER
    return [(f"{k} - {v.note}", k) for k, v in presets.items()]


def _hardware_summary() -> str:
    ok, why = hardware.cuda_status()
    gpus = ", ".join(hardware.gpu_names()) or "ไม่พบ GPU"
    wcpp = ", ".join(whisper_cpp_variants()) or "ยังไม่ได้ติดตั้ง"
    return (f"**GPU:** {gpus} · **CUDA:** {'พร้อมใช้' if ok else why} · "
            f"**whisper.cpp:** {wcpp} · **Auto จะใช้:** `{resolve_backend('auto')}`")


def _process(upload, local_path, context, backend, model, language, prompt, beam, batch,
             reuse, do_analyze, claude_model, effort, api_key, progress=gr.Progress()):
    src = (local_path or "").strip().strip('"') or upload
    if not src:
        raise gr.Error("อัปโหลดไฟล์ หรือวาง path ของไฟล์ก่อน")
    if not Path(src).is_file():
        raise gr.Error(f"ไม่พบไฟล์: {src}")
    if do_analyze and not (api_key or os.environ.get("ANTHROPIC_API_KEY")):
        raise gr.Error("ต้องใส่ Anthropic API key (หรือตั้ง ANTHROPIC_API_KEY) เพื่อสรุปหัวข้อ "
                       "- หรือเอาเครื่องหมายถูก 'วิเคราะห์ด้วย Claude' ออกเพื่อถอดเสียงอย่างเดียว")

    opts = RunOptions(
        asr=AsrOptions(backend=backend, model=model, language=language, prompt=prompt,
                       beam_size=int(beam), batch_size=int(batch)),
        analyze=do_analyze, claude_model=claude_model, effort=effort, context=context,
        api_key=api_key or None, reuse_transcript=reuse,
    )
    try:
        result = run(src, opts, lambda f, m: progress(f, desc=m))
    except AnalysisError as e:
        raise gr.Error(f"วิเคราะห์ไม่สำเร็จ: {e}") from e
    except (RuntimeError, ValueError) as e:
        raise gr.Error(str(e)) from e

    report = result.markdown or (
        "### ถอดเสียงเสร็จแล้ว\n\nไม่ได้วิเคราะห์ด้วย Claude - ดาวน์โหลดไฟล์ทรานสคริปต์ด้านล่าง\n\n```\n"
        + "\n".join(f"{s.start:8.1f}s  {s.text}" for s in result.segments[:40]) + "\n...\n```")
    return report, [str(p) for p in result.files.values()], f"บันทึกไฟล์ไว้ที่ `{result.out_dir}`"


def build() -> gr.Blocks:
    with gr.Blocks(title="Lecture Notes") as demo:
        gr.Markdown("# 🎓 Lecture Notes\nถอดเสียงคลิปบรรยาย → แบ่งหัวข้อตามเวลา → "
                    "หาจุดที่พูดผิดหรืออธิบายไม่ชัด (ภาษาไทย)")
        gr.Markdown(_hardware_summary())
        with gr.Row():
            with gr.Column(scale=1):
                upload = gr.File(label="อัปโหลดวิดีโอ/ไฟล์เสียง หรือทรานสคริปต์ (.srt / .vtt / .json)",
                                 file_types=["video", "audio", ".srt", ".vtt", ".json"], type="filepath")
                local_path = gr.Textbox(label="หรือวาง path ไฟล์ในเครื่อง (เร็วกว่าสำหรับไฟล์ใหญ่หลาย GB)",
                                        placeholder=r"C:\Users\me\Desktop\lecture.mp4")
                context = gr.Textbox(label="บริบท (ไม่บังคับ)",
                                     placeholder="เช่น ติวสรีรวิทยา CVS สำหรับนักศึกษาแพทย์ปี 2")
                with gr.Accordion("ถอดเสียง (Speech recognition)", open=False):
                    backend = gr.Dropdown([(v, k) for k, v in BACKENDS.items()], value="auto", label="Backend")
                    model = gr.Dropdown(_model_choices("auto"), value="large-v3", allow_custom_value=True,
                                        label="Whisper model")
                    language = gr.Dropdown(["th", "en", "auto"], value="th", allow_custom_value=True,
                                           label="ภาษาที่พูด")
                    prompt = gr.Textbox(DEFAULT_PROMPT, label="คำใบ้ (วิชา + ศัพท์เฉพาะ ช่วยให้สะกดถูก)", lines=2)
                    beam = gr.Slider(1, 10, value=5, step=1, label="Beam size (มาก = แม่นขึ้นแต่ช้าลง)")
                    batch = gr.Slider(1, 32, value=8, step=1,
                                      label="Batch size (NVIDIA เท่านั้น; ลดลงถ้า VRAM ไม่พอ)")
                    reuse = gr.Checkbox(True, label="ใช้ทรานสคริปต์เดิมถ้าเคยถอดไฟล์นี้แล้ว (model/ภาษาเดียวกัน)")
                with gr.Accordion("วิเคราะห์ด้วย Claude", open=False):
                    do_analyze = gr.Checkbox(True, label="วิเคราะห์ด้วย Claude (แบ่งหัวข้อ + หาจุดผิด)")
                    claude_model = gr.Dropdown([(v, k) for k, v in CLAUDE_MODELS.items()],
                                               value="claude-opus-5-5", label="Claude model")
                    effort = gr.Dropdown(EFFORTS, value="high", label="Effort (สูง = ละเอียดขึ้น แพงขึ้น)")
                    api_key = gr.Textbox(label="Anthropic API key (เว้นว่างถ้าตั้ง ANTHROPIC_API_KEY แล้ว)",
                                         type="password")
                go = gr.Button("เริ่ม", variant="primary")
            with gr.Column(scale=2):
                status = gr.Markdown()
                report = gr.Markdown(label="รายงาน")
                files = gr.File(label="ดาวน์โหลด", file_count="multiple")

        backend.change(lambda b: gr.Dropdown(choices=_model_choices(b), value="large-v3"), backend, model)
        go.click(_process,
                 [upload, local_path, context, backend, model, language, prompt, beam, batch,
                  reuse, do_analyze, claude_model, effort, api_key],
                 [report, files, status])
    return demo


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Lecture Notes web UI")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    a = p.parse_args()
    build().queue().launch(server_port=a.port, inbrowser=not a.no_browser)


if __name__ == "__main__":
    main()
