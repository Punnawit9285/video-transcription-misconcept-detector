"""Render the analysis as a Thai Markdown report."""

from __future__ import annotations

ISSUE_LABELS = {
    "factual_error": "เนื้อหาผิด",
    "misspoken": "พูดผิด/พูดสลับ",
    "unclear": "อธิบายไม่ชัด",
    "oversimplified": "ย่อจนคลาดเคลื่อน",
    "inconsistent": "ขัดกับที่พูดไว้",
    "possible_asr_error": "ฟังไม่ชัด/ควรฟังซ้ำ",
}
SEVERITY_LABELS = {"high": "🔴 สูง", "medium": "🟠 กลาง", "low": "🟡 ต่ำ"}


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def to_markdown(analysis: dict, *, source_name: str, asr_info: str) -> str:
    out = [f"# {analysis.get('title') or source_name}", ""]
    out += [f"> ไฟล์: `{source_name}` · ถอดเสียงด้วย {asr_info}"
            + (f" · วิเคราะห์ด้วย {analysis['_meta']['model']}" if "_meta" in analysis else ""), ""]
    if analysis.get("overview"):
        out += ["## ภาพรวม", "", analysis["overview"], ""]

    topics = analysis.get("topics", [])
    out += ["## หัวข้อตามช่วงเวลา", "", "| ช่วงเวลา | หัวข้อ | สรุป |", "|---|---|---|"]
    for t in topics:
        out.append(f"| {t['start']}–{t['end']} | **{_cell(t['title'])}** | {_cell(t['summary'])} |")
    out.append("")
    for i, t in enumerate(topics, 1):
        if t.get("subtopics"):
            out.append(f"**{i}. {t['title']}** ({t['start']}–{t['end']})")
            out += [f"- `{s['time']}` {s['title']}" for s in t["subtopics"]]
            out.append("")

    issues = analysis.get("issues", [])
    out += ["## จุดที่พูดผิดหรือไม่ชัดเจน", ""]
    if not issues:
        out += ["ไม่พบจุดที่ต้องแก้ไข", ""]
    else:
        counts = {}
        for it in issues:
            counts[it["type"]] = counts.get(it["type"], 0) + 1
        out += ["รวม " + ", ".join(f"{ISSUE_LABELS.get(k, k)} {v}" for k, v in counts.items()), ""]
        out += ["| เวลา | ประเภท | ระดับ | ที่พูดในคลิป | ปัญหา | ควรแก้เป็น |", "|---|---|---|---|---|---|"]
        for it in issues:
            out.append(
                f"| {it['time']} | {ISSUE_LABELS.get(it['type'], it['type'])} "
                f"| {SEVERITY_LABELS.get(it['severity'], it['severity'])} | “{_cell(it['quote'])}” "
                f"| {_cell(it['problem'])} | {_cell(it['correction'])} |"
            )
        out.append("")
    out += ["---", "_ทรานสคริปต์สร้างโดย AI อาจมีคำที่ถอดผิด โปรดฟังคลิปจริงยืนยันก่อนแก้ไข_", ""]
    return "\n".join(out)
