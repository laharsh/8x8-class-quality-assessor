"""Load organizer-provided transcript files (dev/calibration only)."""

import json
import re
from pathlib import Path
from typing import List

from class_assessor.models import TranscriptSegment

SRT_TS = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)


def _ts(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _strip_webvtt_header(text: str) -> str:
    lines = text.splitlines()
    if not lines or not lines[0].strip().upper().startswith("WEBVTT"):
        return text
    body: List[str] = []
    past_header = False
    for line in lines[1:]:
        if not past_header:
            if line.strip() == "":
                past_header = True
            continue
        body.append(line)
    return "\n".join(body) if body else text


def _parse_srt_vtt(text: str) -> List[TranscriptSegment]:
    text = _strip_webvtt_header(text)
    segments: List[TranscriptSegment] = []
    blocks = re.split(r"\n\s*\n", text.strip())
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        if lines[0].isdigit():
            lines = lines[1:]
        m = SRT_TS.match(lines[0])
        if not m:
            continue
        start = _ts(*m.groups()[:4])
        end = _ts(*m.groups()[4:])
        body = " ".join(lines[1:])
        # Drop "Speaker: " prefix; keep spoken text for rubric keywords
        body = re.sub(r"^[\w\s.'’-]+:\s*", "", body)
        if body:
            segments.append(TranscriptSegment(start=start, end=end, text=body))
    return segments


def _parse_json(data: object) -> List[TranscriptSegment]:
    segments: List[TranscriptSegment] = []
    if isinstance(data, dict) and "segments" in data:
        items = data["segments"]
    elif isinstance(data, list):
        items = data
    else:
        return segments
    for item in items:
        if not isinstance(item, dict):
            continue
        text = (item.get("text") or item.get("transcript") or "").strip()
        if not text:
            continue
        start = float(item.get("start", item.get("start_time", 0)))
        end = float(item.get("end", item.get("end_time", start + 1)))
        segments.append(TranscriptSegment(start=start, end=end, text=text))
    return segments


def load_transcript_file(path: Path) -> List[TranscriptSegment]:
    text = path.read_text(encoding="utf-8", errors="replace")
    suffix = path.suffix.lower()

    if suffix == ".json":
        return _parse_json(json.loads(text))
    if suffix in (".srt", ".vtt"):
        return _parse_srt_vtt(text)

    # Plain text: one segment (no timestamps) — still usable for keyword rubric
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    if SRT_TS.search(text):
        return _parse_srt_vtt(text)
    joined = " ".join(lines)
    return [TranscriptSegment(start=0.0, end=0.0, text=joined)]
