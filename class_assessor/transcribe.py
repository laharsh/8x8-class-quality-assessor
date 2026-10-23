import os
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import soundfile as sf

from class_assessor.config import (
    TRANSCRIBE_FULL_IF_SHORTER_SEC,
    TRANSCRIBE_MID_DURATION_SEC,
    TRANSCRIBE_MID_OFFSET_SEC,
    TRANSCRIBE_OPENING_SEC,
    TRANSCRIBE_CLOSING_SEC,
    WHISPER_BEAM_SIZE,
    WHISPER_CPU_THREADS,
    WHISPER_MODEL,
)
from class_assessor.media import extract_audio_wav, probe_duration_sec
from class_assessor.models import TranscriptSegment

ProgressFn = Callable[[str, float], None]


@lru_cache(maxsize=2)
def _whisper_model(model_name: str):
    from faster_whisper import WhisperModel

    threads = WHISPER_CPU_THREADS or max(1, (os.cpu_count() or 4) - 1)
    return WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        cpu_threads=threads,
    )


def _fmt_mmss(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def _load_wav_mono_16k(wav_path: Path) -> Tuple[np.ndarray, float]:
    """Load ffmpeg PCM wav — avoids PyAV in faster-whisper (PyAV 19 breaks on Streamlit)."""
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        import librosa

        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    duration = float(len(data)) / 16000.0
    return np.ascontiguousarray(data, dtype=np.float32), duration


def transcribe_audio(
    wav_path: Path,
    *,
    model_name: Optional[str] = None,
    time_offset_sec: float = 0.0,
    progress: Optional[ProgressFn] = None,
    phase_label: str = "",
    phase_index: int = 1,
    phase_total: int = 1,
) -> List[TranscriptSegment]:
    name = model_name or WHISPER_MODEL
    model = _whisper_model(name)
    audio, duration = _load_wav_mono_16k(wav_path)
    segments, info = model.transcribe(
        audio,
        beam_size=WHISPER_BEAM_SIZE,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    out: List[TranscriptSegment] = []
    if not duration:
        duration = getattr(info, "duration", None) or 0.0
    base_frac = 0.30 + 0.40 * ((phase_index - 1) / max(1, phase_total))
    span_frac = 0.40 / max(1, phase_total)

    for seg in segments:
        text = (seg.text or "").strip()
        if text:
            out.append(
                TranscriptSegment(
                    start=seg.start + time_offset_sec,
                    end=seg.end + time_offset_sec,
                    text=text,
                )
            )
        if progress and duration:
            inner = min(1.0, seg.end / duration)
            progress(
                f"{phase_label} — pass {phase_index}/{phase_total}, "
                f"{_fmt_mmss(seg.end)} of {_fmt_mmss(duration)} in this clip",
                base_frac + span_frac * inner,
            )
    return out


def _stratified_clips(duration: float) -> List[Tuple[str, float, float]]:
    """(name, start_sec, duration_sec) covering intro, mid-class, and closing."""
    opening = min(TRANSCRIBE_OPENING_SEC, duration)
    mid_len = min(TRANSCRIBE_MID_DURATION_SEC, max(0.0, duration - opening))
    mid_start = max(
        opening,
        min(duration - mid_len, duration * 0.5 - TRANSCRIBE_MID_OFFSET_SEC),
    )
    if mid_start + mid_len > duration:
        mid_start = max(0.0, duration - mid_len)

    close_len = min(TRANSCRIBE_CLOSING_SEC, duration)
    close_start = max(0.0, duration - close_len)

    clips: List[Tuple[str, float, float]] = [("opening", 0.0, opening)]
    if mid_len >= 60 and mid_start > opening + 30:
        clips.append(("mid-class", mid_start, mid_len))
    if close_len >= 60 and close_start > opening + 30:
        if not clips or close_start > clips[-1][1] + 60:
            clips.append(("closing", close_start, close_len))
    return clips


def transcribe_from_video(
    video_path: Path,
    work_dir: Path,
    *,
    model_name: Optional[str] = None,
    mode: str = "stratified",
    progress: Optional[ProgressFn] = None,
) -> Tuple[List[TranscriptSegment], str]:
    """
    mode: 'stratified' (3 clips: intro + middle + end) or 'full'.
    """
    duration = probe_duration_sec(video_path)
    if progress:
        progress("Loading Whisper model if needed (first run downloads weights)…", 0.2)
    _whisper_model(model_name or WHISPER_MODEL)

    if mode == "full" or duration <= TRANSCRIBE_FULL_IF_SHORTER_SEC:
        if progress:
            progress("Full transcription — extracting audio (ffmpeg)…", 0.22)
        wav_path = work_dir / "audio_full.wav"
        extract_audio_wav(video_path, wav_path)
        if progress:
            progress(
                f"Full transcription — 1/1 clip (entire ~{_fmt_mmss(duration)} class)",
                0.28,
            )
        segs = transcribe_audio(
            wav_path,
            model_name=model_name,
            progress=progress,
            phase_label="Full class",
            phase_index=1,
            phase_total=1,
        )
        return segs, "Transcript from full-class Whisper pass."

    clips = _stratified_clips(duration)
    total_phases = len(clips)
    merged: List[TranscriptSegment] = []
    parts: List[str] = []

    if progress:
        progress(
            f"Stratified mode: {total_phases} clips from ~{_fmt_mmss(duration)} class "
            f"(not the full recording — faster, covers intro/mid/end)",
            0.22,
        )

    for i, (name, start, clip_dur) in enumerate(clips, start=1):
        wav_path = work_dir / f"audio_{name}.wav"
        if progress:
            progress(
                f"Extracting clip {i}/{total_phases} ({name}: "
                f"{_fmt_mmss(start)}–{_fmt_mmss(start + clip_dur)})…",
                0.25 + 0.05 * i,
            )
        extract_audio_wav(
            video_path, wav_path, start_sec=start, duration_sec=clip_dur
        )
        label = f"Clip {i}/{total_phases} ({name})"
        merged.extend(
            transcribe_audio(
                wav_path,
                model_name=model_name,
                time_offset_sec=start,
                progress=progress,
                phase_label=label,
                phase_index=i,
                phase_total=total_phases,
            )
        )
        parts.append(
            f"{name} {_fmt_mmss(start)}–{_fmt_mmss(start + clip_dur)}"
        )

    note = (
        f"Stratified transcript ({total_phases} clips: {', '.join(parts)}; "
        f"class ~{_fmt_mmss(duration)}). Gaps between clips were not transcribed."
    )
    return merged, note


def segments_to_text(segments: List[TranscriptSegment]) -> str:
    return "\n".join(
        f"[{int(s.start // 60):02d}:{int(s.start % 60):02d}] {s.text}"
        for s in segments
    )
