import shutil
from pathlib import Path
from typing import BinaryIO, Callable, List, Optional, Union

from class_assessor.assess import assess
from class_assessor.audio_features import summarize_audio
from class_assessor.features import extract_transcript_features
from class_assessor.ingest import materialize_video
from class_assessor.media import extract_audio_wav, probe_duration_sec, sample_frames
from class_assessor.models import AssessmentResult, TranscriptSegment
from class_assessor.transcribe import transcribe_from_video
from class_assessor.vision import analyze_frames

ProgressFn = Callable[[str, float], None]


def _noop_progress(msg: str, frac: float) -> None:
    pass


def _audio_for_analysis(video_path: Path, work_dir: Path, prog: ProgressFn) -> object:
    prog("Extracting audio snippet for energy analysis (ffmpeg)…", 0.45)
    wav_snip = work_dir / "audio_snip.wav"
    extract_audio_wav(video_path, wav_snip, duration_sec=600)
    return summarize_audio(wav_snip)


def run_pipeline(
    source: Union[str, BinaryIO, bytes],
    *,
    suffix: str = ".mp4",
    use_llm: bool = True,
    progress: Optional[ProgressFn] = None,
    transcript_segments: Optional[List[TranscriptSegment]] = None,
    whisper_model: Optional[str] = None,
    transcribe_mode: str = "stratified",
) -> tuple[AssessmentResult, Path]:
    prog = progress or _noop_progress
    is_url = isinstance(source, str) and source.strip().startswith("http")

    if is_url:
        prog("Step 1/6 — Downloading video from URL…", 0.02)
    else:
        prog("Step 1/6 — Saving uploaded video…", 0.02)

    video_path = materialize_video(source, suffix=suffix, progress=prog)
    work_dir = video_path.parent
    size_mb = video_path.stat().st_size / (1024 * 1024)
    duration = probe_duration_sec(video_path)
    prog(
        f"Step 2/6 — Video ready ({size_mb:.1f} MB, ~{int(duration // 60)}:{int(duration % 60):02d}).",
        0.08,
    )

    tx_note = ""
    try:
        frames_dir = work_dir / "frames"
        prog("Step 3/6 — Sampling video frames (ffmpeg)…", 0.12)
        frames = sample_frames(video_path, frames_dir)

        if transcript_segments:
            prog("Step 4/6 — Using your transcript (Whisper skipped)…", 0.35)
            segments = transcript_segments
            tx_note = "Transcript supplied by user (not generated)."
            prog("Step 5/6 — Audio energy from first ~10 min of video…", 0.5)
            audio = _audio_for_analysis(video_path, work_dir, prog)
        else:
            prog(
                f"Step 4/6 — Speech-to-text ({transcribe_mode}, Whisper {whisper_model or 'tiny'})…",
                0.18,
            )
            segments, tx_note = transcribe_from_video(
                video_path,
                work_dir,
                model_name=whisper_model,
                mode=transcribe_mode,
                progress=prog,
            )
            prog("Step 5/6 — Audio energy…", 0.72)
            snip = None
            for candidate in (
                work_dir / "audio_opening.wav",
                work_dir / "audio_full.wav",
                work_dir / "audio_open.wav",
            ):
                if candidate.exists():
                    snip = candidate
                    break
            if snip:
                audio = summarize_audio(snip)
            else:
                audio = _audio_for_analysis(video_path, work_dir, prog)

        prog("Step 5/6 — Face / camera heuristics on frames…", 0.78)
        tfeat = extract_transcript_features(segments)
        vision = analyze_frames(frames)

        prog("Step 6/6 — Scoring class quality…", 0.88)
        result = assess(segments, tfeat, vision, audio, use_llm=use_llm)
        if tx_note:
            result.limitations.insert(0, tx_note)

        prog("Complete", 1.0)
        return result, work_dir
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
