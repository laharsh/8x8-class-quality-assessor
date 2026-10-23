import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from class_assessor.config import FRAME_INTERVAL_SEC, MAX_FRAMES


def _run(cmd: List[str]) -> None:
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n{proc.stderr or proc.stdout}"
        )


def probe_duration_sec(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return 0.0
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0


def extract_audio_wav(
    video_path: Path,
    wav_path: Path,
    *,
    start_sec: float = 0.0,
    duration_sec: Optional[float] = None,
) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y"]
    if start_sec > 0:
        cmd.extend(["-ss", str(start_sec)])
    cmd.extend(
        [
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
        ]
    )
    if duration_sec is not None and duration_sec > 0:
        cmd.extend(["-t", str(duration_sec)])
    cmd.append(str(wav_path))
    _run(cmd)


def sample_frames(video_path: Path, out_dir: Path) -> List[Tuple[float, Path]]:
    """Sample frames every FRAME_INTERVAL_SEC, capped at MAX_FRAMES."""
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = probe_duration_sec(video_path)
    if duration <= 0:
        duration = 3600.0

    interval = FRAME_INTERVAL_SEC
    n = int(duration // interval) + 1
    if n > MAX_FRAMES:
        interval = max(interval, duration / MAX_FRAMES)

    pattern = str(out_dir / "frame_%04d.jpg")
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps=1/{interval}",
            "-frames:v",
            str(MAX_FRAMES),
            pattern,
        ]
    )

    frames: List[Tuple[float, Path]] = []
    for i, path in enumerate(sorted(out_dir.glob("frame_*.jpg"))):
        ts = i * interval
        frames.append((ts, path))
    return frames
