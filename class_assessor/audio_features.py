from pathlib import Path
from typing import List

import librosa
import numpy as np

from class_assessor.config import (
    HIGH_SILENCE_RATIO_THRESHOLD,
    LOW_ENERGY_RMS_THRESHOLD,
)
from class_assessor.models import AudioSummary


def summarize_audio(wav_path: Path) -> AudioSummary:
    y, sr = librosa.load(str(wav_path), sr=16000, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))
    if duration <= 0:
        return AudioSummary(
            duration_sec=0.0,
            mean_rms=0.0,
            silence_ratio=1.0,
            notes=["No audio track detected."],
        )

    rms = librosa.feature.rms(y=y)[0]
    mean_rms = float(np.mean(rms))
    silence = float(np.mean(rms < 0.01))

    notes: List[str] = []
    if mean_rms < LOW_ENERGY_RMS_THRESHOLD:
        notes.append("Low overall vocal energy — delivery may sound flat.")
    if silence > HIGH_SILENCE_RATIO_THRESHOLD:
        notes.append("High proportion of silence — pacing may be slow or gaps long.")

    if not notes:
        notes.append("Audio energy and pacing look typical for a live class.")

    return AudioSummary(
        duration_sec=duration,
        mean_rms=mean_rms,
        silence_ratio=silence,
        notes=notes,
    )
