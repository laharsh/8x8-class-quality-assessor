from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from class_assessor.config import (
    DARK_FRAME_MEAN_THRESHOLD,
    MIN_FACE_VISIBLE_RATIO,
)
from class_assessor.models import VisionSummary

_FACE_CASCADE = None


def _get_face_cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        if not hasattr(cv2, "CascadeClassifier"):
            return None
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(path)
    return _FACE_CASCADE


def analyze_frames(frames: List[Tuple[float, Path]]) -> VisionSummary:
    if not frames:
        return VisionSummary(
            frames_sampled=0,
            dark_frame_ratio=1.0,
            face_visible_ratio=0.0,
            avg_brightness=0.0,
            notes=["No frames extracted — check ffmpeg."],
        )

    dark = 0
    face_hits = 0
    brightness_vals: List[float] = []
    notes: List[str] = []

    cascade = _get_face_cascade()
    if cascade is None:
        notes.append(
            "Face detection unavailable (OpenCV 5+); using brightness-only video heuristics."
        )

    for ts, path in frames:
        img = cv2.imread(str(path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_b = float(np.mean(gray))
        brightness_vals.append(mean_b)
        if mean_b < DARK_FRAME_MEAN_THRESHOLD:
            dark += 1

        if cascade is not None:
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
            )
            if len(faces) > 0:
                face_hits += 1
        elif mean_b >= DARK_FRAME_MEAN_THRESHOLD and mean_b > 40:
            # Bright non-dark frame as weak proxy when Haar cascades are missing
            face_hits += 1

    n = len(brightness_vals) or 1
    dark_ratio = dark / n
    face_ratio = face_hits / n
    avg_b = float(np.mean(brightness_vals)) if brightness_vals else 0.0

    if dark_ratio > 0.4:
        notes.append(
            f"Many dark or blank frames ({dark_ratio:.0%}) — possible camera-off periods."
        )
    if face_ratio < MIN_FACE_VISIBLE_RATIO:
        notes.append(
            f"Tutor face visible in only {face_ratio:.0%} of sampled frames."
        )
    else:
        notes.append(
            f"Tutor face detected in {face_ratio:.0%} of sampled frames."
        )

    return VisionSummary(
        frames_sampled=len(frames),
        dark_frame_ratio=dark_ratio,
        face_visible_ratio=face_ratio,
        avg_brightness=avg_b,
        notes=notes,
    )
