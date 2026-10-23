from pathlib import Path

# Media processing
FRAME_INTERVAL_SEC = 45
MAX_FRAMES = 24
WHISPER_MODEL = "tiny"  # tiny fastest on CPU; use base/small if accuracy matters
WHISPER_BEAM_SIZE = 1  # 1 = much faster; 5 = slower, slightly better
WHISPER_CPU_THREADS = 0  # 0 = let faster-whisper decide

# MP4-only stratified clips (intro + mid + end) — faster than full class, better than ends-only
TRANSCRIBE_OPENING_SEC = 12 * 60
TRANSCRIBE_MID_DURATION_SEC = 8 * 60
TRANSCRIBE_MID_OFFSET_SEC = 4 * 60  # mid clip centered ~50% of class
TRANSCRIBE_CLOSING_SEC = 10 * 60
TRANSCRIBE_FULL_IF_SHORTER_SEC = 28 * 60  # classes shorter than this → full transcribe

OPENING_GREETING_SEC = 180  # greeting after this = "late" (B482-style)

MAX_VIDEO_BYTES = 800 * 1024 * 1024  # 800 MB

# Audio heuristics
LOW_ENERGY_RMS_THRESHOLD = 0.02
HIGH_SILENCE_RATIO_THRESHOLD = 0.35

# Face / camera
DARK_FRAME_MEAN_THRESHOLD = 25
MIN_FACE_VISIBLE_RATIO = 0.25

PROJECT_ROOT = Path(__file__).resolve().parent.parent
