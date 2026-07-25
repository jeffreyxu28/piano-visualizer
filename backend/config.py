"""Shared constants for the piano transcription backend.

These values must match the settings the model was trained with
(see the training notebook) or the checkpoint's weights will not
line up with the spectrogram shapes the network expects.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "best_piano_model.pt"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
STATIC_DIR = PROJECT_ROOT / "static"

# ---------------------------------------------------------------------------
# Audio / spectrogram settings (must match training)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16_000
N_FFT = 2_048
HOP_LENGTH = 512
N_MELS = 229
FMIN = 20.0
FMAX = 8_000.0

# ---------------------------------------------------------------------------
# Piano key range
# ---------------------------------------------------------------------------
MIN_MIDI = 21  # A0
MAX_MIDI = 108  # C8
N_KEYS = 88

# ---------------------------------------------------------------------------
# Chunking for inference on long files
# ---------------------------------------------------------------------------
CHUNK_SECONDS = 10.0
WINDOW_OVERLAP_SECONDS = 2.0

# ---------------------------------------------------------------------------
# Note decoding thresholds (must match training/eval choices)
# ---------------------------------------------------------------------------
ONSET_THRESHOLD = 0.50
FRAME_THRESHOLD = 0.50
MIN_START_FRAME_PROB = 0.30
FRAME_ONLY_START_THRESHOLD = 0.70
OFF_FRAMES_REQUIRED = 2

# ---------------------------------------------------------------------------
# Upload / processing limits
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm"}

# Overridable via environment variables so a resource-constrained deployment
# (e.g. a free hosting tier with limited RAM) can set tighter caps without
# touching code - local/default runs are unaffected.
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 100 * 1024 * 1024))  # 100 MB
MAX_AUDIO_SECONDS = int(os.environ.get("MAX_AUDIO_SECONDS", 20 * 60))  # 20 minutes
MIN_AUDIO_SECONDS = 0.25
