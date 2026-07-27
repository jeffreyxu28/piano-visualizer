"""Audio loading and log-mel spectrogram generation.

Pipeline: uploaded file -> mono float32 waveform @ 16 kHz -> log-mel
spectrogram. This mirrors the preprocessing used during training exactly,
since the trained network only ever saw spectrograms produced this way.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from backend.config import (
    FMAX,
    FMIN,
    HOP_LENGTH,
    MAX_AUDIO_SECONDS,
    MIN_AUDIO_SECONDS,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
)


class AudioError(Exception):
    """Raised for any problem loading or validating uploaded audio."""


def probe_duration_seconds(path: Path) -> float | None:
    """Best-effort duration check without decoding the whole file.

    Returns None if the container format can't be probed this way (e.g. some
    MP3s) - full decoding will still validate duration afterwards.
    """
    try:
        with sf.SoundFile(str(path)) as f:
            if f.samplerate <= 0:
                return None
            return float(len(f)) / float(f.samplerate)
    except Exception:
        return None


def load_audio_mono_16k(path: Path) -> np.ndarray:
    """Load an audio file of any supported format as mono float32 @ 16kHz.

    Raises AudioError with a user-facing message on any failure: missing
    file, corrupted/unreadable audio, empty audio, or audio outside the
    allowed duration range.
    """
    if not path.exists() or path.stat().st_size == 0:
        raise AudioError("The uploaded file is empty or could not be saved.")

    probed = probe_duration_seconds(path)
    if probed is not None and probed > MAX_AUDIO_SECONDS:
        minutes = MAX_AUDIO_SECONDS / 60
        raise AudioError(
            f"This file is too long ({probed / 60:.1f} min). "
            f"The maximum supported length is {minutes:.1f} minutes."
        )

    try:
        audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    except Exception as exc:
        raise AudioError(
            "Could not decode this audio file. It may be corrupted or in an "
            "unsupported format. Supported formats: WAV, MP3, FLAC, M4A, OGG, WEBM."
        ) from exc

    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0:
        raise AudioError("The audio file contains no audio data.")

    duration = len(audio) / SAMPLE_RATE
    if duration < MIN_AUDIO_SECONDS:
        raise AudioError("The audio file is too short to transcribe.")
    if duration > MAX_AUDIO_SECONDS:
        minutes = MAX_AUDIO_SECONDS / 60
        raise AudioError(
            f"This file is too long ({duration / 60:.1f} min). "
            f"The maximum supported length is {minutes:.1f} minutes."
        )

    if not np.isfinite(audio).all():
        raise AudioError("The audio file contains invalid (non-finite) samples.")

    peak = float(np.abs(audio).max())
    if peak < 1e-6:
        raise AudioError("The audio file appears to be silent.")

    return audio


def waveform_to_log_mel(audio: np.ndarray) -> np.ndarray:
    """Convert a mono float32 waveform @ 16kHz into a normalized log-mel
    spectrogram shaped [time, mel_bins], matching the training pipeline.
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
        center=True,
    )

    db = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    normalized = (db + 80.0) / 80.0
    return normalized.T.astype(np.float32)  # [time, mel]
