"""One-off dev script: export a real 16kHz-mono waveform plus the exact
log-mel spectrogram backend/audio.py computes for it, so the JS
implementation (mel-spectrogram.js) can be checked against ground truth in
the browser instead of just "looking right".

Not part of the served app - run manually:
    python webml/export_test_vectors.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from backend.audio import load_audio_mono_16k, waveform_to_log_mel

OUTPUT_DIR = Path(__file__).resolve().parent

# Reuse a real cached MAESTRO recording if available; otherwise fall back
# to a short synthetic chirp so this still runs standalone.
CANDIDATE_AUDIO = Path(
    "training/maestro_data/maestro-v3.0.0/2015/"
    "MIDI-Unprocessed_R1_D2-21-22_mid--AUDIO-from_mp3_22_R1_2015_wav--3.wav"
)


def get_waveform() -> np.ndarray:
    project_root = Path(__file__).resolve().parent.parent
    audio_path = project_root / CANDIDATE_AUDIO
    if audio_path.exists():
        print(f"Using real recording: {audio_path}")
        waveform = load_audio_mono_16k(audio_path)
        return waveform[: 16000 * 8]  # first 8 seconds is plenty to validate

    print("Real recording not found, using a synthetic chirp instead.")
    sr = 16000
    t = np.linspace(0, 4.0, sr * 4, endpoint=False)
    freq = 220 + 400 * (t / t[-1])
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def main() -> None:
    waveform = get_waveform().astype(np.float32)
    mel = waveform_to_log_mel(waveform)

    waveform.tofile(OUTPUT_DIR / "test_waveform.bin")
    mel.astype(np.float32).tofile(OUTPUT_DIR / "test_waveform_mel_reference.bin")

    print(f"waveform: {waveform.shape} samples")
    print(f"reference mel: {mel.shape} (frames x mels)")
    print("Wrote test_waveform.bin and test_waveform_mel_reference.bin")


if __name__ == "__main__":
    main()
