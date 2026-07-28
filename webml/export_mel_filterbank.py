"""One-off dev script: export librosa's mel filterbank matrix as a flat
binary Float32Array so the browser can compute log-mel spectrograms without
reimplementing librosa's filter design algorithm - just a linear STFT
(straightforward to hand-roll: power-of-two FFT, Hann window) followed by
a matrix multiply against this precomputed, fixed filterbank.

Not part of the served app - run manually:
    python webml/export_mel_filterbank.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa
import numpy as np

from backend.config import FMAX, FMIN, N_FFT, N_MELS, SAMPLE_RATE

OUTPUT_PATH = Path(__file__).resolve().parent / "mel_filterbank.bin"


def main() -> None:
    filterbank = librosa.filters.mel(
        sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX
    ).astype(np.float32)

    print(f"filterbank shape: {filterbank.shape} (n_mels x n_fft_bins)")
    assert filterbank.shape == (N_MELS, N_FFT // 2 + 1)

    filterbank.tofile(OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size} bytes)")
    print("Row-major float32, shape [229, 1025] - reshape accordingly in JS.")


if __name__ == "__main__":
    main()
