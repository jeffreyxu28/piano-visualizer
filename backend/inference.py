"""End-to-end orchestration: audio file -> note events + export artifacts.

The trained checkpoint is loaded once (lazily, on first request) and reused
for every subsequent transcription. This module never trains or fine-tunes
the model - it is inference only.
"""

from __future__ import annotations

import threading
from pathlib import Path

import torch

from backend.audio import AudioError, load_audio_mono_16k
from backend.config import CHECKPOINT_PATH, SAMPLE_RATE
from backend.model import PianoTranscriptionModel
from backend.note_decoder import decode_notes, predict_probabilities, save_csv, save_midi


class ModelNotAvailableError(Exception):
    """Raised when best_piano_model.pt is missing or fails to load."""


class TranscriptionError(Exception):
    """Raised when the transcription pipeline fails for a given file."""


_model_lock = threading.Lock()
_model: PianoTranscriptionModel | None = None
_device: torch.device | None = None


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model() -> tuple[PianoTranscriptionModel, torch.device]:
    """Lazily load and cache the trained model. Thread-safe."""
    global _model, _device

    if _model is not None and _device is not None:
        return _model, _device

    with _model_lock:
        if _model is not None and _device is not None:
            return _model, _device

        if not CHECKPOINT_PATH.exists():
            raise ModelNotAvailableError(
                f"Model checkpoint not found at {CHECKPOINT_PATH}. "
                "Place your trained 'best_piano_model.pt' file in the project root."
            )

        device = get_device()
        try:
            model = PianoTranscriptionModel().to(device)
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict)
            model.eval()
        except Exception as exc:
            raise ModelNotAvailableError(
                f"Failed to load the model checkpoint: {exc}"
            ) from exc

        _model = model
        _device = device
        return _model, _device


def transcribe_audio_file(
    audio_path: Path,
    csv_output_path: Path | None = None,
    midi_output_path: Path | None = None,
) -> tuple[list[dict], float]:
    """Run the full pipeline on a saved audio file and return
    (notes, duration_seconds). Optionally writes CSV/MIDI export files.
    """
    model, device = get_model()

    try:
        waveform = load_audio_mono_16k(audio_path)
    except AudioError:
        raise
    except Exception as exc:
        raise TranscriptionError(f"Failed to read audio: {exc}") from exc

    duration_seconds = len(waveform) / SAMPLE_RATE

    try:
        onset_probability, frame_probability = predict_probabilities(model, waveform, device)
        notes = decode_notes(onset_probability, frame_probability)
    except Exception as exc:
        raise TranscriptionError(f"Transcription failed: {exc}") from exc

    if csv_output_path is not None:
        save_csv(notes, csv_output_path)
    if midi_output_path is not None:
        save_midi(notes, midi_output_path)

    return notes, duration_seconds
