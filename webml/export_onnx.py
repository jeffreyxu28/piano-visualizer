"""One-off dev script: export the trained PyTorch model to ONNX for
in-browser inference, and validate the ONNX output matches PyTorch exactly
on real data before any JS work depends on it.

Not part of the served app - run manually:
    python webml/export_onnx.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import onnxruntime as ort
import torch

from backend.config import CHECKPOINT_PATH, N_MELS
from backend.model import PianoTranscriptionModel

OUTPUT_DIR = Path(__file__).resolve().parent
ONNX_PATH = OUTPUT_DIR / "piano_model.onnx"


def load_model() -> PianoTranscriptionModel:
    model = PianoTranscriptionModel()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export(model: PianoTranscriptionModel) -> None:
    dummy = torch.randn(1, 313, N_MELS)  # a representative ~10s window
    torch.onnx.export(
        model,
        (dummy,),
        str(ONNX_PATH),
        input_names=["mel"],
        output_names=["onset_logits", "frame_logits"],
        dynamic_axes={
            "mel": {1: "time"},
            "onset_logits": {1: "time"},
            "frame_logits": {1: "time"},
        },
        opset_version=17,
        dynamo=False,  # the newer dynamo-based exporter baked the window
        # size in as a constant instead of honoring dynamic_axes for this
        # LSTM-containing model; the legacy TorchScript-based exporter has
        # much more mature RNN/LSTM support and correctly stays dynamic.
    )
    print(f"Exported: {ONNX_PATH} ({ONNX_PATH.stat().st_size / 1e6:.2f} MB)")


def validate(model: PianoTranscriptionModel) -> None:
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(0)
    max_onset_diff = 0.0
    max_frame_diff = 0.0

    for trial, time_steps in enumerate([1, 5, 50, 200, 313, 500]):
        mel = rng.standard_normal((1, time_steps, N_MELS)).astype(np.float32)

        with torch.no_grad():
            torch_onset, torch_frame = model(torch.from_numpy(mel))
        torch_onset = torch_onset.numpy()
        torch_frame = torch_frame.numpy()

        onnx_onset, onnx_frame = session.run(
            ["onset_logits", "frame_logits"], {"mel": mel}
        )

        onset_diff = float(np.abs(torch_onset - onnx_onset).max())
        frame_diff = float(np.abs(torch_frame - onnx_frame).max())
        max_onset_diff = max(max_onset_diff, onset_diff)
        max_frame_diff = max(max_frame_diff, frame_diff)

        print(
            f"trial {trial} (time_steps={time_steps}): "
            f"onset_max_diff={onset_diff:.2e} frame_max_diff={frame_diff:.2e}"
        )

    print(f"\nOverall max diff - onset: {max_onset_diff:.2e}, frame: {max_frame_diff:.2e}")
    if max_onset_diff < 1e-3 and max_frame_diff < 1e-3:
        print("PASS: ONNX output matches PyTorch within tight tolerance.")
    else:
        print("FAIL: ONNX output diverges from PyTorch - do not ship this export.")
        sys.exit(1)


if __name__ == "__main__":
    print("Loading trained checkpoint...")
    model = load_model()
    print("Exporting to ONNX...")
    export(model)
    print("\nValidating ONNX output against PyTorch on random inputs...")
    validate(model)
