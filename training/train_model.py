"""Offline training script - reproduces the MAESTRO training recipe used to
produce best_piano_model.pt for the web app's inference pipeline.

This is intentionally NOT part of the running FastAPI app (backend/) - the
app only ever does inference against an already-trained checkpoint. Run
this manually, once, to (re)produce that checkpoint:

    python training/train_model.py

Progress is printed as it goes (unbuffered) so it can be tailed from a log
file when run in the background. The subset size / epoch count can be
overridden with environment variables (see the *_DEFAULT constants below)
without editing this file, e.g. for a quick smoke test:

    PV_TRAIN_RECORDINGS=2 PV_VAL_RECORDINGS=1 PV_TEST_RECORDINGS=1 \
    PV_TRAIN_SAMPLES=4 PV_VAL_SAMPLES=2 PV_TEST_SAMPLES=2 PV_EPOCHS=1 \
    python training/train_model.py

Model architecture, spectrogram settings, and key range are imported from
backend/ so the trained checkpoint is guaranteed to match what the
inference pipeline expects - there is exactly one definition of the model
and one definition of the audio preprocessing, shared by training and
serving.
"""

from __future__ import annotations

import os
import random
import sys
import time
from functools import lru_cache
from pathlib import Path

import kagglehub
import librosa
import numpy as np
import pandas as pd
import pretty_midi
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.audio import waveform_to_log_mel
from backend.config import (
    CHECKPOINT_PATH,
    CHUNK_SECONDS,
    HOP_LENGTH,
    MAX_MIDI,
    MIN_MIDI,
    N_KEYS,
    SAMPLE_RATE,
)
from backend.model import PianoTranscriptionModel

# ---------------------------------------------------------------------------
# Config - overridable via environment variables for smoke testing.
# ---------------------------------------------------------------------------
DATASET_HANDLE = "alonhaviv/the-maestro-dataset-v3-0-0"
DATA_DIR = Path(__file__).resolve().parent / "maestro_data"
SUBSET_METADATA_PATH = DATA_DIR / "subset_metadata.csv"

DOWNLOAD_TRAIN_RECORDINGS = int(os.environ.get("PV_TRAIN_RECORDINGS", 150))
DOWNLOAD_VAL_RECORDINGS = int(os.environ.get("PV_VAL_RECORDINGS", 40))
DOWNLOAD_TEST_RECORDINGS = int(os.environ.get("PV_TEST_RECORDINGS", 40))

TRAIN_SAMPLES_PER_EPOCH = int(os.environ.get("PV_TRAIN_SAMPLES", 600))
VAL_SAMPLES = int(os.environ.get("PV_VAL_SAMPLES", 120))
TEST_SAMPLES = int(os.environ.get("PV_TEST_SAMPLES", 180))
EPOCHS = int(os.environ.get("PV_EPOCHS", 18))
BATCH_SIZE = int(os.environ.get("PV_BATCH_SIZE", 4))
LEARNING_RATE = 3e-4

ONSET_POS_WEIGHT = 10.0
FRAME_POS_WEIGHT = 3.0

RANDOM_SEED = 2026


def seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Dataset download (Kaggle, anonymous/public access)
# ---------------------------------------------------------------------------
def locate_downloaded_file(result: str | Path, filename: str) -> Path | None:
    result_path = Path(result)
    if result_path.is_file():
        return result_path.resolve()
    if result_path.exists() and result_path.is_dir():
        matches = list(result_path.rglob(filename))
        if matches:
            return matches[0].resolve()
    return None


DOWNLOAD_RETRIES = 5
DOWNLOAD_RETRY_BASE_DELAY = 5.0  # seconds; doubles each retry


def download_kaggle_file(candidates: list[str]) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    for kaggle_path in candidates:
        expected = DATA_DIR / Path(kaggle_path)
        if expected.exists() and expected.is_file():
            return expected.resolve()

        for attempt in range(1, DOWNLOAD_RETRIES + 1):
            try:
                result = kagglehub.dataset_download(
                    DATASET_HANDLE, path=kaggle_path, output_dir=str(DATA_DIR)
                )
                found = locate_downloaded_file(result, Path(kaggle_path).name)
                if found is not None:
                    return found
                break  # download "succeeded" but file wasn't found - try next candidate
            except Exception as exc:
                if attempt < DOWNLOAD_RETRIES:
                    delay = DOWNLOAD_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    print(
                        f"  retry {attempt}/{DOWNLOAD_RETRIES} for {kaggle_path} "
                        f"after error ({exc}); waiting {delay:.0f}s",
                        flush=True,
                    )
                    time.sleep(delay)
                else:
                    errors.append(f"{kaggle_path}: {exc}")

    raise RuntimeError(
        "Could not download the requested MAESTRO file. Tried:\n" + "\n".join(errors)
    )


def download_official_metadata() -> Path:
    print("\nGetting MAESTRO metadata...", flush=True)
    return download_kaggle_file(
        ["maestro-v3.0.0/maestro-v3.0.0.csv", "maestro-v3.0.0.csv"]
    )


def dataset_file_candidates(relative_path: str) -> list[str]:
    relative_path = relative_path.replace("\\", "/")
    return [f"maestro-v3.0.0/{relative_path}", relative_path]


def choose_rows(metadata: pd.DataFrame, split: str, amount: int, seed: int) -> pd.DataFrame:
    rows = metadata[metadata["split"] == split].copy()
    if rows.empty:
        raise ValueError(f"MAESTRO metadata contains no '{split}' rows.")
    amount = min(amount, len(rows))
    return rows.sample(n=amount, random_state=seed).reset_index(drop=True)


def download_subset() -> None:
    seed_everything()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    metadata_path = download_official_metadata()
    metadata = pd.read_csv(metadata_path)

    required = {"split", "audio_filename", "midi_filename", "duration"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")

    selections = [
        choose_rows(metadata, "train", DOWNLOAD_TRAIN_RECORDINGS, RANDOM_SEED),
        choose_rows(metadata, "validation", DOWNLOAD_VAL_RECORDINGS, RANDOM_SEED + 1),
        choose_rows(metadata, "test", DOWNLOAD_TEST_RECORDINGS, RANDOM_SEED + 2),
    ]
    subset = pd.concat(selections, ignore_index=True)

    print(
        f"\nSelected {len(subset)} performances: "
        f"{DOWNLOAD_TRAIN_RECORDINGS} train, {DOWNLOAD_VAL_RECORDINGS} validation, "
        f"{DOWNLOAD_TEST_RECORDINGS} test.",
        flush=True,
    )

    resolved_rows: list[dict] = []
    skipped = 0

    for index, row in subset.iterrows():
        print(f"[{index + 1}/{len(subset)}] {row['split']} - {row['audio_filename']}", flush=True)
        try:
            audio_path = download_kaggle_file(dataset_file_candidates(str(row["audio_filename"])))
            midi_path = download_kaggle_file(dataset_file_candidates(str(row["midi_filename"])))
        except Exception as exc:
            skipped += 1
            print(f"  SKIPPING this performance after repeated failures: {exc}", flush=True)
            continue

        record = row.to_dict()
        record["local_audio"] = str(audio_path)
        record["local_midi"] = str(midi_path)
        resolved_rows.append(record)

        # Persist after every file so a crash/network drop never loses more
        # than the single in-flight download - already-resolved rows are
        # cached to disk by kagglehub and won't be re-fetched on restart.
        pd.DataFrame(resolved_rows).to_csv(SUBSET_METADATA_PATH, index=False)

    resolved = pd.DataFrame(resolved_rows)
    if resolved.empty:
        raise RuntimeError("Every performance in the subset failed to download.")

    min_required = {
        "train": max(1, DOWNLOAD_TRAIN_RECORDINGS // 2),
        "validation": max(1, DOWNLOAD_VAL_RECORDINGS // 2),
        "test": max(1, DOWNLOAD_TEST_RECORDINGS // 2),
    }
    counts = resolved["split"].value_counts()
    for split, minimum in min_required.items():
        if counts.get(split, 0) < minimum:
            raise RuntimeError(
                f"Too many '{split}' downloads failed ({counts.get(split, 0)}/"
                f"{minimum} minimum needed) - check network connectivity and retry."
            )

    print(f"\nSubset ready: {SUBSET_METADATA_PATH.resolve()} ({skipped} performances skipped)", flush=True)
    print(resolved["split"].value_counts(), flush=True)


def load_subset_metadata() -> pd.DataFrame:
    if not SUBSET_METADATA_PATH.exists():
        print("Dataset subset has not been downloaded yet, so downloading it now.", flush=True)
        download_subset()

    metadata = pd.read_csv(SUBSET_METADATA_PATH)

    missing_files = []
    for _, row in metadata.iterrows():
        if not Path(str(row["local_audio"])).exists():
            missing_files.append(str(row["local_audio"]))
        if not Path(str(row["local_midi"])).exists():
            missing_files.append(str(row["local_midi"]))

    if missing_files:
        print("Some subset files are missing. Rebuilding the subset...", flush=True)
        download_subset()
        metadata = pd.read_csv(SUBSET_METADATA_PATH)

    return metadata


# ---------------------------------------------------------------------------
# Labels / chunking
# ---------------------------------------------------------------------------
@lru_cache(maxsize=512)
def load_midi_notes(midi_path_string: str) -> tuple[tuple[int, float, float], ...]:
    midi = pretty_midi.PrettyMIDI(midi_path_string)
    notes: list[tuple[int, float, float]] = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            if MIN_MIDI <= note.pitch <= MAX_MIDI and note.end > note.start:
                notes.append((note.pitch, float(note.start), float(note.end)))
    notes.sort(key=lambda item: item[1])
    return tuple(notes)


def read_audio_chunk(audio_path: Path, start_sec: float, duration_sec: float) -> np.ndarray:
    with sf.SoundFile(str(audio_path)) as audio_file:
        original_sr = int(audio_file.samplerate)
        start_frame = max(0, int(round(start_sec * original_sr)))
        start_frame = min(start_frame, len(audio_file))
        number_of_frames = int(np.ceil(duration_sec * original_sr))

        audio_file.seek(start_frame)
        audio = audio_file.read(frames=number_of_frames, dtype="float32", always_2d=True)

    mono = (
        np.zeros(0, dtype=np.float32)
        if audio.size == 0
        else audio.mean(axis=1).astype(np.float32)
    )

    if original_sr != SAMPLE_RATE and len(mono) > 0:
        mono = librosa.resample(mono, orig_sr=original_sr, target_sr=SAMPLE_RATE).astype(np.float32)

    target_samples = int(round(duration_sec * SAMPLE_RATE))
    if len(mono) < target_samples:
        mono = np.pad(mono, (0, target_samples - len(mono)))
    else:
        mono = mono[:target_samples]

    return mono.astype(np.float32)


def build_labels(midi_path: Path, chunk_start_sec: float, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    onset = np.zeros((num_frames, N_KEYS), dtype=np.float32)
    frame = np.zeros((num_frames, N_KEYS), dtype=np.float32)

    chunk_end_sec = chunk_start_sec + CHUNK_SECONDS
    seconds_per_frame = HOP_LENGTH / SAMPLE_RATE

    for pitch, note_start, note_end in load_midi_notes(str(midi_path)):
        if note_end <= chunk_start_sec or note_start >= chunk_end_sec:
            continue

        key_index = pitch - MIN_MIDI
        local_start = note_start - chunk_start_sec
        local_end = note_end - chunk_start_sec

        active_start = max(0.0, local_start)
        active_end = min(CHUNK_SECONDS, local_end)

        start_index = max(0, int(np.floor(active_start / seconds_per_frame)))
        end_index = min(num_frames, int(np.ceil(active_end / seconds_per_frame)))

        if end_index > start_index:
            frame[start_index:end_index, key_index] = 1.0

        if 0.0 <= local_start < CHUNK_SECONDS:
            onset_index = int(round(local_start / seconds_per_frame))
            if 0 <= onset_index < num_frames:
                onset[onset_index, key_index] = 1.0

    return onset, frame


class MaestroChunkDataset(Dataset):
    def __init__(self, split: str, samples_per_epoch: int, training: bool):
        metadata = load_subset_metadata()
        metadata = metadata[metadata["split"] == split].reset_index(drop=True)

        if metadata.empty:
            raise ValueError(f"No downloaded rows found for split='{split}'.")

        self.metadata = metadata
        self.samples_per_epoch = samples_per_epoch
        self.training = training
        self.weights = metadata["duration"].astype(float).clip(lower=1.0).tolist()
        self.fixed_chunks: list[tuple[int, float]] = []

        if not training:
            rng = random.Random(RANDOM_SEED + (1 if split == "validation" else 2))
            population = list(range(len(metadata)))
            for _ in range(samples_per_epoch):
                row_index = rng.choices(population, weights=self.weights, k=1)[0]
                duration = float(metadata.iloc[row_index]["duration"])
                max_start = max(0.0, duration - CHUNK_SECONDS)
                start = rng.uniform(0.0, max_start) if max_start > 0 else 0.0
                self.fixed_chunks.append((row_index, start))

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _choose_chunk(self, index: int) -> tuple[pd.Series, float]:
        if self.training:
            row_index = random.choices(range(len(self.metadata)), weights=self.weights, k=1)[0]
            row = self.metadata.iloc[row_index]
            duration = float(row["duration"])
            max_start = max(0.0, duration - CHUNK_SECONDS)
            start = random.uniform(0.0, max_start) if max_start > 0 else 0.0
            return row, start

        row_index, start = self.fixed_chunks[index]
        return self.metadata.iloc[row_index], start

    def __getitem__(self, index: int):
        row, chunk_start = self._choose_chunk(index)
        audio_path = Path(str(row["local_audio"]))
        midi_path = Path(str(row["local_midi"]))

        audio = read_audio_chunk(audio_path, chunk_start, CHUNK_SECONDS)
        log_mel = waveform_to_log_mel(audio)
        onset, frame = build_labels(midi_path, chunk_start, len(log_mel))

        return torch.from_numpy(log_mel), torch.from_numpy(onset), torch.from_numpy(frame)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
def add_counts(logits: torch.Tensor, targets: torch.Tensor, totals: list[int]) -> None:
    predictions = torch.sigmoid(logits) >= 0.5
    truth = targets >= 0.5
    totals[0] += int((predictions & truth).sum().item())
    totals[1] += int((predictions & ~truth).sum().item())
    totals[2] += int((~predictions & truth).sum().item())


def f1_from_counts(totals: list[int]) -> float:
    tp, fp, fn = totals
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else (2 * tp) / denominator


@torch.no_grad()
def validate_model(model, loader, onset_loss_function, frame_loss_function, device):
    model.eval()
    total_loss = 0.0
    onset_counts = [0, 0, 0]
    frame_counts = [0, 0, 0]

    for mel, onset_target, frame_target in loader:
        mel = mel.to(device)
        onset_target = onset_target.to(device)
        frame_target = frame_target.to(device)

        onset_logits, frame_logits = model(mel)
        loss = onset_loss_function(onset_logits, onset_target) + frame_loss_function(frame_logits, frame_target)
        total_loss += float(loss.item())

        add_counts(onset_logits, onset_target, onset_counts)
        add_counts(frame_logits, frame_target, frame_counts)

    return (
        total_loss / max(1, len(loader)),
        f1_from_counts(onset_counts),
        f1_from_counts(frame_counts),
    )


def make_loss_functions(device: torch.device) -> tuple[nn.Module, nn.Module]:
    onset_pos_weight = torch.full((N_KEYS,), ONSET_POS_WEIGHT, device=device)
    frame_pos_weight = torch.full((N_KEYS,), FRAME_POS_WEIGHT, device=device)
    return (
        nn.BCEWithLogitsLoss(pos_weight=onset_pos_weight),
        nn.BCEWithLogitsLoss(pos_weight=frame_pos_weight),
    )


def train_model() -> None:
    seed_everything()
    device = get_device()
    print(f"\nTraining device: {device}", flush=True)
    if device.type == "cpu":
        print("Training will work on CPU, but a CUDA-capable GPU will be much faster.", flush=True)

    train_dataset = MaestroChunkDataset("train", TRAIN_SAMPLES_PER_EPOCH, True)
    validation_dataset = MaestroChunkDataset("validation", VAL_SAMPLES, False)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = PianoTranscriptionModel().to(device)
    onset_loss_function, frame_loss_function = make_loss_functions(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_validation_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for step, (mel, onset_target, frame_target) in enumerate(progress, start=1):
            mel = mel.to(device)
            onset_target = onset_target.to(device)
            frame_target = frame_target.to(device)

            optimizer.zero_grad(set_to_none=True)
            onset_logits, frame_logits = model(mel)

            onset_loss = onset_loss_function(onset_logits, onset_target)
            frame_loss = frame_loss_function(frame_logits, frame_target)
            loss = onset_loss + frame_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.item())
            progress.set_postfix(loss=f"{running_loss / step:.4f}")

        validation_loss, onset_f1, frame_f1 = validate_model(
            model, validation_loader, onset_loss_function, frame_loss_function, device
        )

        print(
            f"Epoch {epoch}: val_loss={validation_loss:.4f} | "
            f"onset_F1={onset_f1:.4f} | frame_F1={frame_f1:.4f}",
            flush=True,
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": validation_loss,
                    "onset_f1": onset_f1,
                    "frame_f1": frame_f1,
                },
                CHECKPOINT_PATH,
            )
            print(f"Saved new best model -> {CHECKPOINT_PATH.resolve()}", flush=True)

    print("\nTraining complete.", flush=True)


def load_trained_model(device: torch.device) -> PianoTranscriptionModel:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"No trained model exists at {CHECKPOINT_PATH}. Train the model first.")
    model = PianoTranscriptionModel().to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate_test_set() -> None:
    device = get_device()
    model = load_trained_model(device)

    test_dataset = MaestroChunkDataset("test", TEST_SAMPLES, False)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    onset_loss_function, frame_loss_function = make_loss_functions(device)
    test_loss, onset_f1, frame_f1 = validate_model(
        model, test_loader, onset_loss_function, frame_loss_function, device
    )

    print("\nTEST RESULTS", flush=True)
    print(f"Loss:      {test_loss:.4f}", flush=True)
    print(f"Onset F1:  {onset_f1:.4f}", flush=True)
    print(f"Frame F1:  {frame_f1:.4f}", flush=True)


def main() -> None:
    print("===== 1. DOWNLOAD DATASET SUBSET =====", flush=True)
    print(
        f"train={DOWNLOAD_TRAIN_RECORDINGS} val={DOWNLOAD_VAL_RECORDINGS} "
        f"test={DOWNLOAD_TEST_RECORDINGS} | epochs={EPOCHS} "
        f"train_samples/epoch={TRAIN_SAMPLES_PER_EPOCH} val_samples={VAL_SAMPLES} "
        f"test_samples={TEST_SAMPLES}",
        flush=True,
    )
    download_subset()

    print("\n===== 2. TRAIN MODEL =====", flush=True)
    train_model()

    print("\n===== 3. EVALUATE TEST SET =====", flush=True)
    evaluate_test_set()

    print("\nDone. best_piano_model.pt is ready for the web app.", flush=True)


if __name__ == "__main__":
    main()
