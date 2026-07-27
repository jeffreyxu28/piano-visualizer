"""Turn model onset/frame probabilities into discrete note events.

Ported from the training notebook's inference cell: sliding-window
prediction over long spectrograms (with overlap-averaging), then a
stateful per-key decoder that converts onset/frame probability curves
into (pitch, start, end) note events.
"""

from __future__ import annotations

import csv
import gc
import os
from pathlib import Path

import numpy as np
import pretty_midi
import torch

from backend.audio import waveform_to_log_mel
from backend.config import (
    CHUNK_SECONDS,
    FRAME_ONLY_START_THRESHOLD,
    FRAME_THRESHOLD,
    HOP_LENGTH,
    MIN_MIDI,
    MIN_START_FRAME_PROB,
    N_KEYS,
    OFF_FRAMES_REQUIRED,
    ONSET_THRESHOLD,
    SAMPLE_RATE,
    WINDOW_OVERLAP_SECONDS,
)
from backend.model import PianoTranscriptionModel


@torch.no_grad()
def predict_probabilities(
    model: PianoTranscriptionModel,
    waveform: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over long audio using overlapping windows, averaging
    predictions in the overlap regions so chunk boundaries stay smooth.

    The log-mel spectrogram is computed one ~10s window at a time (matching
    the model's own chunking) rather than for the whole file up front. For
    a long recording, one big STFT over the entire signal is the single
    largest memory allocation in the whole pipeline; computing it window by
    window keeps peak memory bounded to a single window's spectrogram
    instead of scaling with the song's length.
    """
    total_frames = 1 + len(waveform) // HOP_LENGTH
    window_frames = int(CHUNK_SECONDS * SAMPLE_RATE / HOP_LENGTH) + 1
    overlap_frames = max(1, int(WINDOW_OVERLAP_SECONDS * SAMPLE_RATE / HOP_LENGTH))
    stride = max(1, window_frames - overlap_frames)

    onset_sum = np.zeros((total_frames, N_KEYS), dtype=np.float32)
    frame_sum = np.zeros((total_frames, N_KEYS), dtype=np.float32)
    counts = np.zeros((total_frames, 1), dtype=np.float32)

    # On a memory-constrained host, PyTorch's CPU allocator can retain
    # freed tensor memory in its own internal pool rather than returning it
    # to the OS between forward passes. That's invisible for a short clip
    # (a handful of windows) but on a long recording (dozens of windows in
    # one request) the retained-but-idle pool can grow enough to matter.
    # Dropping references and forcing a collection periodically keeps that
    # bounded instead of climbing for the whole request. Opt-in via env var
    # so normal local runs (with headroom to spare) skip the overhead.
    gc_interval = int(os.environ.get("INFERENCE_GC_INTERVAL", "0"))
    window_index = 0

    for start in range(0, total_frames, stride):
        end = min(total_frames, start + window_frames)
        needed_frames = end - start

        start_sample = start * HOP_LENGTH
        end_sample = min(len(waveform), start_sample + needed_frames * HOP_LENGTH)
        chunk_waveform = waveform[start_sample:end_sample]
        if len(chunk_waveform) == 0:
            break

        chunk_mel = waveform_to_log_mel(chunk_waveform)[:needed_frames]
        actual_end = start + len(chunk_mel)
        if len(chunk_mel) == 0:
            break

        chunk_tensor = torch.from_numpy(chunk_mel).unsqueeze(0).to(device)
        onset_logits, frame_logits = model(chunk_tensor)
        onset_probability = torch.sigmoid(onset_logits)[0].cpu().numpy()
        frame_probability = torch.sigmoid(frame_logits)[0].cpu().numpy()

        onset_sum[start:actual_end] += onset_probability
        frame_sum[start:actual_end] += frame_probability
        counts[start:actual_end] += 1.0

        del chunk_waveform, chunk_mel, chunk_tensor
        del onset_logits, frame_logits, onset_probability, frame_probability

        window_index += 1
        if gc_interval and window_index % gc_interval == 0:
            gc.collect()

        if end >= total_frames:
            break

    counts = np.maximum(counts, 1.0)
    return onset_sum / counts, frame_sum / counts


def make_note(pitch: int, start_frame: int, end_frame: int, onset_confidence: float) -> dict:
    seconds_per_frame = HOP_LENGTH / SAMPLE_RATE
    start_sec = start_frame * seconds_per_frame
    end_sec = max(start_sec + seconds_per_frame, end_frame * seconds_per_frame)

    return {
        "midi_note": pitch,
        "note_name": pretty_midi.note_number_to_name(pitch),
        "start_sec": round(float(start_sec), 4),
        "end_sec": round(float(end_sec), 4),
        "duration_sec": round(float(end_sec - start_sec), 4),
        "onset_confidence": round(float(onset_confidence), 4),
    }


def decode_notes(
    onset_probability: np.ndarray,
    frame_probability: np.ndarray,
) -> list[dict]:
    notes: list[dict] = []

    onset_binary = onset_probability >= ONSET_THRESHOLD
    onset_events = onset_binary.copy()
    if len(onset_events) > 1:
        onset_events[1:] = onset_binary[1:] & ~onset_binary[:-1]

    for key_index in range(N_KEYS):
        pitch = MIN_MIDI + key_index
        active = False
        start_frame = 0
        start_confidence = 0.0
        off_count = 0
        previous_frame_active = False

        for t in range(len(frame_probability)):
            onset_now = bool(onset_events[t, key_index])
            frame_prob = float(frame_probability[t, key_index])
            frame_now = frame_prob >= FRAME_THRESHOLD

            frame_started = frame_now and not previous_frame_active
            should_start = (onset_now and frame_prob >= MIN_START_FRAME_PROB) or (
                frame_started and frame_prob >= FRAME_ONLY_START_THRESHOLD
            )

            if not active:
                if should_start:
                    active = True
                    start_frame = t
                    start_confidence = float(onset_probability[t, key_index])
                    off_count = 0
                previous_frame_active = frame_now
                continue

            if onset_now and t > start_frame + 1:
                notes.append(make_note(pitch, start_frame, t, start_confidence))
                start_frame = t
                start_confidence = float(onset_probability[t, key_index])
                off_count = 0
                previous_frame_active = frame_now
                continue

            if frame_now:
                off_count = 0
            else:
                off_count += 1
                if off_count >= OFF_FRAMES_REQUIRED:
                    end_frame = max(start_frame + 1, t - OFF_FRAMES_REQUIRED + 1)
                    notes.append(make_note(pitch, start_frame, end_frame, start_confidence))
                    active = False
                    off_count = 0

            previous_frame_active = frame_now

        if active:
            notes.append(
                make_note(
                    pitch,
                    start_frame,
                    max(start_frame + 1, len(frame_probability) - 1),
                    start_confidence,
                )
            )

    notes.sort(key=lambda note: float(note["start_sec"]))
    return notes


def save_midi(notes: list[dict], output_path: Path) -> None:
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0, name="Piano")

    for note in notes:
        piano.notes.append(
            pretty_midi.Note(
                velocity=max(1, min(127, int(round(80 * note["onset_confidence"] + 20)))),
                pitch=int(note["midi_note"]),
                start=float(note["start_sec"]),
                end=float(note["end_sec"]),
            )
        )

    midi.instruments.append(piano)
    midi.write(str(output_path))


def save_csv(notes: list[dict], output_path: Path) -> None:
    fieldnames = [
        "midi_note",
        "note_name",
        "start_sec",
        "end_sec",
        "duration_sec",
        "onset_confidence",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for note in notes:
            writer.writerow(note)
