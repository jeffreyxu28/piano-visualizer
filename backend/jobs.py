"""In-memory registry of completed transcription jobs.

Each upload gets a job id used to serve the original audio back to the
browser for playback, plus the generated CSV/MIDI/JSON export files.
This is a simple process-local dict, which is sufficient for a
single-server local/dev deployment.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

EXTENSION_MEDIA_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".webm": "audio/webm",
}


@dataclass
class Job:
    job_id: str
    audio_path: Path
    audio_media_type: str
    original_filename: str
    csv_path: Path
    midi_path: Path
    json_path: Path


_jobs: dict[str, Job] = {}
_lock = threading.Lock()


def new_job_id() -> str:
    return uuid.uuid4().hex


def register_job(job: Job) -> None:
    with _lock:
        _jobs[job.job_id] = job


def get_job(job_id: str) -> Job | None:
    with _lock:
        return _jobs.get(job_id)
