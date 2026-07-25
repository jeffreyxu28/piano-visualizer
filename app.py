"""FastAPI application entry point.

Serves the piano-visualizer frontend and exposes the transcription API:

    POST /api/transcribe          upload audio, run the trained model, get notes back
    GET  /api/audio/{job_id}      stream the ORIGINAL uploaded audio (range-enabled)
    GET  /api/download/{job_id}/{kind}   download csv / midi / json outputs

Run with:  uvicorn app:app --reload
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from backend.audio import AudioError
from backend.config import (
    ALLOWED_EXTENSIONS,
    MAX_UPLOAD_BYTES,
    OUTPUTS_DIR,
    STATIC_DIR,
    UPLOADS_DIR,
)
from backend.inference import ModelNotAvailableError, TranscriptionError, transcribe_audio_file
from backend.jobs import EXTENSION_MEDIA_TYPES, Job, get_job, new_job_id, register_job
from backend.streaming import ranged_file_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("piano-visualizer")

app = FastAPI(title="Piano Falling-Notes Visualizer")


@app.on_event("startup")
def _ensure_directories() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Error handlers - never fail silently, always return a useful message
# ---------------------------------------------------------------------------
@app.exception_handler(AudioError)
async def handle_audio_error(request: Request, exc: AudioError):
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.exception_handler(ModelNotAvailableError)
async def handle_model_error(request: Request, exc: ModelNotAvailableError):
    logger.error("Model unavailable: %s", exc)
    return JSONResponse(status_code=503, content={"error": str(exc)})


@app.exception_handler(TranscriptionError)
async def handle_transcription_error(request: Request, exc: TranscriptionError):
    logger.exception("Transcription failed")
    return JSONResponse(status_code=422, content={"error": str(exc)})


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file was uploaded.")

    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{extension}'. Supported formats: "
                f"{', '.join(sorted(ALLOWED_EXTENSIONS))}."
            ),
        )

    job_id = new_job_id()
    audio_path = UPLOADS_DIR / f"{job_id}{extension}"

    size = 0
    try:
        with open(audio_path, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
                    )
                out.write(chunk)
    except HTTPException:
        audio_path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    if size == 0:
        audio_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    csv_path = OUTPUTS_DIR / f"{job_id}.csv"
    midi_path = OUTPUTS_DIR / f"{job_id}.mid"
    json_path = OUTPUTS_DIR / f"{job_id}.json"

    try:
        notes, duration_seconds = await run_in_threadpool(
            transcribe_audio_file, audio_path, csv_path, midi_path
        )
    except (AudioError, ModelNotAvailableError, TranscriptionError):
        audio_path.unlink(missing_ok=True)
        raise

    json_path.write_text(json.dumps(notes, indent=2), encoding="utf-8")

    job = Job(
        job_id=job_id,
        audio_path=audio_path,
        audio_media_type=EXTENSION_MEDIA_TYPES.get(extension, "application/octet-stream"),
        original_filename=file.filename,
        csv_path=csv_path,
        midi_path=midi_path,
        json_path=json_path,
    )
    register_job(job)

    return {
        "job_id": job_id,
        "note_count": len(notes),
        "duration_sec": round(duration_seconds, 3),
        "notes": notes,
        "audio_url": f"/api/audio/{job_id}",
        "csv_url": f"/api/download/{job_id}/csv",
        "midi_url": f"/api/download/{job_id}/midi",
        "json_url": f"/api/download/{job_id}/json",
    }


@app.get("/api/audio/{job_id}")
async def get_audio(job_id: str, request: Request):
    job = get_job(job_id)
    if job is None or not job.audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio not found for this job.")
    return ranged_file_response(request, job.audio_path, job.audio_media_type)


_DOWNLOAD_KINDS = {
    "csv": ("csv_path", "text/csv"),
    "midi": ("midi_path", "audio/midi"),
    "json": ("json_path", "application/json"),
}


@app.get("/api/download/{job_id}/{kind}")
async def download(job_id: str, kind: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if kind not in _DOWNLOAD_KINDS:
        raise HTTPException(status_code=400, detail=f"Unknown export type '{kind}'.")

    attr, media_type = _DOWNLOAD_KINDS[kind]
    path: Path = getattr(job, attr)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{kind} export not found.")

    stem = Path(job.original_filename).stem or job.job_id
    extension = {"csv": "csv", "midi": "mid", "json": "json"}[kind]
    return FileResponse(
        path,
        media_type=media_type,
        filename=f"{stem}_transcription.{extension}",
    )


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")
