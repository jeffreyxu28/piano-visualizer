# Song Visualizer — Piano Transcription Visualizer

Upload solo-piano audio (or record live), run it through a trained CNN +
BiLSTM transcription model, and watch a synchronized falling-note
visualization over an 88-key keyboard while the **original** audio plays
back.

## Live demo

**https://song-visualizer.onrender.com**

Hosted on Render's free tier (512 MB RAM, 0.1 shared CPU), so a few things
to expect:

- It spins down after 15 minutes with no traffic, and the *first* request
  after that takes 1-2 minutes (server cold start + model load).
- A short clip (well under a minute) transcribes in a few seconds once
  warm. Longer songs take proportionally longer - the CPU is the
  bottleneck, not correctness - budget several minutes for a full-length
  song rather than expecting instant results.
- Uploads on the demo are capped at ~38 MB / 5 minutes (tested directly to
  confirm it doesn't crash the instance). Run it locally (below) for the
  full 100 MB / 20 minute limits and much faster processing.

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. **MP3/M4A decoding, and live microphone recording, require `ffmpeg`** on
   your system PATH (WAV/FLAC/OGG work without it via `soundfile`). The
   browser's "Record Live" feature captures audio as WebM/Opus, which also
   needs ffmpeg to decode. Install it from https://ffmpeg.org or via your
   package manager, then confirm with `ffmpeg -version`.

3. Copy your trained checkpoint into the project root so the path matches:

   ```
   piano-visualizer/best_piano_model.pt
   ```

   The app loads this file lazily on the first transcription request. If
   it's missing, uploads will fail with a clear "checkpoint not found"
   error instead of a crash.

4. Run the server:

   ```bash
   uvicorn app:app --reload
   ```

5. Open http://127.0.0.1:8000 in your browser.

## How it works

```
uploaded piano audio
  -> resample to mono 16 kHz              (backend/audio.py)
  -> log-mel spectrogram (2048/512/229)   (backend/audio.py)
  -> sliding 10s windows, overlap-averaged (backend/note_decoder.py)
  -> CNN + BiLSTM onset/frame model       (backend/model.py)
  -> onset + frame probabilities
  -> decoded note events                  (backend/note_decoder.py)
  -> JSON returned to the browser
  -> Canvas falling-note animation, synced to <audio>.currentTime
```

The frontend never generates or heuristically guesses notes — every tile
you see comes directly from the model's decoded output. The original
uploaded file is what plays back; nothing is resynthesized.

## Project layout

```
piano-visualizer/
├── app.py                  FastAPI app: routes, uploads, error handling
├── best_piano_model.pt     your trained checkpoint (you provide this)
├── requirements.txt
├── backend/
│   ├── config.py           shared constants (must match training config)
│   ├── model.py            PianoTranscriptionModel (CNN + BiLSTM)
│   ├── audio.py            decode/validate audio, log-mel spectrogram
│   ├── note_decoder.py     sliding-window inference + onset/frame decoding
│   ├── inference.py        lazy model loading + end-to-end orchestration
│   ├── jobs.py             in-memory per-upload job registry
│   └── streaming.py        HTTP range support for audio seeking
├── static/
│   ├── index.html
│   ├── styles.css
│   ├── app.js               screen state machine + control wiring
│   └── js/
│       ├── piano-layout.js  88-key geometry (white/black key positions)
│       ├── visualizer.js    Canvas rendering, driven by audio.currentTime
│       ├── particles.js     deterministic impact particle effects
│       ├── player.js        <audio> wrapper + requestAnimationFrame loop
│       └── upload.js        upload XHR with progress reporting
├── uploads/                 saved original audio per job (for playback)
└── outputs/                 generated CSV / MIDI / JSON per job
```

## Notes on accuracy

The model was trained primarily on solo-piano MAESTRO recordings. Clean,
unaccompanied piano audio (no vocals, no other instruments, minimal room
noise) will transcribe far more accurately than mixed or noisy recordings.

## What's not built yet

- **MP4 export** of the visualization + audio. The architecture is already
  set up for this: the visualizer draws everything from `(notes,
  currentTime)`, so a server-side or headless-browser renderer can drive
  the same `Visualizer` class frame-by-frame and mux the frames with the
  original audio using ffmpeg. Left out of this milestone to keep the
  interactive site the priority.
- Batch/queue processing for many simultaneous uploads (current job store
  is in-memory and single-process — fine for local/dev use).
