import { Visualizer } from "./js/visualizer.js?v=5";
import { Player } from "./js/player.js?v=4";
import { transcribeFile } from "./js/upload.js?v=3";
import { Recorder, isRecordingSupported } from "./js/recorder.js";

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------
const screens = {
  landing: document.getElementById("screen-landing"),
  loading: document.getElementById("screen-loading"),
  error: document.getElementById("screen-error"),
  visualizer: document.getElementById("screen-visualizer"),
};

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");

const modeBtnUpload = document.getElementById("mode-btn-upload");
const modeBtnRecord = document.getElementById("mode-btn-record");
const uploadPanel = document.getElementById("upload-panel");
const recordPanel = document.getElementById("record-panel");

const recordIdleBlock = document.getElementById("record-idle-block");
const recordActiveBlock = document.getElementById("record-active-block");
const recordPreviewBlock = document.getElementById("record-preview-block");
const recordStartBtn = document.getElementById("record-start-btn");
const recordStopBtn = document.getElementById("record-stop-btn");
const recordTimer = document.getElementById("record-timer");
const recordPreviewAudio = document.getElementById("record-preview-audio");
const recordDuration = document.getElementById("record-duration");
const recordUseBtn = document.getElementById("record-use-btn");
const recordRerecordBtn = document.getElementById("record-rerecord-btn");
const recordError = document.getElementById("record-error");

const loadingFilename = document.getElementById("loading-filename");
const loadingStatus = document.getElementById("loading-status");
const loadingBarFill = document.getElementById("loading-bar-fill");

const errorMessageEl = document.getElementById("error-message");
const errorRetryBtn = document.getElementById("error-retry");

const vizFilename = document.getElementById("viz-filename");
const vizStage = document.getElementById("viz-stage");
const vizCanvas = document.getElementById("viz-canvas");
const vizEmptyHint = document.getElementById("viz-empty-hint");

const audioEl = document.getElementById("audio-player");

const btnPlay = document.getElementById("btn-play");
const iconPlay = document.getElementById("icon-play");
const iconPause = document.getElementById("icon-pause");
const btnRestart = document.getElementById("btn-restart");
const btnFullscreen = document.getElementById("btn-fullscreen");

const seekBar = document.getElementById("seek-bar");
const timeCurrent = document.getElementById("time-current");
const timeTotal = document.getElementById("time-total");
const volumeBar = document.getElementById("volume-bar");

const volumeValue = document.getElementById("volume-value");

const lookaheadSlider = document.getElementById("lookahead-slider");
const lookaheadValue = document.getElementById("lookahead-value");
const speedSelect = document.getElementById("speed-select");
const toggleLabels = document.getElementById("toggle-labels");
const toggleEffects = document.getElementById("toggle-effects");
const togglePitchColor = document.getElementById("toggle-pitch-color");
const noteColorPicker = document.getElementById("note-color-picker");
const noteColorOption = document.getElementById("note-color-option");

const newFileBtn = document.getElementById("new-file-btn");
const downloadBtn = document.getElementById("download-btn");
const downloadMenu = document.getElementById("download-menu");
const downloadCsv = document.getElementById("download-csv");
const downloadMidi = document.getElementById("download-midi");
const downloadJson = document.getElementById("download-json");

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let visualizer = null;
let player = null;
let isSeeking = false;

let recorder = null;
let recordedBlob = null;
let recordedExtension = null;
let recordedObjectUrl = null;

function showScreen(name) {
  for (const [key, el] of Object.entries(screens)) {
    el.classList.toggle("screen-active", key === name);
  }
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) seconds = 0;
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

// ---------------------------------------------------------------------------
// Upload flow
// ---------------------------------------------------------------------------
const ALLOWED_EXTENSIONS = [".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm"];

function hasAllowedExtension(filename) {
  const lower = filename.toLowerCase();
  return ALLOWED_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

async function handleFileSelected(file) {
  if (!file) return;

  if (!hasAllowedExtension(file.name)) {
    showError(
      `Unsupported file type. Please upload one of: ${ALLOWED_EXTENSIONS.join(", ")}.`
    );
    return;
  }

  loadingFilename.textContent = file.name;
  loadingStatus.textContent = "Uploading audio…";
  loadingBarFill.style.animation = "none";
  loadingBarFill.style.width = "0%";
  showScreen("loading");

  try {
    const result = await transcribeFile(file, ({ phase, pct }) => {
      if (phase === "uploading") {
        loadingStatus.textContent = "Uploading audio…";
        loadingBarFill.style.width = `${Math.min(100, pct * 60)}%`;
      } else if (phase === "processing") {
        loadingStatus.textContent =
          "Running the transcription model (CNN + BiLSTM onset/frame detection)…";
        loadingBarFill.style.animation = "";
        loadingBarFill.style.width = "";
      }
    });

    await enterVisualizer(file.name, result);
  } catch (err) {
    showError(err.message || "Transcription failed.");
  }
}

function showError(message) {
  errorMessageEl.textContent = message;
  showScreen("error");
}

// ---------------------------------------------------------------------------
// Visualizer setup
// ---------------------------------------------------------------------------
async function enterVisualizer(filename, result) {
  vizFilename.textContent = `${filename} · ${result.note_count} notes · ${formatTime(result.duration_sec)}`;

  downloadCsv.href = result.csv_url;
  downloadMidi.href = result.midi_url;
  downloadJson.href = result.json_url;

  showScreen("visualizer");

  if (!visualizer) {
    visualizer = new Visualizer(vizCanvas, vizStage);
    player = new Player(audioEl, visualizer, {
      onTick: handleTick,
      onPlayStateChange: handlePlayStateChange,
      onDurationChange: handleDurationChange,
      onError: () => showError("Playback error: the audio could not be played."),
    });
    applyOptionsFromControls();
  }

  visualizer.setNotes(result.notes);
  vizEmptyHint.classList.remove("hidden");

  try {
    await player.load(result.audio_url);
  } catch {
    showError("The transcribed audio could not be loaded for playback.");
    return;
  }

  seekBar.max = String(player.duration || result.duration_sec || 0);
  timeTotal.textContent = formatTime(player.duration || result.duration_sec);
  timeCurrent.textContent = "0:00";
  seekBar.value = "0";
}

function applyOptionsFromControls() {
  visualizer.setOptions({
    lookaheadSeconds: parseFloat(lookaheadSlider.value),
    showLabels: toggleLabels.checked,
    effectsEnabled: toggleEffects.checked,
    colorByPitch: togglePitchColor.checked,
  });
  visualizer.setCustomColor(noteColorPicker.value);
  updateNoteColorOptionState();
  player.setVolume(parseFloat(volumeBar.value));
  player.setPlaybackRate(parseFloat(speedSelect.value));
}

function updateVolumeLabel() {
  volumeValue.textContent = `${Math.round(parseFloat(volumeBar.value) * 100)}%`;
}

function updateNoteColorOptionState() {
  noteColorOption.classList.toggle("option-disabled", togglePitchColor.checked);
}

// ---------------------------------------------------------------------------
// Player <-> controls wiring
// ---------------------------------------------------------------------------
function handleTick(currentTime, duration) {
  if (isSeeking) return;
  timeCurrent.textContent = formatTime(currentTime);
  if (duration > 0) {
    seekBar.max = String(duration);
    seekBar.value = String(currentTime);
    const pct = (currentTime / duration) * 100;
    seekBar.style.setProperty("--seek-pct", `${pct}%`);
  }
}

function handlePlayStateChange(isPlaying) {
  iconPlay.hidden = isPlaying;
  iconPause.hidden = !isPlaying;
  vizEmptyHint.classList.toggle("hidden", isPlaying || player.currentTime > 0);
}

function handleDurationChange(duration) {
  timeTotal.textContent = formatTime(duration);
  seekBar.max = String(duration || 0);
}

btnPlay.addEventListener("click", () => {
  vizEmptyHint.classList.add("hidden");
  player?.togglePlay();
});

btnRestart.addEventListener("click", () => {
  player?.restart();
  vizEmptyHint.classList.add("hidden");
});

seekBar.addEventListener("pointerdown", () => {
  isSeeking = true;
});
seekBar.addEventListener("input", () => {
  const value = parseFloat(seekBar.value);
  timeCurrent.textContent = formatTime(value);
  const duration = parseFloat(seekBar.max) || 1;
  seekBar.style.setProperty("--seek-pct", `${(value / duration) * 100}%`);
  player?.seekTo(value);
});
seekBar.addEventListener("pointerup", () => {
  isSeeking = false;
});

volumeBar.addEventListener("input", () => {
  player?.setVolume(parseFloat(volumeBar.value));
  updateVolumeLabel();
});

speedSelect.addEventListener("change", () => {
  player?.setPlaybackRate(parseFloat(speedSelect.value));
});

lookaheadSlider.addEventListener("input", () => {
  const value = parseFloat(lookaheadSlider.value);
  lookaheadValue.textContent = `${value.toFixed(1)}s`;
  visualizer?.setOptions({ lookaheadSeconds: value });
});

toggleLabels.addEventListener("change", () => {
  visualizer?.setOptions({ showLabels: toggleLabels.checked });
});
toggleEffects.addEventListener("change", () => {
  visualizer?.setOptions({ effectsEnabled: toggleEffects.checked });
});
togglePitchColor.addEventListener("change", () => {
  visualizer?.setOptions({ colorByPitch: togglePitchColor.checked });
  updateNoteColorOptionState();
});

noteColorPicker.addEventListener("input", () => {
  visualizer?.setCustomColor(noteColorPicker.value);
});

document.addEventListener("keydown", (event) => {
  if (!screens.visualizer.classList.contains("screen-active")) return;
  if (event.target.tagName === "INPUT" || event.target.tagName === "SELECT") return;
  if (event.code === "Space") {
    event.preventDefault();
    player?.togglePlay();
  }
});

// ---------------------------------------------------------------------------
// Fullscreen
// ---------------------------------------------------------------------------
btnFullscreen.addEventListener("click", () => {
  if (document.fullscreenElement) {
    document.exitFullscreen();
  } else {
    screens.visualizer.requestFullscreen?.();
  }
});

// ---------------------------------------------------------------------------
// Download dropdown
// ---------------------------------------------------------------------------
downloadBtn.addEventListener("click", (event) => {
  event.stopPropagation();
  downloadMenu.classList.toggle("open");
});
document.addEventListener("click", () => downloadMenu.classList.remove("open"));

// ---------------------------------------------------------------------------
// New file / retry
// ---------------------------------------------------------------------------
function resetToLanding() {
  player?.pause();
  fileInput.value = "";
  resetRecordPanel();
  showScreen("landing");
}
newFileBtn.addEventListener("click", resetToLanding);
errorRetryBtn.addEventListener("click", resetToLanding);

// ---------------------------------------------------------------------------
// Upload interactions
// ---------------------------------------------------------------------------
const browseBtn = document.getElementById("browse-btn");

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  handleFileSelected(file);
});

browseBtn.addEventListener("click", (event) => {
  event.stopPropagation();
  fileInput.click();
});

dropzone.addEventListener("click", (event) => {
  if (event.target === browseBtn) return;
  fileInput.click();
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
  });
});
dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer?.files?.[0];
  handleFileSelected(file);
});

window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => e.preventDefault());

// ---------------------------------------------------------------------------
// Upload / Record mode toggle
// ---------------------------------------------------------------------------
function setMode(mode) {
  const isUpload = mode === "upload";
  modeBtnUpload.classList.toggle("mode-btn-active", isUpload);
  modeBtnRecord.classList.toggle("mode-btn-active", !isUpload);
  uploadPanel.classList.toggle("mode-panel-active", isUpload);
  recordPanel.classList.toggle("mode-panel-active", !isUpload);

  if (isUpload) {
    recorder?.cancel();
  }
}

modeBtnUpload.addEventListener("click", () => setMode("upload"));
modeBtnRecord.addEventListener("click", () => setMode("record"));

if (!isRecordingSupported()) {
  modeBtnRecord.disabled = true;
  modeBtnRecord.title = "Microphone recording isn't supported in this browser.";
}

// ---------------------------------------------------------------------------
// Live recording - captured audio feeds into the exact same
// handleFileSelected() upload/transcribe flow used for file uploads.
// ---------------------------------------------------------------------------
function showRecordBlock(name) {
  recordIdleBlock.hidden = name !== "idle";
  recordActiveBlock.hidden = name !== "active";
  recordPreviewBlock.hidden = name !== "preview";
}

function resetRecordPanel() {
  recorder?.cancel();
  recorder = null;
  recordedBlob = null;
  recordedExtension = null;
  if (recordedObjectUrl) {
    URL.revokeObjectURL(recordedObjectUrl);
    recordedObjectUrl = null;
  }
  recordPreviewAudio.removeAttribute("src");
  recordError.hidden = true;
  showRecordBlock("idle");
}

recordStartBtn.addEventListener("click", async () => {
  recordError.hidden = true;
  const nextRecorder = new Recorder({
    onTick: (elapsed) => {
      recordTimer.textContent = formatTime(elapsed);
    },
  });

  try {
    await nextRecorder.start();
    recorder = nextRecorder;
    recordTimer.textContent = "0:00";
    showRecordBlock("active");
  } catch (err) {
    let message = err?.message || "Could not start recording.";
    if (err?.name === "NotAllowedError") {
      message = "Microphone access was denied. Allow microphone access in your browser and try again.";
    } else if (err?.name === "NotFoundError") {
      message = "No microphone was found on this device.";
    }
    recordError.textContent = message;
    recordError.hidden = false;
  }
});

recordStopBtn.addEventListener("click", async () => {
  if (!recorder) return;
  try {
    const result = await recorder.stop();
    recordedBlob = result.blob;
    recordedExtension = result.extension;
    if (recordedObjectUrl) URL.revokeObjectURL(recordedObjectUrl);
    recordedObjectUrl = URL.createObjectURL(recordedBlob);
    recordPreviewAudio.src = recordedObjectUrl;
    recordDuration.textContent = formatTime(result.durationSec);
    showRecordBlock("preview");
  } catch (err) {
    recordError.textContent = err.message || "Recording failed.";
    recordError.hidden = false;
    showRecordBlock("idle");
  } finally {
    recorder = null;
  }
});

recordUseBtn.addEventListener("click", () => {
  if (!recordedBlob) return;
  const file = new File(
    [recordedBlob],
    `live-recording-${Date.now()}.${recordedExtension}`,
    { type: recordedBlob.type }
  );
  handleFileSelected(file);
});

recordRerecordBtn.addEventListener("click", () => {
  resetRecordPanel();
});

updateVolumeLabel();
updateNoteColorOptionState();
