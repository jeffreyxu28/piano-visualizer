// Wraps getUserMedia + MediaRecorder to capture live piano audio through
// the microphone. The resulting Blob is fed into the exact same upload ->
// transcribe pipeline used for uploaded files - recording is just another
// way to produce a File.

const PREFERRED_MIME_TYPES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/ogg;codecs=opus",
  "audio/ogg",
  "audio/mp4",
];

function pickSupportedMimeType() {
  if (typeof MediaRecorder === "undefined" || !MediaRecorder.isTypeSupported) return "";
  for (const type of PREFERRED_MIME_TYPES) {
    if (MediaRecorder.isTypeSupported(type)) return type;
  }
  return "";
}

const EXTENSION_BY_MIME = {
  "audio/webm": "webm",
  "audio/ogg": "ogg",
  "audio/mp4": "m4a",
};

function extensionForMimeType(mimeType) {
  const base = (mimeType || "").split(";")[0].trim();
  return EXTENSION_BY_MIME[base] || "webm";
}

export function isRecordingSupported() {
  return (
    typeof navigator !== "undefined" &&
    !!navigator.mediaDevices?.getUserMedia &&
    typeof MediaRecorder !== "undefined"
  );
}

export class Recorder {
  constructor(callbacks = {}) {
    this.callbacks = callbacks;
    this.stream = null;
    this.mediaRecorder = null;
    this.chunks = [];
    this.startTime = 0;
    this._tickInterval = null;
  }

  async start() {
    if (!isRecordingSupported()) {
      throw new Error("This browser doesn't support microphone recording.");
    }

    // Voice-call processing (echo cancellation, noise suppression, auto
    // gain) is tuned for speech and audibly distorts piano recordings, so
    // it's explicitly turned off here.
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });

    const preferredType = pickSupportedMimeType();
    this.chunks = [];
    this.mediaRecorder = preferredType
      ? new MediaRecorder(this.stream, { mimeType: preferredType })
      : new MediaRecorder(this.stream);

    this.mimeType = this.mediaRecorder.mimeType || preferredType || "audio/webm";
    this.extension = extensionForMimeType(this.mimeType);

    this.mediaRecorder.addEventListener("dataavailable", (event) => {
      if (event.data && event.data.size > 0) this.chunks.push(event.data);
    });

    this.mediaRecorder.start();
    this.startTime = performance.now();
    this._tickInterval = window.setInterval(() => {
      this.callbacks.onTick?.((performance.now() - this.startTime) / 1000);
    }, 200);
  }

  stop() {
    return new Promise((resolve, reject) => {
      if (!this.mediaRecorder || this.mediaRecorder.state === "inactive") {
        reject(new Error("Not currently recording."));
        return;
      }

      this.mediaRecorder.addEventListener(
        "stop",
        () => {
          if (this._tickInterval !== null) window.clearInterval(this._tickInterval);
          this._tickInterval = null;
          this.stream?.getTracks().forEach((track) => track.stop());

          const durationSec = (performance.now() - this.startTime) / 1000;
          const blob = new Blob(this.chunks, { type: this.mimeType });

          if (blob.size === 0) {
            reject(new Error("No audio was captured. Check your microphone and try again."));
            return;
          }

          resolve({ blob, durationSec, extension: this.extension, mimeType: this.mimeType });
        },
        { once: true }
      );

      this.mediaRecorder.stop();
    });
  }

  // Abandon an in-progress recording (e.g. the user navigates away)
  // without resolving a result. Safe to call at any time.
  cancel() {
    if (this._tickInterval !== null) window.clearInterval(this._tickInterval);
    this._tickInterval = null;
    try {
      if (this.mediaRecorder && this.mediaRecorder.state !== "inactive") {
        this.mediaRecorder.stop();
      }
    } catch {
      /* already stopped */
    }
    this.stream?.getTracks().forEach((track) => track.stop());
  }
}
