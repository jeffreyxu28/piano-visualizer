// Faithful port of backend/note_decoder.py's decode_notes()/make_note() -
// converts onset/frame probability curves into discrete note events. Kept
// as a direct line-for-line translation (same variable names, same
// control flow) rather than a "cleaner" JS rewrite, specifically so it's
// easy to eyeball against the Python original and keep the two in sync.

const HOP_LENGTH = 512;
const SAMPLE_RATE = 16000;
const MIN_MIDI = 21;
const N_KEYS = 88;

const ONSET_THRESHOLD = 0.5;
const FRAME_THRESHOLD = 0.5;
const MIN_START_FRAME_PROB = 0.3;
const FRAME_ONLY_START_THRESHOLD = 0.7;
const OFF_FRAMES_REQUIRED = 2;

function makeNote(pitch, startFrame, endFrame, onsetConfidence, noteNameFn) {
  const secondsPerFrame = HOP_LENGTH / SAMPLE_RATE;
  const startSec = startFrame * secondsPerFrame;
  const endSec = Math.max(startSec + secondsPerFrame, endFrame * secondsPerFrame);

  return {
    midi_note: pitch,
    note_name: noteNameFn(pitch),
    start_sec: round4(startSec),
    end_sec: round4(endSec),
    duration_sec: round4(endSec - startSec),
    onset_confidence: round4(onsetConfidence),
  };
}

function round4(x) {
  return Math.round(x * 10000) / 10000;
}

/**
 * @param onsetProbability Float32Array[numFrames * N_KEYS], row-major (frame-major)
 * @param frameProbability Float32Array[numFrames * N_KEYS], same layout
 * @param numFrames
 * @param noteNameFn (midi:number) => string - e.g. noteName from piano-layout.js
 */
export function decodeNotes(onsetProbability, frameProbability, numFrames, noteNameFn) {
  const notes = [];

  // onset_events: a rising-edge detector over the onset-probability
  // threshold crossing (matches the numpy diff-based version in Python).
  const at = (arr, t, key) => arr[t * N_KEYS + key];
  const onsetBinaryAt = (t, key) => at(onsetProbability, t, key) >= ONSET_THRESHOLD;

  for (let keyIndex = 0; keyIndex < N_KEYS; keyIndex++) {
    const pitch = MIN_MIDI + keyIndex;
    let active = false;
    let startFrame = 0;
    let startConfidence = 0.0;
    let offCount = 0;
    let previousFrameActive = false;

    for (let t = 0; t < numFrames; t++) {
      const onsetNow = onsetBinaryAt(t, keyIndex) && !(t > 0 && onsetBinaryAt(t - 1, keyIndex));
      const frameProb = at(frameProbability, t, keyIndex);
      const frameNow = frameProb >= FRAME_THRESHOLD;

      const frameStarted = frameNow && !previousFrameActive;
      const shouldStart =
        (onsetNow && frameProb >= MIN_START_FRAME_PROB) ||
        (frameStarted && frameProb >= FRAME_ONLY_START_THRESHOLD);

      if (!active) {
        if (shouldStart) {
          active = true;
          startFrame = t;
          startConfidence = at(onsetProbability, t, keyIndex);
          offCount = 0;
        }
        previousFrameActive = frameNow;
        continue;
      }

      if (onsetNow && t > startFrame + 1) {
        notes.push(makeNote(pitch, startFrame, t, startConfidence, noteNameFn));
        startFrame = t;
        startConfidence = at(onsetProbability, t, keyIndex);
        offCount = 0;
        previousFrameActive = frameNow;
        continue;
      }

      if (frameNow) {
        offCount = 0;
      } else {
        offCount += 1;
        if (offCount >= OFF_FRAMES_REQUIRED) {
          const endFrame = Math.max(startFrame + 1, t - OFF_FRAMES_REQUIRED + 1);
          notes.push(makeNote(pitch, startFrame, endFrame, startConfidence, noteNameFn));
          active = false;
          offCount = 0;
        }
      }

      previousFrameActive = frameNow;
    }

    if (active) {
      notes.push(
        makeNote(
          pitch,
          startFrame,
          Math.max(startFrame + 1, numFrames - 1),
          startConfidence,
          noteNameFn
        )
      );
    }
  }

  notes.sort((a, b) => a.start_sec - b.start_sec);
  return notes;
}
