// Minimal Standard MIDI File (SMF format 0) writer - just enough to turn
// our decoded notes into a downloadable .mid, matching backend/note_decoder.py's
// save_midi() (same velocity mapping from onset_confidence, program 0/piano).

const TICKS_PER_QUARTER = 480;
const MICROSECONDS_PER_QUARTER = 500000; // 120 BPM
const TICKS_PER_SECOND = TICKS_PER_QUARTER / (MICROSECONDS_PER_QUARTER / 1e6);

function secondsToTicks(seconds) {
  return Math.round(seconds * TICKS_PER_SECOND);
}

function encodeVariableLength(value) {
  const bytes = [value & 0x7f];
  value >>= 7;
  while (value > 0) {
    bytes.unshift((value & 0x7f) | 0x80);
    value >>= 7;
  }
  return bytes;
}

function velocityFromConfidence(onsetConfidence) {
  return Math.max(1, Math.min(127, Math.round(80 * onsetConfidence + 20)));
}

/**
 * @param notes array of {midi_note, start_sec, end_sec, onset_confidence}
 * @returns Uint8Array - a complete .mid file
 */
export function notesToMidi(notes) {
  const events = [];
  for (const note of notes) {
    const velocity = velocityFromConfidence(note.onset_confidence);
    events.push({ tick: secondsToTicks(note.start_sec), type: "on", pitch: note.midi_note, velocity });
    events.push({ tick: secondsToTicks(note.end_sec), type: "off", pitch: note.midi_note, velocity: 0 });
  }
  // Stable sort by tick; note-offs before note-ons at the same tick avoids
  // a spurious zero-length gap when one note ends exactly as another starts.
  events.sort((a, b) => a.tick - b.tick || (a.type === "off" ? -1 : 1));

  const trackBytes = [];
  let lastTick = 0;

  // Tempo meta event up front (FF 51 03 <3-byte tempo>).
  trackBytes.push(...encodeVariableLength(0), 0xff, 0x51, 0x03,
    (MICROSECONDS_PER_QUARTER >> 16) & 0xff,
    (MICROSECONDS_PER_QUARTER >> 8) & 0xff,
    MICROSECONDS_PER_QUARTER & 0xff);
  // Program change to Acoustic Grand Piano (program 0), channel 0.
  trackBytes.push(...encodeVariableLength(0), 0xc0, 0x00);

  for (const event of events) {
    const delta = Math.max(0, event.tick - lastTick);
    lastTick = event.tick;
    const status = event.type === "on" ? 0x90 : 0x80;
    trackBytes.push(...encodeVariableLength(delta), status, event.pitch & 0x7f, event.velocity & 0x7f);
  }

  // End of track meta event.
  trackBytes.push(...encodeVariableLength(0), 0xff, 0x2f, 0x00);

  const headerChunk = [
    0x4d, 0x54, 0x68, 0x64, // "MThd"
    0x00, 0x00, 0x00, 0x06, // chunk length = 6
    0x00, 0x00, // format 0
    0x00, 0x01, // 1 track
    (TICKS_PER_QUARTER >> 8) & 0xff, TICKS_PER_QUARTER & 0xff,
  ];

  const trackLength = trackBytes.length;
  const trackChunk = [
    0x4d, 0x54, 0x72, 0x6b, // "MTrk"
    (trackLength >> 24) & 0xff,
    (trackLength >> 16) & 0xff,
    (trackLength >> 8) & 0xff,
    trackLength & 0xff,
    ...trackBytes,
  ];

  return new Uint8Array([...headerChunk, ...trackChunk]);
}
