// 88-key piano geometry: MIDI 21 (A0) through MIDI 108 (C8).
// Produces pixel positions so black keys line up correctly against white
// keys regardless of canvas width.

export const MIN_MIDI = 21;
export const MAX_MIDI = 108;

const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

// Fractional "diatonic" offset of each pitch class from its octave's C,
// measured in units of one white-key width. Black keys sit between their
// neighboring white keys using proportions modeled on real keyboards.
const DIATONIC_OFFSET = [
  0,      // C
  0.596,  // C#
  1,      // D
  1.617,  // D#
  2,      // E
  3,      // F
  3.404,  // F#
  4,      // G
  4.362,  // G#
  5,      // A
  5.319,  // A#
  6,      // B
];

const BLACK_PITCH_CLASSES = new Set([1, 3, 6, 8, 10]);
const BLACK_KEY_WIDTH_RATIO = 0.583;
const BLACK_KEY_HEIGHT_RATIO = 0.62;

function diatonicPosition(midi) {
  const pitchClass = ((midi % 12) + 12) % 12;
  const octaveIndex = Math.floor(midi / 12);
  return octaveIndex * 7 + DIATONIC_OFFSET[pitchClass];
}

export function noteName(midi) {
  const pitchClass = ((midi % 12) + 12) % 12;
  const octave = Math.floor(midi / 12) - 1;
  return `${NOTE_NAMES[pitchClass]}${octave}`;
}

export function isBlackKey(midi) {
  return BLACK_PITCH_CLASSES.has(((midi % 12) + 12) % 12);
}

export function buildKeyRange() {
  const keys = [];
  for (let midi = MIN_MIDI; midi <= MAX_MIDI; midi++) {
    keys.push({ midi, isBlack: isBlackKey(midi), name: noteName(midi) });
  }
  return keys;
}

const MIN_DIATONIC = diatonicPosition(MIN_MIDI);
const MAX_DIATONIC = diatonicPosition(MAX_MIDI);
export const TOTAL_WHITE_KEYS = Math.round(MAX_DIATONIC - MIN_DIATONIC) + 1;

/**
 * Compute pixel geometry for the full keyboard given an available width.
 * Returns { whiteKeyWidth, blackKeyWidth, keyboardHeight, blackKeyHeight,
 *           positions: Map<midi, {x, width, isBlack}> }
 */
export function computeKeyboardLayout(canvasWidth, keyboardHeight) {
  const whiteKeyWidth = canvasWidth / TOTAL_WHITE_KEYS;
  const blackKeyWidth = whiteKeyWidth * BLACK_KEY_WIDTH_RATIO;
  const blackKeyHeight = keyboardHeight * BLACK_KEY_HEIGHT_RATIO;

  const positions = new Map();
  for (let midi = MIN_MIDI; midi <= MAX_MIDI; midi++) {
    const centerUnits = diatonicPosition(midi) - MIN_DIATONIC;
    const black = isBlackKey(midi);
    const x = black
      ? centerUnits * whiteKeyWidth - blackKeyWidth / 2
      : centerUnits * whiteKeyWidth;
    const width = black ? blackKeyWidth : whiteKeyWidth;
    positions.set(midi, { x, width, isBlack: black, centerX: x + width / 2 });
  }

  return { whiteKeyWidth, blackKeyWidth, keyboardHeight, blackKeyHeight, positions };
}
