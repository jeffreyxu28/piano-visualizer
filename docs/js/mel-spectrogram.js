// In-browser log-mel spectrogram, built to numerically match
// backend/audio.py's waveform_to_log_mel() (librosa.feature.melspectrogram
// with n_fft=2048, hop_length=512, n_mels=229, fmin=20, fmax=8000,
// power=2.0, center=True, pad_mode="constant" [zero-padding], window="hann")
// followed by librosa.power_to_db(ref=max, top_db=80) and the same
// (db + 80) / 80 normalization used at training/inference time.
//
// The mel filterbank itself (which depends on librosa's Slaney-normalized
// filter design, not just arithmetic) is NOT reimplemented here - it's
// precomputed once by webml/export_mel_filterbank.py and loaded as a flat
// binary asset. This file only needs to get the STFT bit-for-bit
// equivalent; the filterbank matrix multiply after that is exact by
// construction.

const N_FFT = 2048;
const HOP_LENGTH = 512;
const N_MELS = 229;

let cachedFilterbank = null; // Float32Array[N_MELS * (N_FFT/2 + 1)], loaded once

export async function loadMelFilterbank(url) {
  if (cachedFilterbank) return cachedFilterbank;
  const buf = await fetch(url).then((r) => r.arrayBuffer());
  cachedFilterbank = new Float32Array(buf);
  const expected = N_MELS * (N_FFT / 2 + 1);
  if (cachedFilterbank.length !== expected) {
    throw new Error(
      `mel filterbank size mismatch: got ${cachedFilterbank.length}, expected ${expected}`
    );
  }
  return cachedFilterbank;
}

// Precomputed once: index-bit-reversal table and twiddle factors for a
// fixed-size (N_FFT) radix-2 Cooley-Tukey FFT - n_fft is a compile-time
// constant here (2048, a power of two) so there's no need for a general
// arbitrary-length FFT.
const FFT_BITS = Math.log2(N_FFT);
if (!Number.isInteger(FFT_BITS)) {
  throw new Error("N_FFT must be a power of two for this FFT implementation.");
}

function buildBitReverseTable(n, bits) {
  const table = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    let reversed = 0;
    let x = i;
    for (let b = 0; b < bits; b++) {
      reversed = (reversed << 1) | (x & 1);
      x >>= 1;
    }
    table[i] = reversed;
  }
  return table;
}

const BIT_REVERSE = buildBitReverseTable(N_FFT, FFT_BITS);

// cosTable[k]/sinTable[k] for k in [0, N_FFT/2) - the twiddle factors
// e^{-2*pi*i*k/N_FFT}, reused across every FFT call.
const HALF_N = N_FFT / 2;
const COS_TABLE = new Float64Array(HALF_N);
const SIN_TABLE = new Float64Array(HALF_N);
for (let k = 0; k < HALF_N; k++) {
  const angle = (-2 * Math.PI * k) / N_FFT;
  COS_TABLE[k] = Math.cos(angle);
  SIN_TABLE[k] = Math.sin(angle);
}

const HANN_WINDOW = new Float64Array(N_FFT);
for (let i = 0; i < N_FFT; i++) {
  // librosa/scipy's periodic Hann window (matches "hann" in librosa.stft,
  // which uses scipy.signal.get_window with fftbins=True by default).
  HANN_WINDOW[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N_FFT);
}

/**
 * In-place iterative radix-2 Cooley-Tukey FFT.
 * real/imag are Float64Array of length N_FFT; imag should be zero-filled
 * for a real-valued input frame.
 */
function fftInPlace(real, imag) {
  const n = N_FFT;

  for (let i = 0; i < n; i++) {
    const j = BIT_REVERSE[i];
    if (j > i) {
      let tmp = real[i]; real[i] = real[j]; real[j] = tmp;
      tmp = imag[i]; imag[i] = imag[j]; imag[j] = tmp;
    }
  }

  for (let size = 2; size <= n; size *= 2) {
    const halfSize = size / 2;
    const tableStride = n / size;
    for (let start = 0; start < n; start += size) {
      for (let k = 0; k < halfSize; k++) {
        const twIdx = k * tableStride;
        const tCos = COS_TABLE[twIdx];
        const tSin = SIN_TABLE[twIdx];

        const evenIdx = start + k;
        const oddIdx = start + k + halfSize;

        const oddReal = real[oddIdx];
        const oddImag = imag[oddIdx];
        const twReal = oddReal * tCos - oddImag * tSin;
        const twImag = oddReal * tSin + oddImag * tCos;

        real[oddIdx] = real[evenIdx] - twReal;
        imag[oddIdx] = imag[evenIdx] - twImag;
        real[evenIdx] = real[evenIdx] + twReal;
        imag[evenIdx] = imag[evenIdx] + twImag;
      }
    }
  }
}

/**
 * Compute the log-mel spectrogram for a mono float32 waveform already at
 * the model's target sample rate (16kHz) - matches waveform_to_log_mel()
 * in backend/audio.py frame-for-frame.
 *
 * Returns Float32Array[numFrames * N_MELS], row-major (frame-major).
 */
export function computeLogMelSpectrogram(waveform, filterbank) {
  const n = waveform.length;
  // librosa's center=True frame count formula (n_fft even): 1 + n // hop
  const numFrames = 1 + Math.floor(n / HOP_LENGTH);
  const numFreqBins = N_FFT / 2 + 1;
  const padAmount = N_FFT >> 1;

  // Zero-padded copy of the signal (pad_mode="constant"), padAmount zeros
  // on each side, matching librosa's center=True + pad_mode="constant".
  const padded = new Float64Array(n + 2 * padAmount);
  for (let i = 0; i < n; i++) padded[padAmount + i] = waveform[i];

  const real = new Float64Array(N_FFT);
  const imag = new Float64Array(N_FFT);
  const melPower = new Float32Array(numFrames * N_MELS);
  const powerFrame = new Float64Array(numFreqBins);

  let globalMaxDb = -Infinity;
  const dbFrames = new Float32Array(numFrames * N_MELS);

  for (let frame = 0; frame < numFrames; frame++) {
    const frameStart = frame * HOP_LENGTH;

    for (let i = 0; i < N_FFT; i++) {
      const sampleIdx = frameStart + i;
      const sample = sampleIdx < padded.length ? padded[sampleIdx] : 0;
      real[i] = sample * HANN_WINDOW[i];
      imag[i] = 0;
    }

    fftInPlace(real, imag);

    for (let bin = 0; bin < numFreqBins; bin++) {
      const re = real[bin];
      const im = imag[bin];
      powerFrame[bin] = re * re + im * im; // power=2.0
    }

    // mel = filterbank (N_MELS x numFreqBins) @ powerFrame
    for (let m = 0; m < N_MELS; m++) {
      let acc = 0;
      const rowOffset = m * numFreqBins;
      for (let bin = 0; bin < numFreqBins; bin++) {
        acc += filterbank[rowOffset + bin] * powerFrame[bin];
      }
      melPower[frame * N_MELS + m] = acc;
    }
  }

  // librosa.power_to_db(ref=np.max, top_db=80):
  //   db = 10*log10(max(x, amin)) - 10*log10(max(ref_power, amin))
  //   db = max(db, db.max() - top_db)
  // Since log10 is monotonic, max(10*log10(x)) == 10*log10(max(x)), so the
  // max of the per-element dB values computed below already equals the
  // reference (max-power) dB value - no need to compute it separately.
  const AMIN = 1e-10;
  for (let i = 0; i < melPower.length; i++) {
    const db = 10 * Math.log10(Math.max(AMIN, melPower[i]));
    dbFrames[i] = db;
    if (db > globalMaxDb) globalMaxDb = db;
  }

  const TOP_DB = 80;
  const out = new Float32Array(dbFrames.length);
  for (let i = 0; i < dbFrames.length; i++) {
    const relativeDb = Math.max(dbFrames[i] - globalMaxDb, -TOP_DB);
    out[i] = (relativeDb + TOP_DB) / TOP_DB; // matches (db + 80) / 80 in backend/audio.py
  }

  return { data: out, numFrames, numMels: N_MELS };
}
