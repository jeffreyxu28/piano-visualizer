// Port of backend/note_decoder.py's predict_probabilities(): sliding
// ~10s-window inference over the whole recording, averaging predictions
// in the overlap regions. Requires an already-created onnxruntime-web
// `ort.InferenceSession` (loaded by the caller) and the mel filterbank
// (see mel-spectrogram.js).

import { computeLogMelSpectrogram } from "./mel-spectrogram.js";

const SAMPLE_RATE = 16000;
const HOP_LENGTH = 512;
const N_KEYS = 88;
const CHUNK_SECONDS = 10.0;
const WINDOW_OVERLAP_SECONDS = 2.0;

/**
 * @param session ort.InferenceSession for the exported PianoTranscriptionModel
 * @param waveform Float32Array, mono, 16kHz
 * @param filterbank Float32Array mel filterbank (see loadMelFilterbank)
 * @param onProgress optional (fraction:number) => void, called after each window
 * @returns { onsetProbability: Float32Array, frameProbability: Float32Array, numFrames: number }
 *          both arrays row-major [numFrames * N_KEYS], averaged over overlaps
 */
export async function predictProbabilities(session, waveform, filterbank, onProgress) {
  const totalFrames = 1 + Math.floor(waveform.length / HOP_LENGTH);
  const windowFrames = Math.floor((CHUNK_SECONDS * SAMPLE_RATE) / HOP_LENGTH) + 1;
  const overlapFrames = Math.max(1, Math.floor((WINDOW_OVERLAP_SECONDS * SAMPLE_RATE) / HOP_LENGTH));
  const stride = Math.max(1, windowFrames - overlapFrames);

  const onsetSum = new Float32Array(totalFrames * N_KEYS);
  const frameSum = new Float32Array(totalFrames * N_KEYS);
  const counts = new Float32Array(totalFrames);

  const totalWindows = Math.max(1, Math.ceil(totalFrames / stride));
  let windowIndex = 0;

  for (let start = 0; start < totalFrames; start += stride) {
    const end = Math.min(totalFrames, start + windowFrames);
    const neededFrames = end - start;

    const startSample = start * HOP_LENGTH;
    const endSample = Math.min(waveform.length, startSample + neededFrames * HOP_LENGTH);
    const chunkWaveform = waveform.subarray(startSample, endSample);
    if (chunkWaveform.length === 0) break;

    const chunkMel = computeLogMelSpectrogram(chunkWaveform, filterbank);
    const usableFrames = Math.min(chunkMel.numFrames, neededFrames);
    if (usableFrames <= 0) break;
    const actualEnd = start + usableFrames;

    const melData =
      usableFrames === chunkMel.numFrames
        ? chunkMel.data
        : chunkMel.data.subarray(0, usableFrames * chunkMel.numMels);

    const inputTensor = new window.ort.Tensor("float32", melData, [1, usableFrames, chunkMel.numMels]);
    const results = await session.run({ mel: inputTensor });
    const onsetLogits = results.onset_logits.data;
    const frameLogits = results.frame_logits.data;

    for (let t = 0; t < usableFrames; t++) {
      const globalFrame = start + t;
      for (let k = 0; k < N_KEYS; k++) {
        const idx = t * N_KEYS + k;
        const globalIdx = globalFrame * N_KEYS + k;
        onsetSum[globalIdx] += sigmoid(onsetLogits[idx]);
        frameSum[globalIdx] += sigmoid(frameLogits[idx]);
      }
      counts[globalFrame] += 1;
    }

    windowIndex += 1;
    onProgress?.(Math.min(1, windowIndex / totalWindows));

    if (end >= totalFrames) break;
  }

  const onsetProbability = new Float32Array(totalFrames * N_KEYS);
  const frameProbability = new Float32Array(totalFrames * N_KEYS);
  for (let f = 0; f < totalFrames; f++) {
    const c = Math.max(1, counts[f]);
    for (let k = 0; k < N_KEYS; k++) {
      const idx = f * N_KEYS + k;
      onsetProbability[idx] = onsetSum[idx] / c;
      frameProbability[idx] = frameSum[idx] / c;
    }
  }

  return { onsetProbability, frameProbability, numFrames: totalFrames };
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}
