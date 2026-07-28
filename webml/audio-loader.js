// Decodes an uploaded audio file (any format the browser's decoder
// supports - WAV/MP3/OGG/M4A/FLAC/WEBM) into a mono Float32Array at the
// model's target sample rate, entirely client-side via the Web Audio API.
// No network round-trip and no dependency on backend/audio.py's
// librosa-based resampling - the browser's own decoder + OfflineAudioContext
// resampling handles both format decoding and sample-rate conversion.

const TARGET_SAMPLE_RATE = 16000;

export async function loadAudioMono16k(file) {
  const arrayBuffer = await file.arrayBuffer();

  // Decode at whatever native sample rate the file has first.
  const probeCtx = new (window.AudioContext || window.webkitAudioContext)();
  let decoded;
  try {
    decoded = await probeCtx.decodeAudioData(arrayBuffer.slice(0));
  } finally {
    probeCtx.close();
  }

  const durationAtTarget = decoded.duration;
  const targetLength = Math.ceil(durationAtTarget * TARGET_SAMPLE_RATE);

  const offlineCtx = new OfflineAudioContext(1, targetLength, TARGET_SAMPLE_RATE);
  const source = offlineCtx.createBufferSource();

  // Downmix to mono ourselves (equal-weight average across channels,
  // matching librosa/soundfile's mono conversion) - OfflineAudioContext's
  // built-in channel mixing uses different weights for >2 channels.
  const monoBuffer = downmixToMono(decoded, offlineCtx);
  source.buffer = monoBuffer;
  source.connect(offlineCtx.destination);
  source.start(0);

  const rendered = await offlineCtx.startRendering();
  return rendered.getChannelData(0).slice(); // copy out of the AudioBuffer's internal storage
}

function downmixToMono(audioBuffer, ctx) {
  if (audioBuffer.numberOfChannels === 1) return audioBuffer;

  const mono = ctx.createBuffer(1, audioBuffer.length, audioBuffer.sampleRate);
  const out = mono.getChannelData(0);
  const channels = audioBuffer.numberOfChannels;

  for (let ch = 0; ch < channels; ch++) {
    const data = audioBuffer.getChannelData(ch);
    for (let i = 0; i < data.length; i++) {
      out[i] += data[i] / channels;
    }
  }
  return mono;
}
