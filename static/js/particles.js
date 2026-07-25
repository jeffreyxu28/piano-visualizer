// Deterministic key-impact particle bursts.
//
// Particles are NOT simulated with accumulated delta-time (that would
// desync from the audio clock on pause/seek). Instead every particle's
// position is a pure function of "time since note onset", so scrubbing
// the seek bar always produces the exact correct frame.

const PARTICLES_PER_BURST = 7;
const LIFETIME_SEC = 0.45;
const GRAVITY = 260; // px/s^2

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hashSeed(midi, startSec) {
  return Math.floor(midi * 7349 + startSec * 100003) >>> 0;
}

/**
 * Returns an array of particle draw descriptors for a note's onset burst,
 * or an empty array if `age` (seconds since note.start_sec) is outside the
 * particle lifetime window.
 */
export function getImpactParticles(midi, startSec, keyCenterX, keyTopY, age) {
  if (age < 0 || age > LIFETIME_SEC) return [];

  const rand = mulberry32(hashSeed(midi, startSec));
  const particles = [];
  const lifeFrac = age / LIFETIME_SEC;

  for (let i = 0; i < PARTICLES_PER_BURST; i++) {
    const angle = -Math.PI / 2 + (rand() - 0.5) * Math.PI * 0.9;
    const speed = 60 + rand() * 90;
    const vx = Math.cos(angle) * speed;
    const vy = Math.sin(angle) * speed;

    const x = keyCenterX + vx * age;
    const y = keyTopY + vy * age + 0.5 * GRAVITY * age * age;
    const radius = (1 - lifeFrac) * (2 + rand() * 2.2);
    const alpha = Math.max(0, 1 - lifeFrac);

    particles.push({ x, y, radius, alpha });
  }

  return particles;
}

export function getImpactGlow(age, glowDuration = 0.22) {
  if (age < 0 || age > glowDuration) return 0;
  return 1 - age / glowDuration;
}
