// Deterministic key-impact effects: a "fizzle" spark/ember burst plus a
// fast-expanding shockwave ring and a brief white flash, played when a
// note hits the keyboard.
//
// Everything here is a pure function of "time since note onset" (age), not
// an accumulated simulation - scrubbing the seek bar always reproduces the
// exact same frame, and pausing freezes it in place. Each particle gets
// its own independently-seeded RNG (keyed by note + particle index) so its
// trajectory is fixed regardless of which other particles are alive at a
// given instant - that's what keeps motion smooth frame to frame instead
// of particles "teleporting" as neighbors fade in and out.

const PARTICLES_PER_BURST = 14;
const GRAVITY = 300; // px/s^2

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

function hashSeed(midi, startSec, salt) {
  return Math.floor(midi * 7349 + startSec * 100003 + salt * 104729) >>> 0;
}

/**
 * Returns ember descriptors for a note's onset burst (empty once every
 * ember has fizzled out). Each ember carries a `heat` value - 1 for a
 * white-hot spark fresh off the key, fading toward 0 (the note's own
 * color) as it cools - so the caller can render a flame-like gradient.
 */
export function getImpactParticles(midi, startSec, keyCenterX, keyTopY, age) {
  const particles = [];

  for (let i = 0; i < PARTICLES_PER_BURST; i++) {
    const rand = mulberry32(hashSeed(midi, startSec, i));
    const lifetime = 0.32 + rand() * 0.4;
    const delay = rand() * 0.035;

    // Bail before the trig/remaining-rand() work below - most calls to
    // this function are for particles outside their own lifetime window.
    const localAge = age - delay;
    if (localAge < 0 || localAge > lifetime) continue;

    const angle = -Math.PI / 2 + (rand() - 0.5) * Math.PI * 1.1;
    const speed = 45 + rand() * 125;
    const flickerPhase = rand() * Math.PI * 2;
    const sizeSeed = rand();

    const vx = Math.cos(angle) * speed;
    const vy = Math.sin(angle) * speed;
    const x = keyCenterX + vx * localAge;
    const y = keyTopY + vy * localAge + 0.5 * GRAVITY * localAge * localAge;

    const lifeFrac = localAge / lifetime;
    const flicker = 0.7 + 0.3 * Math.sin(localAge * 45 + flickerPhase);
    const radius = (1 - lifeFrac * 0.8) * (1.3 + sizeSeed * 2.1);
    const alpha = Math.max(0, 1 - lifeFrac * lifeFrac) * flicker;
    const heat = Math.max(0, 1 - lifeFrac * 1.7);

    particles.push({ x, y, radius, alpha, heat });
  }

  return particles;
}

const SHOCKWAVE_DURATION = 0.3;

/** A thin ring that snaps outward from the key at the moment of impact. */
export function getImpactShockwave(age) {
  if (age < 0 || age > SHOCKWAVE_DURATION) return null;
  const t = age / SHOCKWAVE_DURATION;
  const eased = 1 - (1 - t) * (1 - t);
  return {
    radius: 4 + eased * 44,
    alpha: (1 - t) * (1 - t) * 0.75,
    lineWidth: 2.5 * (1 - t) + 0.5,
  };
}

const FLASH_DURATION = 0.1;

/** A near-instant white-hot flash right as the note lands. */
export function getImpactFlash(age) {
  if (age < 0 || age > FLASH_DURATION) return 0;
  return 1 - age / FLASH_DURATION;
}

export function getImpactGlow(age, glowDuration = 0.26) {
  if (age < 0 || age > glowDuration) return 0;
  const t = age / glowDuration;
  return (1 - t) * (1 - t);
}
