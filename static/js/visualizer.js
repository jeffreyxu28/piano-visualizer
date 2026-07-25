// Canvas falling-note visualizer.
//
// The single source of truth for "where is playback right now" is the
// `currentTime` value passed into render() every frame - it always comes
// from the <audio> element's currentTime (see player.js). Nothing here
// keeps its own clock, so pause/resume/seek/slowdown are all correct for
// free: the same (notes, currentTime) pair always produces the same frame.

import { MIN_MIDI, MAX_MIDI, computeKeyboardLayout, noteName } from "./piano-layout.js";
import { getImpactParticles, getImpactGlow } from "./particles.js";

const TOP_MARGIN = 18;
const MIN_KEYBOARD_HEIGHT = 108;
const MAX_KEYBOARD_HEIGHT = 168;
const PAST_NOTE_BUFFER = 0.08; // keep drawing briefly after a note ends for a soft fade

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function mapRange(value, inMin, inMax, outMin, outMax) {
  const t = clamp((value - inMin) / (inMax - inMin), 0, 1);
  return outMin + t * (outMax - outMin);
}

function pitchHue(midi) {
  return mapRange(midi, MIN_MIDI, MAX_MIDI, 22, 268);
}

// #rrggbb -> hue in degrees [0,360). Falls back to the default accent hue
// (232, indigo-blue) if the string can't be parsed.
function hexToHue(hex) {
  const match = /^#?([0-9a-f]{6})$/i.exec(hex || "");
  if (!match) return 232;

  const int = parseInt(match[1], 16);
  const r = ((int >> 16) & 255) / 255;
  const g = ((int >> 8) & 255) / 255;
  const b = (int & 255) / 255;

  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const delta = max - min;
  if (delta === 0) return 232;

  let hue;
  if (max === r) hue = ((g - b) / delta) % 6;
  else if (max === g) hue = (b - r) / delta + 2;
  else hue = (r - g) / delta + 4;

  hue *= 60;
  return hue < 0 ? hue + 360 : hue;
}

export class Visualizer {
  constructor(canvas, stageEl) {
    this.canvas = canvas;
    this.stageEl = stageEl;
    this.ctx = canvas.getContext("2d");

    this.notes = [];
    this.notesByKey = new Map(); // midi -> notes[] for that key, sorted by start

    this.options = {
      lookaheadSeconds: 4,
      showLabels: false,
      effectsEnabled: true,
      colorByPitch: true,
      customColorHue: 232, // used instead of pitch coloring when colorByPitch is off
    };

    this.width = 0;
    this.height = 0;
    this.dpr = Math.max(1, window.devicePixelRatio || 1);
    this.layout = null;

    if (typeof ResizeObserver !== "undefined") {
      this._resizeObserver = new ResizeObserver(() => this._handleResize());
      this._resizeObserver.observe(this.stageEl);
    } else {
      window.addEventListener("resize", () => this._handleResize());
    }
    this._handleResize();
  }

  setNotes(notes) {
    this.notes = notes;
    this.notesByKey = new Map();
    for (const note of notes) {
      if (!this.notesByKey.has(note.midi_note)) this.notesByKey.set(note.midi_note, []);
      this.notesByKey.get(note.midi_note).push(note);
    }
  }

  setOptions(partial) {
    Object.assign(this.options, partial);
  }

  setCustomColor(hexString) {
    this.options.customColorHue = hexToHue(hexString);
  }

  _handleResize() {
    const rect = this.stageEl.getBoundingClientRect();
    this.width = Math.max(1, Math.floor(rect.width));
    this.height = Math.max(1, Math.floor(rect.height));
    this.dpr = Math.max(1, window.devicePixelRatio || 1);

    this.canvas.width = Math.floor(this.width * this.dpr);
    this.canvas.height = Math.floor(this.height * this.dpr);
    this.canvas.style.width = `${this.width}px`;
    this.canvas.style.height = `${this.height}px`;
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);

    const keyboardHeight = clamp(this.height * 0.2, MIN_KEYBOARD_HEIGHT, MAX_KEYBOARD_HEIGHT);
    this.keyboardTopY = this.height - keyboardHeight;
    this.fallAreaHeight = this.keyboardTopY - TOP_MARGIN;
    this.layout = computeKeyboardLayout(this.width, keyboardHeight);
  }

  render(currentTime) {
    const { ctx, width, height } = this;
    if (!this.layout) return;

    this._drawBackground(ctx, width, height);

    const lookahead = this.options.lookaheadSeconds;
    const pixelsPerSecond = this.fallAreaHeight / lookahead;

    const visibleNotes = [];
    const activeMidiSet = new Set();
    for (const note of this.notes) {
      const timeUntilStart = note.start_sec - currentTime;
      const timeSinceEnd = currentTime - note.end_sec;
      if (timeUntilStart > lookahead) continue;
      if (timeSinceEnd > PAST_NOTE_BUFFER) continue;
      visibleNotes.push(note);
      if (note.start_sec <= currentTime && currentTime < note.end_sec) {
        activeMidiSet.add(note.midi_note);
      }
    }

    this._drawFallingNotes(ctx, visibleNotes, currentTime, pixelsPerSecond);
    this._drawKeyboard(ctx, activeMidiSet, currentTime);

    if (this.options.effectsEnabled) {
      this._drawImpactEffects(ctx, visibleNotes, currentTime);
    }
  }

  _drawBackground(ctx, width, height) {
    ctx.clearRect(0, 0, width, height);

    const bg = ctx.createLinearGradient(0, 0, 0, height);
    bg.addColorStop(0, "rgba(10,12,20,0)");
    bg.addColorStop(1, "rgba(10,12,20,0.35)");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, width, this.keyboardTopY);

    // Hit-line glow just above the keyboard.
    const hitGlow = ctx.createLinearGradient(0, this.keyboardTopY - 40, 0, this.keyboardTopY);
    hitGlow.addColorStop(0, "rgba(124,155,255,0)");
    hitGlow.addColorStop(1, "rgba(124,155,255,0.10)");
    ctx.fillStyle = hitGlow;
    ctx.fillRect(0, this.keyboardTopY - 40, width, 40);
    ctx.strokeStyle = "rgba(255,255,255,0.14)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, this.keyboardTopY);
    ctx.lineTo(width, this.keyboardTopY);
    ctx.stroke();
  }

  _tileColor(note) {
    const hue = this.options.colorByPitch ? pitchHue(note.midi_note) : this.options.customColorHue;
    return {
      hue,
      core: `hsl(${hue} 90% 68%)`,
      edge: `hsl(${hue} 95% 52%)`,
      glow: `hsla(${hue}, 95%, 60%,`,
    };
  }

  _drawFallingNotes(ctx, visibleNotes, currentTime, pixelsPerSecond) {
    const { keyboardTopY, layout, options } = this;

    for (const note of visibleNotes) {
      const pos = layout.positions.get(note.midi_note);
      if (!pos) continue;

      const headOffset = Math.max(0, note.start_sec - currentTime);
      const tailOffset = note.end_sec - currentTime;

      const bottomY = keyboardTopY - headOffset * pixelsPerSecond;
      let topY = keyboardTopY - tailOffset * pixelsPerSecond;
      topY = Math.min(topY, bottomY - 3);

      const tileHeight = bottomY - topY;
      if (tileHeight <= 0) continue;

      const widthRatio = pos.isBlack ? 0.76 : 0.8;
      const tileWidth = pos.width * widthRatio;
      const x = pos.centerX - tileWidth / 2;

      const isActive = note.start_sec <= currentTime && currentTime < note.end_sec;
      const color = this._tileColor(note);

      ctx.save();
      if (options.effectsEnabled) {
        ctx.shadowColor = color.glow + (isActive ? "0.85)" : "0.45)");
        ctx.shadowBlur = isActive ? 22 : 12;
      }

      const grad = ctx.createLinearGradient(0, topY, 0, bottomY);
      grad.addColorStop(0, color.edge);
      grad.addColorStop(1, color.core);
      ctx.fillStyle = grad;

      const radius = Math.min(7, tileWidth / 2, tileHeight / 2);
      this._roundRect(ctx, x, topY, tileWidth, tileHeight, radius);
      ctx.fill();

      ctx.shadowBlur = 0;
      ctx.strokeStyle = "rgba(255,255,255,0.35)";
      ctx.lineWidth = 1;
      this._roundRect(ctx, x + 0.5, topY + 0.5, tileWidth - 1, Math.max(0, tileHeight - 1), radius);
      ctx.stroke();
      ctx.restore();

      if (options.showLabels && tileHeight > 22 && tileWidth > 14) {
        ctx.save();
        ctx.fillStyle = "rgba(10,10,16,0.75)";
        ctx.font = "600 10px 'Segoe UI', sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(noteName(note.midi_note), pos.centerX, topY + 13);
        ctx.restore();
      }
    }
  }

  _roundRect(ctx, x, y, w, h, r) {
    const radius = Math.max(0, Math.min(r, w / 2, h / 2));
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.arcTo(x + w, y, x + w, y + h, radius);
    ctx.arcTo(x + w, y + h, x, y + h, radius);
    ctx.arcTo(x, y + h, x, y, radius);
    ctx.arcTo(x, y, x + w, y, radius);
    ctx.closePath();
  }

  // Square top edge (flush against the fall area / adjacent keys), rounded
  // bottom-front edge - reads as a physical key shape rather than a pill.
  _roundRectBottom(ctx, x, y, w, h, r) {
    const radius = Math.max(0, Math.min(r, w / 2, h / 2));
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + w, y);
    ctx.lineTo(x + w, y + h - radius);
    ctx.arcTo(x + w, y + h, x + w - radius, y + h, radius);
    ctx.lineTo(x + radius, y + h);
    ctx.arcTo(x, y + h, x, y + h - radius, radius);
    ctx.closePath();
  }

  _drawKeyboard(ctx, activeMidiSet, currentTime) {
    const { layout, keyboardTopY, height } = this;
    const keyboardHeight = height - keyboardTopY;

    // White keys first.
    for (let midi = MIN_MIDI; midi <= MAX_MIDI; midi++) {
      const pos = layout.positions.get(midi);
      if (pos.isBlack) continue;
      this._drawKey(ctx, midi, pos, keyboardTopY, keyboardHeight, activeMidiSet.has(midi), false);
    }
    // Black keys on top.
    for (let midi = MIN_MIDI; midi <= MAX_MIDI; midi++) {
      const pos = layout.positions.get(midi);
      if (!pos.isBlack) continue;
      this._drawKey(ctx, midi, pos, keyboardTopY, layout.blackKeyHeight, activeMidiSet.has(midi), true);
    }
  }

  _drawKey(ctx, midi, pos, topY, keyHeight, isActive, isBlack) {
    const gap = 1.5;
    const x = pos.x + gap / 2;
    const w = pos.width - gap;
    const radius = isBlack ? 4 : 6;

    // A black key's active glow must never visibly touch the white keys it
    // sits between - clip drawing to a tight box around just this key
    // (a small margin for the glow to breathe, not enough to cross the ~1px
    // gap into a neighboring key's own footprint).
    ctx.save();
    if (isBlack) {
      const clipMargin = 2;
      ctx.beginPath();
      ctx.rect(x - clipMargin, topY - clipMargin, w + clipMargin * 2, keyHeight + clipMargin * 2);
      ctx.clip();
    }

    if (isActive) {
      const hue = this.options.colorByPitch ? pitchHue(midi) : this.options.customColorHue;
      const glowColor = `hsla(${hue}, 95%, 65%, 0.9)`;
      if (this.options.effectsEnabled) {
        ctx.shadowColor = glowColor;
        ctx.shadowBlur = isBlack ? 10 : 22;
      }
      const grad = ctx.createLinearGradient(0, topY, 0, topY + keyHeight);
      grad.addColorStop(0, `hsl(${hue} 95% 70%)`);
      grad.addColorStop(1, isBlack ? `hsl(${hue} 70% 35%)` : `hsl(${hue} 85% 82%)`);
      ctx.fillStyle = grad;
      this._roundRectBottom(ctx, x, topY, w, keyHeight, radius);
      ctx.fill();
    } else {
      const grad = ctx.createLinearGradient(0, topY, 0, topY + keyHeight);
      if (isBlack) {
        grad.addColorStop(0, "#2b2d38");
        grad.addColorStop(0.85, "#151620");
        grad.addColorStop(1, "#0d0e14");
      } else {
        grad.addColorStop(0, "#ffffff");
        grad.addColorStop(0.92, "#f2f3f8");
        grad.addColorStop(1, "#d8dae4");
      }
      ctx.fillStyle = grad;
      this._roundRectBottom(ctx, x, topY, w, keyHeight, radius);
      ctx.fill();
    }

    ctx.shadowBlur = 0;

    if (isBlack) {
      // Thin highlight along the top edge reads as a beveled, raised key.
      ctx.strokeStyle = "rgba(255,255,255,0.18)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x + 1, topY + 0.5);
      ctx.lineTo(x + w - 1, topY + 0.5);
      ctx.stroke();

      ctx.strokeStyle = "rgba(0,0,0,0.55)";
      this._roundRectBottom(ctx, x + 0.5, topY, w - 1, keyHeight - 0.5, radius);
      ctx.stroke();
    } else {
      ctx.strokeStyle = "rgba(20,22,30,0.16)";
      ctx.lineWidth = 1;
      this._roundRectBottom(ctx, x + 0.5, topY, w - 1, keyHeight - 0.5, radius);
      ctx.stroke();

      // A slim brighter strip at the front edge of the key for definition.
      if (!isActive) {
        const frontGrad = ctx.createLinearGradient(0, topY + keyHeight - 10, 0, topY + keyHeight - 2);
        frontGrad.addColorStop(0, "rgba(255,255,255,0)");
        frontGrad.addColorStop(1, "rgba(255,255,255,0.55)");
        ctx.fillStyle = frontGrad;
        this._roundRectBottom(ctx, x + 1, topY + keyHeight - 10, w - 2, 9, radius);
        ctx.fill();
      }
    }

    ctx.restore();
  }

  _drawImpactEffects(ctx, visibleNotes, currentTime) {
    for (const note of visibleNotes) {
      const pos = this.layout.positions.get(note.midi_note);
      if (!pos) continue;

      const age = currentTime - note.start_sec;
      const glow = getImpactGlow(age);
      if (glow > 0) {
        ctx.save();
        const radius = 34 * (1 + (1 - glow) * 0.6);
        const g = ctx.createRadialGradient(
          pos.centerX, this.keyboardTopY, 0,
          pos.centerX, this.keyboardTopY, radius
        );
        const hue = this.options.colorByPitch ? pitchHue(note.midi_note) : this.options.customColorHue;
        g.addColorStop(0, `hsla(${hue}, 100%, 75%, ${0.5 * glow})`);
        g.addColorStop(1, `hsla(${hue}, 100%, 60%, 0)`);
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(pos.centerX, this.keyboardTopY, radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }

      const particles = getImpactParticles(note.midi_note, note.start_sec, pos.centerX, this.keyboardTopY, age);
      if (particles.length) {
        ctx.save();
        const hue = this.options.colorByPitch ? pitchHue(note.midi_note) : this.options.customColorHue;
        for (const p of particles) {
          ctx.fillStyle = `hsla(${hue}, 95%, 75%, ${p.alpha})`;
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();
      }
    }
  }
}
