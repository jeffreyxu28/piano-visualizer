// Thin wrapper around the <audio> element that drives the render loop.
//
// The audio element's `currentTime` is the ONLY clock in this app. Every
// animation frame reads it fresh and asks the visualizer to draw exactly
// that instant - there is no separate incrementing timer, so pausing,
// seeking, restarting, and playback-rate changes all stay correct without
// any special-case handling.
//
// One deliberate, narrowly-scoped exception: once the audio genuinely
// finishes, there's nothing left to stay in sync with, so a short
// wall-clock "grace period" lets the last note's key-glow/ember decay
// finish playing out instead of freezing the instant the source file
// ends (which, for a recording with little trailing silence, could cut
// the decay off mid-fade).

const END_GRACE_SECONDS = 1.5;

export class Player {
  constructor(audioEl, visualizer, callbacks = {}) {
    this.audio = audioEl;
    this.visualizer = visualizer;
    this.callbacks = callbacks;
    this.isSeeking = false;
    this._rafId = null;
    this._intervalId = null;

    // Volume can go above 100% (up to 150%) which HTMLMediaElement.volume
    // can't express on its own (it's clamped to [0,1]) - a Web Audio
    // GainNode sits after the element and provides the extra headroom.
    // Built lazily on first setVolume()/play() call (a user gesture is
    // required to start an AudioContext in most browsers).
    this._audioContext = null;
    this._gainNode = null;
    this._pendingVolume = 1.0;
    this._endedAt = null; // performance.now() timestamp when 'ended' fired, or null

    this.audio.addEventListener("play", () => {
      this._endedAt = null;
      this.callbacks.onPlayStateChange?.(true);
    });
    this.audio.addEventListener("pause", () => this.callbacks.onPlayStateChange?.(false));
    this.audio.addEventListener("ended", () => {
      this._endedAt = performance.now();
      this.callbacks.onPlayStateChange?.(false);
    });
    this.audio.addEventListener("loadedmetadata", () => {
      this.callbacks.onDurationChange?.(this.audio.duration || 0);
    });
    this.audio.addEventListener("error", () => {
      this.callbacks.onError?.(this.audio.error);
    });

    // requestAnimationFrame is the primary, smoothest driver. It can be
    // throttled by the browser in some embedded/backgrounded contexts
    // though, so 'timeupdate' (fired natively by the media element) and a
    // low-frequency interval act as redundant fallbacks - between the
    // three, the visualizer is guaranteed to keep advancing with playback.
    this.audio.addEventListener("timeupdate", () => this._tick());
    this.audio.addEventListener("seeked", () => this._tick());
    this._intervalId = window.setInterval(() => this._tick(), 100);

    this._loop();
  }

  _tick() {
    const realTime = this.audio.currentTime || 0;
    this.visualizer.render(this._effectiveCurrentTime(realTime));

    if (!this.isSeeking) {
      this.callbacks.onTick?.(realTime, this.audio.duration || 0);
    }
  }

  // While actively playing/seeking this is just audio.currentTime. Only
  // once the audio has truly ended does it coast a little further (wall
  // clock, capped at END_GRACE_SECONDS) so final decay effects can finish.
  _effectiveCurrentTime(realTime) {
    if (this._endedAt === null) return realTime;
    const graceElapsed = (performance.now() - this._endedAt) / 1000;
    if (graceElapsed >= END_GRACE_SECONDS) return realTime;
    return realTime + graceElapsed;
  }

  load(url) {
    return new Promise((resolve, reject) => {
      const onLoaded = () => {
        this.audio.removeEventListener("loadedmetadata", onLoaded);
        this.audio.removeEventListener("error", onError);
        resolve();
      };
      const onError = () => {
        this.audio.removeEventListener("loadedmetadata", onLoaded);
        this.audio.removeEventListener("error", onError);
        reject(this.audio.error);
      };
      this.audio.addEventListener("loadedmetadata", onLoaded);
      this.audio.addEventListener("error", onError);
      this.audio.src = url;
      this.audio.load();
    });
  }

  _ensureAudioGraph() {
    if (this._gainNode) {
      if (this._audioContext.state === "suspended") this._audioContext.resume();
      return true;
    }
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return false; // unsupported browser - falls back to plain element.volume

    try {
      this._audioContext = new Ctx();
      const source = this._audioContext.createMediaElementSource(this.audio);
      this._gainNode = this._audioContext.createGain();
      source.connect(this._gainNode).connect(this._audioContext.destination);
      this._gainNode.gain.value = this._pendingVolume;
      this.audio.volume = 1; // all volume control now goes through the gain node
      return true;
    } catch {
      this._audioContext = null;
      this._gainNode = null;
      return false;
    }
  }

  play() {
    this._ensureAudioGraph();
    const result = this.audio.play();
    result?.catch?.((err) => this.callbacks.onError?.(err));
    return result;
  }

  pause() {
    this.audio.pause();
  }

  togglePlay() {
    if (this.audio.paused) return this.play();
    this.pause();
    return Promise.resolve();
  }

  restart() {
    this._endedAt = null;
    this.audio.currentTime = 0;
  }

  seekTo(seconds) {
    this._endedAt = null;
    const duration = this.audio.duration || 0;
    this.audio.currentTime = Math.min(Math.max(0, seconds), duration);
  }

  setVolume(v) {
    const clamped = Math.min(1.5, Math.max(0, v));
    this._pendingVolume = clamped;

    if (this._ensureAudioGraph()) {
      this._gainNode.gain.value = clamped;
    } else {
      // No Web Audio support - degrade gracefully to the element's native
      // [0,1] range (values above 100% simply have no further effect).
      this.audio.volume = Math.min(1, clamped);
    }
  }

  setPlaybackRate(rate) {
    this.audio.playbackRate = rate;
    try {
      this.audio.preservesPitch = true;
      // Vendor-prefixed fallbacks for older engines.
      this.audio.mozPreservesPitch = true;
      this.audio.webkitPreservesPitch = true;
    } catch {
      /* not supported everywhere; playback still works without it */
    }
  }

  get duration() {
    return this.audio.duration || 0;
  }

  get currentTime() {
    return this.audio.currentTime || 0;
  }

  get paused() {
    return this.audio.paused;
  }

  _loop() {
    this._tick();
    this._rafId = requestAnimationFrame(() => this._loop());
  }

  destroy() {
    if (this._rafId !== null) cancelAnimationFrame(this._rafId);
    if (this._intervalId !== null) window.clearInterval(this._intervalId);
    this._audioContext?.close?.();
  }
}
