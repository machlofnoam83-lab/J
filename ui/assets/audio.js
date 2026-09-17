/* ══════════════════════════════════════════════════════════════════════
   J.A.R.V.I.S. — audio playback
   Decodes the WAV frames the local voice engine produces, plays them through
   WebAudio, and feeds an analyser into the HUD so the reactor breathes with
   the voice. No network, no external media.
   ══════════════════════════════════════════════════════════════════════ */
'use strict';

const Voice = (() => {

  let ctxA = null;          // AudioContext
  let analyser = null;
  let master = null;
  let queue = [];           // pending AudioBuffers
  let playing = false;
  let timeBuf = null;
  let muted = false;
  let onStateChange = () => {};
  let currentSource = null;

  function ensure() {
    if (ctxA) return ctxA;
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return null;
    ctxA = new AC();
    master = ctxA.createGain();
    master.gain.value = 1.0;
    analyser = ctxA.createAnalyser();
    analyser.fftSize = 256;
    analyser.smoothingTimeConstant = 0.6;
    master.connect(analyser);
    analyser.connect(ctxA.destination);
    timeBuf = new Float32Array(analyser.fftSize);
    return ctxA;
  }

  async function unlock() {
    const c = ensure();
    if (!c) return false;
    if (c.state === 'suspended') { try { await c.resume(); } catch (_) {} }
    return c.state === 'running';
  }

  /* WAV (PCM16/PCM32/float, mono or stereo) → Float32 mono samples */
  function decodeWav(bytes) {
    const dv = new DataView(bytes);
    if (bytes.byteLength < 44 || dv.getUint32(0, false) !== 0x52494646) return null; // 'RIFF'
    let off = 12, fmt = null, dataOff = -1, dataLen = 0;
    while (off + 8 <= bytes.byteLength) {
      const id = dv.getUint32(off, false);
      const size = dv.getUint32(off + 4, true);
      if (id === 0x666d7420) fmt = { off: off + 8, size };                       // 'fmt '
      else if (id === 0x64617461) { dataOff = off + 8; dataLen = size; break; } // 'data'
      off += 8 + size + (size & 1);
    }
    if (!fmt || dataOff < 0) return null;
    const audioFmt = dv.getUint16(fmt.off, true);
    const channels = dv.getUint16(fmt.off + 2, true) || 1;
    const sampleRate = dv.getUint32(fmt.off + 4, true);
    const bits = dv.getUint16(fmt.off + 14, true);
    const bytesPer = bits / 8;
    const frames = Math.min(Math.floor(dataLen / (bytesPer * channels)),
                            Math.floor((bytes.byteLength - dataOff) / (bytesPer * channels)));
    if (frames <= 0 || sampleRate <= 0) return null;

    const out = new Float32Array(frames);
    let p = dataOff;
    for (let i = 0; i < frames; i++) {
      let acc = 0;
      for (let c = 0; c < channels; c++) {
        let v = 0;
        if (audioFmt === 3 && bits === 32) { v = dv.getFloat32(p, true); p += 4; }
        else if (bits === 32) { v = dv.getInt32(p, true) / 2147483648; p += 4; }
        else if (bits === 24) {
          const b0 = dv.getUint8(p), b1 = dv.getUint8(p + 1), b2 = dv.getInt8(p + 2);
          v = ((b2 << 24) | (b1 << 16) | (b0 << 8)) / 2147483648 * 256; p += 3;
        } else if (bits === 16) { v = dv.getInt16(p, true) / 32768; p += 2; }
        else { v = (dv.getUint8(p) - 128) / 128; p += 1; }
        acc += v;
      }
      out[i] = acc / channels;
    }
    return { samples: out, sampleRate };
  }

  function b64ToBytes(b64) {
    const bin = atob(b64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    return arr;
  }

  function makeBuffer(decoded) {
    const c = ensure(); if (!c) return null;
    const { samples, sampleRate } = decoded;
    const buf = c.createBuffer(1, samples.length, sampleRate);
    buf.copyToChannel(samples, 0);
    return buf;
  }

  function pump() {
    if (playing || !queue.length) {
      if (!queue.length && !playing) onStateChange(false);
      return;
    }
    const c = ensure(); if (!c) { queue = []; return; }
    if (c.state === 'suspended') c.resume().catch(() => {});
    const buf = queue.shift();
    playing = true;
    onStateChange(true);
    const src = c.createBufferSource();
    src.buffer = buf;
    src.connect(master);
    currentSource = src;
    src.onended = () => {
      playing = false; currentSource = null;
      if (queue.length) setTimeout(pump, 12);
      else onStateChange(false);
    };
    try { src.start(); } catch (_) { playing = false; }
  }

  /* public: hand it the base64 WAV frame from the server */
  function playB64(b64) {
    if (muted || !b64) return false;
    const bytes = b64ToBytes(b64);
    const decoded = decodeWav(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
    if (!decoded) return false;
    const buf = makeBuffer(decoded);
    if (!buf) return false;
    queue.push(buf);
    pump();
    return true;
  }

  function stop() {
    queue = [];
    if (currentSource) { try { currentSource.onended = null; currentSource.stop(); } catch (_) {} }
    currentSource = null; playing = false;
    onStateChange(false);
  }

  /* level + waveform sampling — called by the HUD each frame via rAF */
  function startMeter() {
    const tick = () => {
      if (analyser && typeof window.HUD !== 'undefined') {
        analyser.getFloatTimeDomainData(timeBuf);
        let peak = 0;
        for (let i = 0; i < timeBuf.length; i++) { const a = Math.abs(timeBuf[i]); if (a > peak) peak = a; }
        window.HUD.setAudio(playing ? clamp01(peak * 2.6) : 0, playing ? timeBuf : null);
      }
      requestAnimationFrame(tick);
    };
    tick();
  }
  const clamp01 = v => Math.max(0, Math.min(1, v));

  return {
    unlock, playB64, stop, startMeter,
    setMuted(m) { muted = !!m; if (muted) stop(); },
    get muted() { return muted; },
    get speaking() { return playing; },
    get pending() { return queue.length; },
    onState(cb) { onStateChange = cb || (() => {}); },
    decodeWav, ensure
  };
})();

if (typeof window !== 'undefined') window.Voice = Voice;
