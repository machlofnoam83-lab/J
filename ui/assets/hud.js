/* ══════════════════════════════════════════════════════════════════════
   J.A.R.V.I.S. — HUD rendering engine (pure canvas, no libraries)
   arc reactor · neural constellation · circular waveform · gauges · sparks
   ══════════════════════════════════════════════════════════════════════ */
'use strict';

const HUD = (() => {

  // ------------------------------------------------------------------ state
  const S = {
    state: 'booting',        // booting | idle | thinking | speaking | killed | offline
    energy: 0.35,            // 0..1 target
    energyNow: 0.35,
    audio: 0,                // 0..1 instantaneous level from playback
    audioWave: new Float32Array(128),
    wavePhase: 0,
    arcs: [],                // transient lightning bolts
    pulses: [],              // travelling radial pulses
    spin: 0,
    spin2: 0,
    spin3: 0,
    t: 0,
    lastFrame: 0,
    fps: 60
  };

  const PALETTE = {
    cyan: '#5ee7ff', dim: '#2a93ad', ice: '#d8f7ff',
    amber: '#ffb454', red: '#ff4d63', green: '#46e39a', violet: '#9d7bff'
  };
  const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const color = () => ({
    cyan: css('--cyan') || PALETTE.cyan, dim: css('--cyan-dim') || PALETTE.dim,
    ice: css('--ice') || PALETTE.ice, amber: css('--amber') || PALETTE.amber,
    red: css('--red') || PALETTE.red, green: css('--green') || PALETTE.green
  });

  const TAU = Math.PI * 2;
  const lerp = (a, b, t) => a + (b - a) * t;
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  function fit(canvas) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(r.width * dpr));
    const h = Math.max(1, Math.round(r.height * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w: r.width, h: r.height };
  }

  // ══════════════════════════════ arc reactor ══════════════════════════════
  let reactorCanvas = null;

  function ringSegments(ctx, cx, cy, r, count, gap, width, phase, litFn, col, alpha) {
    const step = TAU / count;
    for (let i = 0; i < count; i++) {
      const a0 = phase + i * step + gap / 2;
      const a1 = phase + (i + 1) * step - gap / 2;
      const lit = litFn ? litFn(i, count) : 1;
      ctx.beginPath();
      ctx.arc(cx, cy, r, a0, a1);
      ctx.lineWidth = width;
      ctx.strokeStyle = col;
      ctx.globalAlpha = alpha * (0.22 + 0.78 * lit);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  function coils(ctx, cx, cy, rIn, rOut, count, phase, energy, col) {
    const step = TAU / count;
    for (let i = 0; i < count; i++) {
      const a = phase + i * step;
      const w = step * 0.46;
      const boost = 0.5 + 0.5 * Math.sin(S.t * 2.4 + i * 0.9);
      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(a - w) * rIn, cy + Math.sin(a - w) * rIn);
      ctx.lineTo(cx + Math.cos(a - w * 0.55) * rOut, cy + Math.sin(a - w * 0.55) * rOut);
      ctx.lineTo(cx + Math.cos(a + w * 0.55) * rOut, cy + Math.sin(a + w * 0.55) * rOut);
      ctx.lineTo(cx + Math.cos(a + w) * rIn, cy + Math.sin(a + w) * rIn);
      ctx.closePath();
      const g = ctx.createLinearGradient(cx + Math.cos(a) * rIn, cy + Math.sin(a) * rIn,
                                        cx + Math.cos(a) * rOut, cy + Math.sin(a) * rOut);
      g.addColorStop(0, col);
      g.addColorStop(1, 'rgba(255,255,255,0)');
      ctx.fillStyle = g;
      ctx.globalAlpha = 0.10 + 0.42 * energy * (0.45 + 0.55 * boost);
      ctx.fill();
      ctx.globalAlpha = 0.5 * energy;
      ctx.strokeStyle = col;
      ctx.lineWidth = 0.8;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }

  function spawnArc(C) {
    const n = 2 + Math.floor(Math.random() * 3);
    for (let k = 0; k < n; k++) {
      const a = Math.random() * TAU;
      const r0 = C.r * (0.34 + Math.random() * 0.2);
      const r1 = C.r * (0.62 + Math.random() * 0.34);
      const pts = [];
      const steps = 7;
      for (let i = 0; i <= steps; i++) {
        const t = i / steps;
        const rr = lerp(r0, r1, t) + (Math.random() - 0.5) * C.r * 0.06;
        const aa = a + (Math.random() - 0.5) * 0.35 * t;
        pts.push([C.cx + Math.cos(aa) * rr, C.cy + Math.sin(aa) * rr]);
      }
      S.arcs.push({ pts, life: 1 });
    }
  }

  function spawnPulse(C) {
    S.pulses.push({ a: Math.random() * TAU, r: C.r * 0.2, v: C.r * (0.7 + Math.random() * 0.8), life: 1 });
  }

  function drawReactor(dt) {
    if (!reactorCanvas) return;
    const { ctx, w, h } = fit(reactorCanvas);
    const C = color();
    const cx = w / 2, cy = h / 2;
    const R = Math.min(w, h) / 2 - 8;
    C.r = R; C.cx = cx; C.cy = cy;

    ctx.clearRect(0, 0, w, h);

    // energy easing + audio coupling
    const target = S.state === 'killed' ? 0.04
                 : S.state === 'thinking' ? 0.95
                 : S.state === 'speaking' ? 0.8
                 : S.state === 'offline' ? 0.08 : 0.42;
    S.energy = target;
    S.energyNow = lerp(S.energyNow, S.energy + S.audio * 0.35, clamp(dt * 4, 0, 1));
    const E = clamp(S.energyNow, 0, 1.4);
    const audioKick = S.audio;

    const accent = S.state === 'killed' ? C.red
                 : S.state === 'speaking' ? C.amber
                 : S.state === 'thinking' ? C.cyan : C.cyan;

    // halo
    const halo = ctx.createRadialGradient(cx, cy, R * 0.1, cx, cy, R * 1.06);
    halo.addColorStop(0, `rgba(94,231,255,${0.16 * E})`);
    halo.addColorStop(0.55, `rgba(94,231,255,${0.05 * E})`);
    halo.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = halo;
    ctx.beginPath(); ctx.arc(cx, cy, R * 1.06, 0, TAU); ctx.fill();

    S.spin += dt * (0.18 + 0.5 * E);
    S.spin2 -= dt * (0.32 + 0.8 * E);
    S.spin3 += dt * (0.09 + 0.25 * E);

    // ── outer tick ring
    ctx.save();
    ctx.translate(cx, cy); ctx.rotate(S.spin3);
    const ticks = 96;
    for (let i = 0; i < ticks; i++) {
      const a = (i / ticks) * TAU;
      const long = i % 8 === 0;
      const r0 = R * (long ? 0.93 : 0.965);
      ctx.beginPath();
      ctx.moveTo(Math.cos(a) * r0, Math.sin(a) * r0);
      ctx.lineTo(Math.cos(a) * R, Math.sin(a) * R);
      ctx.strokeStyle = accent;
      ctx.globalAlpha = long ? 0.55 : 0.18;
      ctx.lineWidth = long ? 1.4 : 0.8;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    ctx.restore();

    // ── segmented outer ring (sweep-lit)
    const sweep = (S.t * 0.9) % 1;
    ringSegments(ctx, cx, cy, R * 0.885, 36, 0.045, 3.2, S.spin2,
      (i, n) => {
        const p = i / n;
        let d = Math.abs(p - sweep); d = Math.min(d, 1 - d);
        return clamp(1 - d * 5.5, 0.06, 1) * (0.4 + 0.6 * E);
      }, accent, 0.9);

    // ── dashed mid ring
    ctx.save();
    ctx.translate(cx, cy); ctx.rotate(S.spin);
    ctx.setLineDash([R * 0.055, R * 0.035]);
    ctx.beginPath(); ctx.arc(0, 0, R * 0.80, 0, TAU);
    ctx.strokeStyle = C.dim; ctx.lineWidth = 1.6; ctx.globalAlpha = 0.55; ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();

    // ── coil wedges
    coils(ctx, cx, cy, R * 0.44, R * 0.74, 10, S.spin2 * 0.4, E, accent);

    // ── hex plate ring
    ctx.save();
    ctx.translate(cx, cy); ctx.rotate(-S.spin * 0.6);
    const hexes = 6;
    for (let i = 0; i < hexes; i++) {
      const a = (i / hexes) * TAU;
      const hx = Math.cos(a) * R * 0.615, hy = Math.sin(a) * R * 0.615;
      ctx.beginPath();
      for (let k = 0; k < 6; k++) {
        const aa = (k / 6) * TAU + Math.PI / 6;
        const px = hx + Math.cos(aa) * R * 0.075, py = hy + Math.sin(aa) * R * 0.075;
        k ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
      }
      ctx.closePath();
      ctx.strokeStyle = accent; ctx.globalAlpha = 0.25 + 0.35 * E; ctx.lineWidth = 1; ctx.stroke();
      ctx.fillStyle = accent; ctx.globalAlpha = 0.045 * (1 + audioKick * 3); ctx.fill();
    }
    ctx.globalAlpha = 1;
    ctx.restore();

    // ── inner ring + waveform
    ctx.beginPath(); ctx.arc(cx, cy, R * 0.375, 0, TAU);
    ctx.strokeStyle = accent; ctx.globalAlpha = 0.5; ctx.lineWidth = 1.1; ctx.stroke();
    ctx.globalAlpha = 1;

    if (S.audioWave.length) {
      ctx.beginPath();
      const N = S.audioWave.length;
      for (let i = 0; i < N; i++) {
        const a = (i / N) * TAU - Math.PI / 2;
        const v = S.audioWave[i];
        const rr = R * 0.375 + v * R * 0.20;
        const px = cx + Math.cos(a) * rr, py = cy + Math.sin(a) * rr;
        i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
      }
      ctx.closePath();
      ctx.strokeStyle = S.state === 'speaking' ? C.amber : C.cyan;
      ctx.globalAlpha = 0.30 + 0.55 * clamp(audioKick * 2.4, 0, 1);
      ctx.lineWidth = 1.5;
      ctx.shadowBlur = 14; ctx.shadowColor = ctx.strokeStyle;
      ctx.stroke();
      ctx.shadowBlur = 0; ctx.globalAlpha = 1;
    }

    // ── travelling pulses
    if (Math.random() < dt * (2 + 8 * E)) spawnPulse(C);
    S.pulses = S.pulses.filter(p => {
      p.r += p.v * dt; p.life -= dt * 1.1;
      if (p.life <= 0 || p.r > R) return false;
      const px = cx + Math.cos(p.a) * p.r, py = cy + Math.sin(p.a) * p.r;
      ctx.beginPath(); ctx.arc(px, py, 1.8 + 1.4 * p.life, 0, TAU);
      ctx.fillStyle = accent; ctx.globalAlpha = 0.75 * p.life;
      ctx.shadowBlur = 10; ctx.shadowColor = accent; ctx.fill();
      ctx.shadowBlur = 0; ctx.globalAlpha = 1;
      return true;
    });

    // ── lightning arcs (thinking)
    if (S.state === 'thinking' && Math.random() < dt * 6) spawnArc(C);
    S.arcs = S.arcs.filter(arc => {
      arc.life -= dt * 3.4;
      if (arc.life <= 0) return false;
      ctx.beginPath();
      arc.pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
      ctx.strokeStyle = C.ice; ctx.globalAlpha = arc.life * 0.85;
      ctx.lineWidth = 1.1 + arc.life; ctx.shadowBlur = 12; ctx.shadowColor = accent;
      ctx.stroke(); ctx.shadowBlur = 0; ctx.globalAlpha = 1;
      return true;
    });

    // ── core
    const coreR = R * (0.20 + 0.035 * Math.sin(S.t * 2.2) + 0.05 * audioKick);
    const cg = ctx.createRadialGradient(cx, cy, 0, cx, cy, coreR * 2.1);
    cg.addColorStop(0, `rgba(255,255,255,${0.85 * clamp(E + 0.2, 0, 1)})`);
    cg.addColorStop(0.28, accent);
    cg.addColorStop(0.62, `rgba(94,231,255,${0.16 * E})`);
    cg.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = cg;
    ctx.beginPath(); ctx.arc(cx, cy, coreR * 2.1, 0, TAU); ctx.fill();

    // core triangle (Mark-VII signature)
    ctx.save();
    ctx.translate(cx, cy); ctx.rotate(S.spin * 0.35);
    ctx.beginPath();
    for (let i = 0; i < 3; i++) {
      const a = (i / 3) * TAU - Math.PI / 2;
      const px = Math.cos(a) * coreR * 0.92, py = Math.sin(a) * coreR * 0.92;
      i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    }
    ctx.closePath();
    ctx.strokeStyle = 'rgba(255,255,255,.85)'; ctx.lineWidth = 1.4;
    ctx.globalAlpha = 0.5 + 0.5 * E; ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.restore();

    // ── radar sweep
    const ra = (S.t * 0.7) % TAU;
    const rg = ctx.createConicGradient ? ctx.createConicGradient(ra, cx, cy) : null;
    if (rg) {
      rg.addColorStop(0, `rgba(94,231,255,${0.20 * E})`);
      rg.addColorStop(0.08, 'rgba(94,231,255,0)');
      rg.addColorStop(1, 'rgba(94,231,255,0)');
      ctx.save();
      ctx.beginPath(); ctx.arc(cx, cy, R * 0.94, 0, TAU); ctx.clip();
      ctx.fillStyle = rg; ctx.fillRect(cx - R, cy - R, R * 2, R * 2);
      ctx.restore();
    }
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(ra) * R * 0.94, cy + Math.sin(ra) * R * 0.94);
    ctx.strokeStyle = accent; ctx.globalAlpha = 0.30 * E; ctx.lineWidth = 1; ctx.stroke();
    ctx.globalAlpha = 1;

    // ── corner brackets + readouts
    ctx.strokeStyle = C.dim; ctx.globalAlpha = 0.5; ctx.lineWidth = 1;
    const b = R * 0.99, bl = R * 0.16;
    [[-1,-1],[1,-1],[1,1],[-1,1]].forEach(([sx, sy]) => {
      ctx.beginPath();
      ctx.moveTo(cx + sx * b - sx * bl, cy + sy * b);
      ctx.lineTo(cx + sx * b, cy + sy * b);
      ctx.lineTo(cx + sx * b, cy + sy * b - sy * bl);
      ctx.stroke();
    });
    ctx.globalAlpha = 1;

    ctx.font = '9px ' + (css('--font-mono') || 'monospace');
    ctx.fillStyle = C.dim; ctx.globalAlpha = 0.85; ctx.textAlign = 'center';
    ctx.fillText(`PWR ${(E * 100).toFixed(0)}%`, cx, cy + R + 4);
    ctx.globalAlpha = 1;
  }

  // ══════════════════════ background neural constellation ══════════════════
  let bgCanvas = null;
  const nodes = [];
  const streaks = [];

  function initBg() {
    bgCanvas = document.getElementById('bg-canvas');
    if (!bgCanvas) return;
    const n = 68;
    for (let i = 0; i < n; i++) {
      nodes.push({
        x: Math.random(), y: Math.random(),
        vx: (Math.random() - 0.5) * 0.012, vy: (Math.random() - 0.5) * 0.012,
        r: 0.6 + Math.random() * 1.6, ph: Math.random() * TAU
      });
    }
  }

  function drawBg(dt) {
    if (!bgCanvas) return;
    const { ctx, w, h } = fit(bgCanvas);
    const C = color();
    ctx.clearRect(0, 0, w, h);

    const excited = S.state === 'thinking' ? 2.4 : S.state === 'speaking' ? 1.7 : 1;
    for (const nd of nodes) {
      nd.x += nd.vx * dt * excited; nd.y += nd.vy * dt * excited;
      if (nd.x < 0 || nd.x > 1) nd.vx *= -1;
      if (nd.y < 0 || nd.y > 1) nd.vy *= -1;
      nd.x = clamp(nd.x, 0, 1); nd.y = clamp(nd.y, 0, 1);
    }

    // links
    ctx.lineWidth = 0.6;
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j];
        const dx = (a.x - b.x) * w, dy = (a.y - b.y) * h;
        const d2 = dx * dx + dy * dy;
        const lim = 150 * 150;
        if (d2 < lim) {
          const al = (1 - d2 / lim) * 0.16 * excited;
          ctx.strokeStyle = C.cyan; ctx.globalAlpha = al;
          ctx.beginPath(); ctx.moveTo(a.x * w, a.y * h); ctx.lineTo(b.x * w, b.y * h); ctx.stroke();
        }
      }
    }
    // nodes
    for (const nd of nodes) {
      const tw = 0.5 + 0.5 * Math.sin(S.t * 1.6 + nd.ph);
      ctx.beginPath(); ctx.arc(nd.x * w, nd.y * h, nd.r * (1 + 0.35 * tw), 0, TAU);
      ctx.fillStyle = C.cyan; ctx.globalAlpha = 0.20 + 0.45 * tw;
      ctx.fill();
    }
    ctx.globalAlpha = 1;

    // data streaks
    if (Math.random() < dt * 1.6) {
      streaks.push({ x: Math.random(), y: Math.random(), len: 0.04 + Math.random() * 0.12,
                     sp: (0.25 + Math.random() * 0.6) * (Math.random() < 0.5 ? 1 : -1), life: 1,
                     horiz: Math.random() < 0.7 });
    }
    for (let i = streaks.length - 1; i >= 0; i--) {
      const s = streaks[i];
      s.life -= dt * 0.9;
      if (s.horiz) s.x += s.sp * dt; else s.y += s.sp * dt;
      if (s.life <= 0) { streaks.splice(i, 1); continue; }
      const g = s.horiz
        ? ctx.createLinearGradient(s.x * w, 0, (s.x + s.len) * w, 0)
        : ctx.createLinearGradient(0, s.y * h, 0, (s.y + s.len) * h);
      g.addColorStop(0, 'rgba(94,231,255,0)');
      g.addColorStop(1, `rgba(94,231,255,${0.5 * s.life})`);
      ctx.strokeStyle = g; ctx.lineWidth = 1.2;
      ctx.beginPath();
      if (s.horiz) { ctx.moveTo(s.x * w, s.y * h); ctx.lineTo((s.x + s.len) * w, s.y * h); }
      else { ctx.moveTo(s.x * w, s.y * h); ctx.lineTo(s.x * w, (s.y + s.len) * h); }
      ctx.stroke();
    }
  }

  // ═════════════════════════════ gauges & sparks ══════════════════════════
  const gauges = {};

  function initGauges() {
    document.querySelectorAll('.g-canvas').forEach(cv => {
      gauges[cv.dataset.g] = { canvas: cv, value: 0, shown: 0 };
    });
  }

  function setGauge(name, pct) {
    const g = gauges[name]; if (g) g.value = clamp(pct, 0, 100);
  }

  function drawGauges(dt) {
    const C = color();
    for (const key in gauges) {
      const g = gauges[key];
      const { ctx, w, h } = fit(g.canvas);
      g.shown = lerp(g.shown, g.value, clamp(dt * 3.5, 0, 1));
      const cx = w / 2, cy = h / 2, r = Math.min(w, h) / 2 - 5;
      ctx.clearRect(0, 0, w, h);
      const a0 = Math.PI * 0.75, a1 = Math.PI * 2.25;
      ctx.beginPath(); ctx.arc(cx, cy, r, a0, a1);
      ctx.strokeStyle = 'rgba(255,255,255,.07)'; ctx.lineWidth = 4; ctx.stroke();
      const p = g.shown / 100;
      const col = g.shown > 88 ? C.red : g.shown > 68 ? C.amber : C.cyan;
      ctx.beginPath(); ctx.arc(cx, cy, r, a0, a0 + (a1 - a0) * p);
      ctx.strokeStyle = col; ctx.lineWidth = 4; ctx.lineCap = 'round';
      ctx.shadowBlur = 10; ctx.shadowColor = col; ctx.stroke(); ctx.shadowBlur = 0;
      ctx.beginPath(); ctx.arc(cx, cy, r - 6.5, 0, TAU);
      ctx.strokeStyle = col; ctx.globalAlpha = 0.16; ctx.lineWidth = 1; ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }

  const sparks = {};
  function sparkPush(name, v) {
    if (!sparks[name]) sparks[name] = { canvas: document.getElementById('spark-' + name), data: [] };
    const s = sparks[name];
    if (!s.canvas) return;
    s.data.push(v);
    if (s.data.length > 180) s.data.shift();
  }
  function drawSparks() {
    const C = color();
    for (const k in sparks) {
      const s = sparks[k]; if (!s.canvas) continue;
      const { ctx, w, h } = fit(s.canvas);
      ctx.clearRect(0, 0, w, h);
      if (s.data.length < 2) continue;
      const max = Math.max(4, ...s.data);
      ctx.beginPath();
      s.data.forEach((v, i) => {
        const x = (i / (s.data.length - 1)) * w;
        const y = h - (v / max) * (h - 3) - 1.5;
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      const col = k === 'cpu' ? C.cyan : C.green;
      ctx.strokeStyle = col; ctx.lineWidth = 1.2; ctx.globalAlpha = 0.9;
      ctx.shadowBlur = 8; ctx.shadowColor = col; ctx.stroke(); ctx.shadowBlur = 0;
      ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
      const g = ctx.createLinearGradient(0, 0, 0, h);
      g.addColorStop(0, `rgba(94,231,255,.22)`); g.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = g; ctx.globalAlpha = 0.5; ctx.fill(); ctx.globalAlpha = 1;
    }
  }

  // ══════════════════════════════ main loop ══════════════════════════════
  function frame(now) {
    const dt = Math.min(0.05, (now - (S.lastFrame || now)) / 1000);
    S.lastFrame = now; S.t += dt;
    if (dt > 0) S.fps = lerp(S.fps, 1 / dt, 0.05);

    drawReactor(dt);
    drawBg(dt);
    drawGauges(dt);
    drawSparks();

    S.audio = lerp(S.audio, 0, clamp(dt * 6, 0, 1));
    if (S.audio < 0.002) S.audioWave.fill(0);

    requestAnimationFrame(frame);
  }

  // ══════════════════════════════ public API ══════════════════════════════
  return {
    init() {
      reactorCanvas = document.getElementById('reactor');
      initBg(); initGauges();
      requestAnimationFrame(frame);
      window.addEventListener('resize', () => { reactorCanvas && fit(reactorCanvas); });
    },
    setState(s) { S.state = s; },
    getState() { return S.state; },
    setAudio(level, wave) {
      S.audio = clamp(level, 0, 1);
      if (wave && wave.length) {
        const N = S.audioWave.length;
        for (let i = 0; i < N; i++) S.audioWave[i] = wave[Math.floor(i * wave.length / N)] || 0;
      }
    },
    setGauge, sparkPush,
    get fps() { return Math.round(S.fps); },
    get energy() { return S.energyNow; }
  };
})();

if (typeof window !== 'undefined') window.HUD = HUD;
