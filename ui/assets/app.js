/* ══════════════════════════════════════════════════════════════════════
   J.A.R.V.I.S. — HUD controller
   WebSocket client · transcript · event stream · telemetry · permissions
   · boot theatre (driven by real measurements) · local speech capture
   ══════════════════════════════════════════════════════════════════════ */
'use strict';

(() => {

  // ══════════════════════════════ plumbing ══════════════════════════════
  const $ = s => document.querySelector(s);
  const $$ = s => Array.from(document.querySelectorAll(s));
  const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g,
      c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const hhmmss = t => { const d = new Date(t * 1000); return d.toTimeString().slice(0, 8); };
  const clockFmt = s => {
    s = Math.max(0, Math.floor(s));
    const h = String(Math.floor(s / 3600)).padStart(2, '0');
    const m = String(Math.floor((s % 3600) / 60)).padStart(2, '0');
    const x = String(s % 60).padStart(2, '0');
    return h === '00' ? `${m}:${x}` : `${h}:${m}:${x}`;
  };

  let HTTP = '';            // e.g. http://127.0.0.1:8756
  let WS = '';
  let ws = null;
  let wsAlive = false;
  let restOk = false;          // REST transport proved reachable
  let lastEventId = 0;         // cursor for the polling fallback
  let retry = 600;
  let appState = 'booting';
  let busy = { thinking: 0, speaking: 0, killed: false };
  let agentBusy = {};       // agent id -> expiry ts
  let seq = 0;

  const MODES = {
    chat:   { ph: 'דבר אליי, אדוני… (Enter לשליחה · Shift+Enter לשורה חדשה)', prefix: '' },
    code:   { ph: 'תאר את הכלי שתרצה שייבנה עבורך…', prefix: 'כתוב קוד: ' },
    math:   { ph: 'ביטוי מתמטי — למשל 17*23, שורש של 144, 15 אחוז מ 240', prefix: 'חשב: ' },
    system: { ph: 'פקודת מערכת — מצב, טלמטריה, זיכרון, סקירת קבצים…', prefix: '' }
  };
  let mode = 'chat';

  const AGENTS = [
    { id: 'jarvis',     name: 'JARVIS',     role: 'המנצח' },
    { id: 'nano',       name: 'NANO-1',     role: 'רשת עצבית' },
    { id: 'hephaestus', name: 'HEPHAESTUS', role: 'מתכנת' },
    { id: 'mnemosyne',  name: 'MNEMOSYNE',  role: 'זיכרון' },
    { id: 'argus',      name: 'ARGUS',      role: 'אבטחה' },
    { id: 'hermes',     name: 'HERMES',     role: 'כלים' },
    { id: 'vox',        name: 'VOX',        role: 'קול' }
  ];

  async function api(path, opts) {
    const r = await fetch(HTTP + path, opts);
    if (!r.ok) throw new Error(`${path} → HTTP ${r.status}`);
    return r.json();
  }

  function toast(text, kind, ms) {
    const wrap = $('#toast-wrap'); if (!wrap) return;
    const el = document.createElement('div');
    el.className = 'toast' + (kind ? ' ' + kind : '');
    el.textContent = text;
    wrap.appendChild(el);
    setTimeout(() => { el.classList.add('out'); setTimeout(() => el.remove(), 450); }, ms || 4200);
  }

  // ══════════════════════════ state machine ══════════════════════════
  function linkUp() { return wsAlive || restOk; }

  function recomputeState() {
    let s = 'idle';
    if (!linkUp()) s = 'offline';
    if (busy.speaking) s = 'speaking';
    if (busy.thinking) s = 'thinking';
    if (busy.killed) s = 'killed';
    if (appState === 'booting' && wsAlive) s = 'booting';
    appState = s;
    document.body.dataset.state = s;
    HUD.setState(s);
    const labels = { idle: 'ממתין', thinking: 'חושב', speaking: 'מדבר',
                     killed: 'KILL', offline: 'מנותק', booting: 'מאתחל' };
    const subs = { idle: 'STANDBY', thinking: 'REASONING LOOP', speaking: 'VOICE OUTPUT',
                   killed: 'SYSTEM HALTED', offline: 'NO LINK TO BRAIN', booting: 'POST SEQUENCE' };
    $('#reactor-state').textContent = labels[s] || s;
    $('#reactor-sub').textContent = subs[s] || '';

    const pill = $('#pill-brain');
    pill.className = 'pill ' + (linkUp() ? (busy.killed ? 'bad' : 'ok') : 'bad');
    pill.querySelector('b').textContent = !linkUp() ? 'מנותק'
        : busy.killed ? 'KILL' : (wsAlive ? 'פעיל · WS' : 'פעיל · REST');
  }

  function markBusy(agent, ms) {
    agentBusy[agent] = Date.now() + (ms || 2600);
    renderAgents();
  }
  function tickAgents() {
    const now = Date.now();
    let changed = false;
    for (const k in agentBusy) if (agentBusy[k] < now) { delete agentBusy[k]; changed = true; }
    if (changed) renderAgents();
  }

  // ══════════════════════════ rendering ══════════════════════════
  function renderAgents() {
    const ul = $('#agent-list'); if (!ul) return;
    const live = new Set(Object.keys(agentBusy));
    ul.innerHTML = AGENTS.map(a => {
      const on = a.id === 'jarvis' ? linkUp() && !busy.killed : linkUp();
      const cls = live.has(a.id) ? 'busy' : (on ? 'on' : '');
      return `<li class="${cls}"><i class="a-dot"></i><span class="a-name">${a.name}</span>` +
             `<span class="a-role">${a.role}</span></li>`;
    }).join('');
  }

  function kv(pairs) {
    return pairs.filter(p => p[1] !== undefined && p[1] !== null && p[1] !== '')
      .map(([k, v, hot]) => `<dt class="k">${esc(k)}</dt><dd class="v${hot ? ' hot' : ''}">${esc(v)}</dd>`).join('');
  }

  function renderStatus(st) {
    if (!st) return;
    const b = st.brain || {}, tr = b.train || {};
    $('#model-stats').innerHTML = kv([
      ['מצב', b.available ? 'טעון' : 'לא זמין', true],
      ['פרמטרים', b.params ? (b.params / 1e6).toFixed(2) + 'M' : '—'],
      ['שכבות', b.layers || '—'],
      ['ממד', b.d_model || b.dim || '—'],
      ['backend', b.backend || '—'],
      ['perplexity', tr.ppl != null ? Number(tr.ppl).toFixed(3) : '—', true],
      ['צעדי אימון', tr.steps || '—']
    ]);
    const load = Math.min(100, Math.round((b.available ? 70 : 0) + (tr.ppl ? Math.max(0, 30 - tr.ppl * 10) : 0)));
    $('#bar-load').style.width = load + '%';
    $('#model-hint').textContent = b.available
      ? `ליבה זמינה · ${st.skills || 0} כלים · ${(st.events || 0)} אירועים בזיכרון אפיק`
      : (b.warning || 'המשקולות אינן טעונות — JARVIS פועל על מנוע כללים + ידע מוצהר');

    const m = st.memory || {};
    $('#memory-stats').innerHTML = kv([
      ['אירועים', m.episodes || 0],
      ['עובדות', m.facts || 0],
      ['מיומנויות', m.skills || 0],
      ['שיחות', st.turns || 0]
    ]);

    renderStt(st.stt);
    const v = st.voice || {};
    const cov = v.coverage || {};
    $('#pill-voice').className = 'pill ' + (v.engine === 'concat' ? 'ok' : 'warn');
    $('#pill-voice').querySelector('b').textContent =
      v.engine ? `${v.engine} ${cov.phones_covered || 0}/${cov.phones_needed || 29}` : '—';

    const sec = st.security || {};
    busy.killed = !!sec.killed;
    $('#pill-security').className = 'pill ' + (sec.killed ? 'bad' : sec.dry_run ? 'warn' : 'ok');
    $('#pill-security').querySelector('b').textContent =
      sec.killed ? 'KILL' : `${(sec.level || '').toLowerCase()}${sec.dry_run ? '·dry' : ''}`;
    const lvl = $('#sel-level');
    if (lvl && sec.level && lvl.value !== String(sec.level).toLowerCase()) lvl.value = String(sec.level).toLowerCase();
    const dr = $('#chk-dryrun');
    if (dr && typeof sec.dry_run === 'boolean' && dr.checked !== sec.dry_run) dr.checked = sec.dry_run;

    if (st.theme && document.body.className.indexOf('theme-' + st.theme) < 0) {
      document.body.className = document.body.className.replace(/theme-\w+/, 'theme-' + st.theme);
    }
    $('#kill-overlay').classList.toggle('hidden', !busy.killed);
    renderAgents();
    recomputeState();
  }

  function renderStt(s) {
    if (!s) return;
    $('#stt-stats').innerHTML = kv([
      ['מצב', s.available ? 'זמין' : 'לא נבנה', true],
      ['פקודות', s.commands || 0],
      ['תבניות', s.templates || 0],
      ['הבחנה', s.calibration ? `${Number(s.calibration.self_ref || 0).toFixed(2)} / ${Number(s.calibration.cross_ref || 0).toFixed(2)}` : '—'],
      ['הקלטות משתמש', s.enrolled || 0]
    ]);
    $('#bar-stt').style.width = Math.min(100, (s.commands || 0) * 4) + '%';
    const vocab = s.commands_text || [];
    if (vocab.length) {
      $('#stt-vocab').innerHTML = vocab.slice(0, 14)
        .map(t => `<li><span class="t-ts">קול</span>${esc(t)}</li>`).join('');
    } else if (!s.available) {
      $('#stt-vocab').innerHTML = '<li>הבנק ייבנה אוטומטית בהפעלה הראשונה (~30 שניות)</li>';
    }
  }

  function renderTelemetry(t) {
    if (!t) return;
    const cpu = Number(t.cpu_percent != null ? t.cpu_percent : t.cpu || 0);
    const ram = Number(t.ram_percent != null ? t.ram_percent : t.memory || 0);
    const disk = Number(t.disk_percent != null ? t.disk_percent : t.disk || 0);
    const net = Number((t.net_kb_s != null ? t.net_kb_s : (t.net || 0)) / 100);
    HUD.setGauge('cpu', cpu); HUD.setGauge('ram', ram);
    HUD.setGauge('disk', disk); HUD.setGauge('net', Math.min(100, net));
    $('#g-cpu').textContent = cpu.toFixed(0) + '%';
    $('#g-ram').textContent = ram.toFixed(0) + '%';
    $('#g-disk').textContent = disk.toFixed(0) + '%';
    $('#g-net').textContent = (t.net_kb_s != null ? t.net_kb_s : t.net || 0).toFixed(0) + 'k';
    HUD.sparkPush('cpu', cpu);
    HUD.sparkPush('ram', ram);
  }

  function eventLevel(topic, data) {
    const t = String(topic || '');
    if (/error|fail|deny|kill|timeout|crash|blocked/i.test(t)) return 'bad';
    if (/permission|warn|repair|fallback|degrad/i.test(t)) return 'warn';
    if (/ready|answer|verified|speak\.end|boot/i.test(t)) return 'good';
    if (data && typeof data === 'object' && (data.ok === false)) return 'bad';
    return '';
  }

  function pushEvent(ev) {
    const ul = $('#event-stream'); if (!ul || !ev) return;
    const lvl = eventLevel(ev.topic, ev.data);
    const li = document.createElement('li');
    if (lvl) li.dataset.lvl = lvl;
    let d = ev.data;
    if (d && typeof d === 'object') { try { d = JSON.stringify(d); } catch (_) { d = String(d); } }
    li.innerHTML = `<span class="ev-t">${hhmmss(ev.ts || Date.now() / 1000)}</span>` +
                   `<span class="ev-topic">${esc(ev.topic)}</span>` +
                   `<span class="ev-data">${esc(String(d == null ? '' : d).slice(0, 220))}</span>`;
    ul.appendChild(li);
    while (ul.children.length > 260) ul.removeChild(ul.firstChild);
    ul.scrollTop = ul.scrollHeight;

    // agents light up when they work
    const topic = String(ev.topic || '');
    if (/^agent\./.test(topic)) markBusy(String((ev.data && ev.data.agent) || 'jarvis'));
    if (/hephaestus|coder/i.test(topic)) markBusy('hephaestus');
    if (/^memory\./.test(topic)) markBusy('mnemosyne');
    if (/^security\./.test(topic)) markBusy('argus');
    if (/^voice\./.test(topic)) markBusy('vox');
    if (/^brain\./.test(topic)) markBusy('nano');
  }

  // ───────────────────────── transcript ─────────────────────────
  function addMsg(kind, body, head, opts) {
    const wrap = $('#transcript'); if (!wrap) return null;
    const el = document.createElement('div');
    el.className = 'msg ' + kind;
    const chips = (opts && opts.chips || []).map(c => `<span class="chip ${c.k || ''}">${esc(c.t)}</span>`).join('');
    // Copy / re-speak / drop: a long answer is worth keeping, worth hearing
    // again and worth clearing off the glass — none of which should need a
    // round trip through the model.
    const acts = `<span class="m-acts">` +
      `<button class="ma" data-act="copy" title="העתק את התוכן">⧉</button>` +
      `<button class="ma" data-act="speak" title="הקרא שוב בקול">🔊</button>` +
      `<button class="ma" data-act="del" title="הסר מהמסך">✕</button>` +
      `</span>`;
    el.innerHTML =
      `<div class="m-head"><span>${esc(head || (kind === 'user' ? 'אדוני' : 'אדיאל'))}</span>` +
      `<span class="m-time">${new Date().toTimeString().slice(0, 8)}</span>${acts}</div>` +
      `<div class="m-body">${body}</div>` +
      (chips ? `<div class="m-meta">${chips}</div>` : '') +
      (opts && opts.trace ? `<div class="m-trace">${opts.trace}</div>` : '');
    // Follow the conversation only while the reader is already at the bottom;
    // yanking the scroll back mid-read makes history unusable.
    const stick = wrap.scrollHeight - wrap.scrollTop - wrap.clientHeight < 160;
    wrap.appendChild(el);
    while (wrap.children.length > 90) wrap.removeChild(wrap.firstChild);
    if (stick) wrap.scrollTop = wrap.scrollHeight;
    updateJump();
    return el;
  }

  function updateJump() {
    const t = $('#transcript'), b = $('#jump-latest');
    if (!t || !b) return;
    const far = t.scrollHeight - t.scrollTop - t.clientHeight > 160;
    b.classList.toggle('hidden', !far);
  }

  function addTyping() {
    const wrap = $('#transcript'); if (!wrap) return null;
    const el = document.createElement('div');
    el.className = 'msg bot';
    el.innerHTML = `<div class="m-head"><span>אדיאל</span><span class="m-time">מעבד…</span></div>` +
                   `<div class="m-body"><span class="typing"><i></i><i></i><i></i></span></div>`;
    const stick = wrap.scrollHeight - wrap.scrollTop - wrap.clientHeight < 160;
    wrap.appendChild(el);
    if (stick) wrap.scrollTop = wrap.scrollHeight;
    return el;
  }

  function traceHtml(tr) {
    if (!tr || typeof tr !== 'object') return '';
    const out = [];
    const line = (label, val, cls) => { if (val == null || val === '') return; out.push(`<b>${esc(label)}</b> ${cls ? `<span class="${cls}">` : ''}${esc(String(val)).slice(0, 900)}${cls ? '</span>' : ''}`); };
    if (tr.thought) line('מחשבה', tr.thought);
    if (tr.plan) line('תכנון', Array.isArray(tr.plan) ? tr.plan.join(' → ') : tr.plan);
    if (tr.intent) line('כוונה', tr.intent);
    if (tr.route) line('נתיב', typeof tr.route === 'object' ? JSON.stringify(tr.route) : tr.route);
    if (tr.tool) line('כלי', typeof tr.tool === 'object' ? JSON.stringify(tr.tool) : tr.tool);
    if (tr.tool_result != null) line('תוצאת כלי', typeof tr.tool_result === 'object' ? JSON.stringify(tr.tool_result) : tr.tool_result, 'code-out');
    if (tr.tools && tr.tools.length) tr.tools.forEach((t, i) => line(`כלי ${i + 1}`, typeof t === 'object' ? JSON.stringify(t) : t, 'code-out'));
    if (tr.verification) line('אימות', typeof tr.verification === 'object' ? JSON.stringify(tr.verification) : tr.verification);
    if (tr.memory) line('זיכרון', typeof tr.memory === 'object' ? JSON.stringify(tr.memory) : tr.memory);
    if (tr.knowledge) line('ידע', typeof tr.knowledge === 'object' ? JSON.stringify(tr.knowledge) : tr.knowledge);
    if (tr.code) line('קוד', '\n' + tr.code, 'code-out');
    if (tr.stdout) line('פלט הרצה', tr.stdout, 'code-out');
    if (tr.stderr) line('שגיאות', tr.stderr, 'code-err');
    if (tr.repairs) line('תיקונים עצמיים', typeof tr.repairs === 'object' ? JSON.stringify(tr.repairs) : tr.repairs);
    if (tr.agent) line('סוכן', tr.agent);
    if (tr.model_text) line('פלט הרשת', tr.model_text);
    // anything else
    const known = new Set(['thought','plan','intent','route','tool','tool_result','tools','verification',
                           'memory','knowledge','code','stdout','stderr','repairs','agent','model_text']);
    Object.keys(tr).forEach(k => { if (!known.has(k)) line(k, typeof tr[k] === 'object' ? JSON.stringify(tr[k]) : tr[k]); });
    return out.join('\n');
  }

  function countTurns(n) {
    const el = $('#th-count');
    if (!el) return;
    if (n === undefined) n = $$('#transcript .msg.bot, #transcript .msg.err').length;
    el.textContent = n ? `${n} תשובות · נשמר גם אחרי אתחול` : 'ריק';
  }

  function renderTurn(turn) {
    if (!turn) return;
    const a = turn.answer || {};
    const chips = [];
    if (a.intent) chips.push({ t: 'intent ' + a.intent, k: '' });
    if (a.skill) chips.push({ t: '⚙ ' + a.skill, k: 'skill' });
    if (a.agent) chips.push({ t: '🤖 ' + a.agent, k: 'tool' });
    chips.push({ t: a.grounded ? '✓ מעוגן' : '⚠ לא מעוגן', k: a.grounded ? 'skill' : 'tool' });
    if (a.risk && a.risk !== 'SAFE') chips.push({ t: 'סיכון ' + a.risk, k: 'tool' });
    if (a.confidence != null) chips.push({ t: 'ביטחון ' + Math.round(a.confidence * 100) + '%' });
    chips.push({ t: (turn.ms != null ? turn.ms : a.ms || 0).toFixed(0) + 'ms' });
    addMsg(a.grounded ? 'bot' : 'err', esc(a.text || '…'), turn.voice ? 'אדיאל · 🗣' : 'JARVIS',
           { chips, trace: traceHtml(a.trace) });
    countTurns();
  }

  // ───────────────────────── permissions ─────────────────────────
  const permCards = new Map();
  function renderPermission(req) {
    const box = $('#perm-queue'); if (!box || !req) return;
    if (permCards.has(req.id)) return;
    const empty = box.querySelector('.perm-empty'); if (empty) empty.remove();
    const el = document.createElement('div');
    el.className = 'perm-req';
    el.innerHTML =
      `<div class="p-skill">#${esc(req.id)} · ${esc(req.action || '?')} <span style="color:var(--red)">[${esc(req.level || 'CRITICAL')}]</span></div>` +
      `<div class="p-risk">JARVIS מבקש אישור לבצע פעולה ברמת סיכון גבוהה.</div>` +
      `<div class="p-args">${esc(JSON.stringify(req.args || {})).slice(0, 300)}</div>` +
      `<div class="p-btns"><button class="allow">אשר</button><button class="deny">דחה</button></div>`;
    el.querySelector('.allow').onclick = () => answerPerm(req.id, true);
    el.querySelector('.deny').onclick = () => answerPerm(req.id, false);
    box.appendChild(el);
    permCards.set(req.id, el);
    toast(`בקשת הרשאה #${req.id}: ${req.action}`, 'warn', 9000);
    markBusy('argus', 6000);
  }
  function dropPermission(id) {
    const el = permCards.get(id);
    if (el) { el.remove(); permCards.delete(id); }
    if (!permCards.size) {
      const box = $('#perm-queue');
      if (box && !box.querySelector('.perm-empty'))
        box.innerHTML = '<div class="perm-empty">אין בקשות ממתינות — המערכת במצב מאובטח</div>';
    }
  }
  async function answerPerm(id, allow) {
    dropPermission(id);
    try { send({ type: 'permission', id, allow }); } catch (_) {}
    toast(allow ? `אושרה בקשה #${id}` : `נדחתה בקשה #${id}`, allow ? 'good' : 'err');
  }
  async function syncPermissions() {
    try {
      const r = await api('/api/permissions');
      (r.pending || []).forEach(renderPermission);
      if (!(r.pending || []).length && !permCards.size) {
        $('#perm-queue').innerHTML = '<div class="perm-empty">אין בקשות ממתינות — המערכת במצב מאובטח</div>';
      }
    } catch (_) {}
  }

  // ───────────────────────── memory feed ─────────────────────────
  // ══════════════════════ firewall audit + skill catalogue ══════════════════
  // /api/audit, /api/skills and /api/stt were all implemented server-side and
  // none of them were ever called from the HUD — the permission firewall kept an
  // immutable log of every blocked action and the UI showed none of it. Surfacing
  // them here rather than adding new backend surface.
  const RISK_ORDER = { CRITICAL: 0, WRITE: 1, SAFE: 2 };
  let skillsFilter = null;
  let skillsCache = [];

  function renderAudit(a) {
    if (!a) return;
    const s = a.stats || {}, tail = a.tail || [];
    const el = $('#fw-stats'); if (!el) return;
    el.innerHTML = kv([
      ['רמת הרשאות', s.level || '—', true],
      ['מצב הדמיה', s.dry_run ? 'פעיל' : 'כבוי'],
      ['מתג חירום', s.killed ? 'מופעל' : 'תקין', !!s.killed],
      ['סה״כ החלטות', s.total || 0],
      ['אושרו', s.allowed || 0],
      ['נחסמו', s.blocked || 0, (s.blocked || 0) > 0]
    ]);
    const ul = $('#audit-stream');
    if (ul) {
      ul.innerHTML = tail.length ? tail.slice(0, 18).map(r => {
        const t = new Date((r.ts || 0) * 1000).toTimeString().slice(0, 8);
        const mark = r.allowed ? '✓' : '✕';
        return `<li class="${r.allowed ? '' : 'denied'}">`
             + `<span class="t-ts">${esc(t)}</span>`
             + `<b>${esc(mark)}</b> ${esc(r.action || '?')} `
             + `<span class="lvl-${esc(String(r.level || '').toLowerCase())}">${esc(r.level || '')}</span>`
             + `<div class="a-reason">${esc(r.reason || '')}</div></li>`;
      }).join('') : '<li>אין עדיין רישומים ביומן.</li>';
    }
  }

  async function refreshAudit() {
    try { renderAudit(await api('/api/audit?n=40')); }
    catch (_) { const el = $('#fw-stats'); if (el) el.innerHTML = kv([['מצב', 'לא זמין']]); }
  }

  // ── the sentinel: sequence shapes the per-call firewall cannot see ────────
  function renderSentinel(s) {
    if (!s) return;
    const st = s.stats || {}, tail = s.tail || [];
    const tag = $('#sentinel-state');
    if (tag) {
      const cooling = !!st.cooldown_active;
      tag.textContent = cooling ? 'מגיב' : (st.acting ? 'חמוש' : 'צופה');
      tag.style.color = cooling ? 'var(--red)' : (st.alerts ? 'var(--cyan)' : '');
    }
    const el = $('#sentinel-stats');
    if (el) el.innerHTML = kv([
      ['התראות', st.alerts || 0, (st.alerts || 0) > 0],
      ['פעולות לא־בטוחות בדקה', st.recent_non_safe || 0, (st.recent_non_safe || 0) >= 4],
      ['סירובים ב־2 דק׳', st.recent_denials || 0, (st.recent_denials || 0) >= 3],
      ['הגינה אוטומטית', st.acting ? 'חמושה' : 'כבויה', !!st.acting],
      ['הרשאות בהקפאה', coolingTag(st), coolingTag(st) !== 'לא']
    ]);
    const ul = $('#sentinel-stream');
    if (ul) ul.innerHTML = tail.length
      ? tail.slice().reverse().map(a =>
          `<li><span class="t-ts">${a.ts ? hhmmss(a.ts) : '•'}</span>`
          + `<b>${esc(a.kind)}</b> ${esc(JSON.stringify(a.detail || {}).slice(1, 90))}`
          + (a.acted ? ` <span class="lvl-critical">הגיב</span>` : '') + `</li>`).join('')
      : '<li>הרצף תקין — אין צורות חריגות.</li>';
  }
  function coolingTag(st) { return st && st.cooldown_active ? 'כן' : 'לא'; }

  async function refreshSentinel() {
    try { renderSentinel(await api('/api/sentinel?n=12')); }
    catch (_) { const el = $('#sentinel-stats'); if (el) el.innerHTML = kv([['מצב', 'לא זמין']]); }
  }

  function renderSkills(r) {
    if (!r) return;
    skillsCache = r.skills || [];
    const el = $('#skills-stats'); if (!el) return;
    const by = { SAFE: 0, WRITE: 0, CRITICAL: 0 };
    skillsCache.forEach(s => { by[s.risk] = (by[s.risk] || 0) + 1; });
    el.innerHTML = kv([
      ['סה״כ כלים', r.count || skillsCache.length, true],
      ['SAFE', by.SAFE || 0],
      ['WRITE', by.WRITE || 0],
      ['CRITICAL', by.CRITICAL || 0, (by.CRITICAL || 0) > 0]
    ]);
    const ul = $('#skills-list'); if (!ul) return;
    const rows = skillsCache
      .filter(s => !skillsFilter || s.risk === skillsFilter)
      .slice()
      .sort((a, b) => (RISK_ORDER[a.risk] ?? 9) - (RISK_ORDER[b.risk] ?? 9)
                   || String(a.name).localeCompare(String(b.name)));
    ul.innerHTML = rows.length ? rows.map(s =>
        `<li data-risk="${esc(s.risk)}" title="${esc(s.description || '')}">`
      + `<span class="t-ts">${esc(s.agent || '')}</span>`
      + `<b>${esc(s.name)}</b> <span class="lvl-${esc(String(s.risk || '').toLowerCase())}">${esc(s.risk)}</span>`
      + `<div class="a-reason">${esc(s.description || '')}</div></li>`).join('')
      : '<li>אין כלים ברמה הזו.</li>';
  }

  async function refreshSkills() {
    try { renderSkills(await api('/api/skills')); }
    catch (_) { const el = $('#skills-stats'); if (el) el.innerHTML = kv([['מצב', 'לא זמין']]); }
  }

  // ── screen analysis ─────────────────────────────────────────────────────
  // Drives /api/screen/read, which decodes a screenshot with vision.png.
  // `capture` asks the backend to photograph the screen first; that step is
  // WRITE-risk and still passes through the firewall, so it can come back
  // blocked or waiting on a confirmation — both are shown, not swallowed.
  function pct(x) { return (Math.round((Number(x) || 0) * 100)) + '%'; }

  function renderScreenRead(r) {
    const sum = $('#sr-summary'); if (!sum) return;
    sum.classList.remove('err');
    const stats = $('#sr-stats'), note = $('#sr-note');

    if (!r || !r.ok) {
      const msg = (r && (r.error || r.value)) || 'לא הצלחתי לקרוא את צילום המסך.';
      sum.textContent = msg;
      sum.classList.add('err');
      if (stats) stats.innerHTML = kv([['שלב שנכשל', (r && r.stage) || '—']]);
      if (note) note.textContent = 'קריאת המסך נכשלה — שום ניתוח לא בוצע.';
      ['#bar-bright', '#bar-dark', '#bar-edge'].forEach(id => {
        const b = $(id); if (b) b.style.width = '0%';
      });
      ['#val-bright', '#val-dark', '#val-edge'].forEach(id => {
        const v = $(id); if (v) v.textContent = '—';
      });
      return;
    }

    const d = r.data || {};
    sum.textContent = d.summary_he || r.value || '—';
    if (stats) stats.innerHTML = kv([
      ['קובץ', d.path ? String(d.path).split(/[\\/]/).pop() : '—', true],
      ['מימדים', `${d.width ?? '?'}×${d.height ?? '?'}`],
      ['סוג צבע', d.color_name || '—'],
      ['עומק סיביות', d.bit_depth ? `${d.bit_depth}-bit` : '—'],
      ['שזור', d.interlaced ? 'כן (Adam7)' : 'לא'],
      ['גודל', d.bytes ? `${(d.bytes / 1024).toFixed(1)} KB` : '—'],
      ['אור ממוצע', d.brightness_mean != null ? Number(d.brightness_mean).toFixed(2) : '—']
    ]);

    const bright = Number(d.bright_share) || 0, dark = Number(d.dark_share) || 0;
    // edge energy is unbounded in principle; 0.12 reads as a busy document, so
    // scaling against that keeps the bar meaningful instead of pinned at 100%.
    const edge = Math.min(1, (Number(d.edge_energy) || 0) / 0.12);
    const setBar = (barId, valId, frac, label) => {
      const b = $(barId); if (b) b.style.width = Math.round(frac * 100) + '%';
      const v = $(valId); if (v) v.textContent = label;
    };
    setBar('#bar-bright', '#val-bright', bright, pct(bright));
    setBar('#bar-dark', '#val-dark', dark, pct(dark));
    setBar('#bar-edge', '#val-edge', edge, Number(d.edge_energy || 0).toFixed(3));

    // Always say what this is not. The panel reports pixel statistics; it does
    // not read text, and leaving that unstated would imply a capability the
    // offline build does not have.
    if (note) note.textContent = d.text_extracted === false
      ? 'סטטיסטיקת פיקסלים בלבד — אין OCR במצב מקומי, ולכן לא מוצע טקסט מהמסך. '
        + 'פענוח ה־PNG נעשה בקודק שנכתב כאן (zlib + numpy), בלי Pillow ובלי OpenCV.'
      : '';
  }

  async function runScreenRead(capture) {
    const btnR = $('#btn-screen-read'), btnC = $('#btn-screen-capture');
    const btn = capture ? btnC : btnR;
    const sum = $('#sr-summary');
    [btnR, btnC].forEach(b => { if (b) b.disabled = true; });
    if (sum) { sum.classList.remove('err'); sum.textContent = capture ? 'מצלם וקורא…' : 'קורא…'; }
    try {
      renderScreenRead(await api(capture ? '/api/screen/read?capture=1' : '/api/screen/read'));
    } catch (e) {
      renderScreenRead({ ok: false, stage: capture ? 'capture' : 'read',
                         error: 'השרת לא זמין לקריאת המסך: ' + e.message });
    } finally {
      [btnR, btnC].forEach(b => { if (b) b.disabled = false; });
    }
  }

  function wireScreenRead() {
    const br = $('#btn-screen-read'); if (br) br.addEventListener('click', () => runScreenRead(false));
    const bc = $('#btn-screen-capture'); if (bc) bc.addEventListener('click', () => runScreenRead(true));
    const note = $('#sr-note');
    // State the limitation up front, before anyone presses a button and wonders
    // why there is no text in the result.
    if (note) note.textContent = 'לוחץ "קרא אחרון" מנתח את הצילום השמור האחרון; '
      + '"צלם ונתח" מצלם עכשיו דרך חומת האש ואז מנתח.';
  }

  // ── operational modes ────────────────────────────────────────────────────
  // Twelve postures from brain/modes.py. Switching one changes what JARVIS
  // reaches for first; it grants no new power — the firewall still grades every
  // action at its own risk level, and the panel says so.
  let modesCache = [];

  function renderModes(r) {
    if (!r) return;
    modesCache = r.modes || [];
    const cur = modesCache.find(m => m.active) || {};
    const nm = $('#mode-name'); if (nm) nm.textContent = cur.he || '—';
    const en = $('#mode-en'); if (en) en.textContent = cur.en || '—';
    const ds = $('#mode-desc'); if (ds) ds.textContent = cur.desc || '';
    const wrap = $('#mode-chips'); if (!wrap) return;
    wrap.innerHTML = modesCache.map(m =>
      `<button class="chip${m.active ? ' on' : ''}" data-mid="${esc(m.id)}" `
      + `title="${esc(m.desc)}">${esc(m.he)}</button>`).join('');
  }

  async function refreshModes() {
    try { renderModes(await api('/api/modes')); }
    catch (_) { const nm = $('#mode-name'); if (nm) nm.textContent = 'לא זמין'; }
  }

  async function setMode(mid) {
    try {
      const r = await fetch(HTTP + '/api/mode', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: mid }) });
      const d = await r.json();
      if (!r.ok || !d.ok) { toast(d && d.error ? d.error : 'החלפת מצב נכשלה', 'warn', 5000); return; }
      toast(`מצב: ${d.he} — ${d.desc}`, 'good', 5000);
      renderModes(await api('/api/modes'));
    } catch (e) { toast('החלפת מצב נכשלה: ' + e.message, 'warn', 5000); }
  }

  async function runResearch() {
    const inp = $('#research-topic'), out = $('#research-out');
    const topic = (inp && inp.value || '').trim();
    if (!topic) { toast('מה לחקור, אדוני?', 'warn', 4000); return; }
    if (out) { out.classList.remove('err'); out.textContent = 'חוקר בכל המקורות המקומיים…'; }
    try {
      // Same dispatcher the chat uses, so the research runs under the identical
      // firewall and trace as a spoken request would.
      const d = await post({ type: 'invoke', skill: 'research.query', args: { topic } });
      if (out) {
        out.classList.toggle('err', !(d && d.ok));
        out.textContent = (d && (d.value || d.error)) || 'לא התקבלה תוצאה.';
      }
    } catch (e) {
      if (out) { out.classList.add('err'); out.textContent = 'החקירה נכשלה: ' + e.message; }
    }
  }

  function wireModes() {
    const wrap = $('#mode-chips');
    if (wrap) wrap.addEventListener('click', e => {
      const b = e.target.closest('.chip'); if (b && b.dataset.mid) setMode(b.dataset.mid);
    });
    const br = $('#btn-mode-research'); if (br) br.addEventListener('click', runResearch);
    const inp = $('#research-topic');
    if (inp) inp.addEventListener('keydown', e => { if (e.key === 'Enter') runResearch(); });
    refreshModes();
    setInterval(refreshModes, 15000);
  }

  // ── command palette ───────────────────────────────────────────────────────
  // Sixty tools and twelve postures are a lot to hide behind panels. Ctrl+K
  // puts all of them one fuzzy match away. Selection goes through the very same
  // dispatcher as chat and voice, so a non-SAFE pick still meets the firewall and
  // its confirmation prompt — the palette is a faster hand, not a spare key.
  let paletteIdx = 0, paletteRows = [];

  async function paletteIndex() {
    const rows = [];
    let sk = skillsCache;
    if (!sk || !sk.length) {
      try { sk = (await api('/api/skills')).skills || []; skillsCache = sk; } catch (_) { sk = []; }
    }
    (sk || []).forEach(s => rows.push({
      kind: 'skill', id: s.name, label: s.name, sub: s.description || '',
      tag: s.risk || 'SAFE', hay: `${s.name} ${s.description || ''}`.toLowerCase() }));
    let md = modesCache;
    if (!md || !md.length) {
      try { md = (await api('/api/modes')).modes || []; modesCache = md; } catch (_) { md = []; }
    }
    (md || []).forEach(m => rows.push({
      kind: 'mode', id: m.id, label: `מצב ${m.he}`, sub: m.desc || '',
      tag: m.active ? 'פעיל' : 'MODE', hay: `מצב ${m.he} ${m.id} ${m.en} ${m.desc || ''}`.toLowerCase() }));
    return rows;
  }

  const RECENT_KEY = 'jarvis.palette.recent';
  function paletteRecents() {
    try { return JSON.parse(localStorage.getItem(RECENT_KEY) || '[]'); } catch (_) { return []; }
  }
  function paletteNoteRecent(id) {
    try {
      const list = [id].concat(paletteRecents().filter(x => x !== id)).slice(0, 6);
      localStorage.setItem(RECENT_KEY, JSON.stringify(list));
    } catch (_) {}
  }

  function paletteMatch(rows, q) {
    const t = (q || '').trim().toLowerCase();
    if (!t) {
      // Nothing typed yet: lead with what this HUD actually uses.
      const rec = paletteRecents();
      return rows.slice().sort((a, b) => rec.indexOf(b.id) - rec.indexOf(a.id)).slice(0, 40);
    }
    const toks = t.split(/\s+/);
    return rows.map(r => {
      let s = 0, i = -1;
      for (const tok of toks) {
        const at = r.hay.indexOf(tok);
        if (at < 0) return null;
        s += tok.length * (at === 0 ? 2 : 1);
        if (i < 0 || at < i) i = at;
      }
      return { r, s };
    }).filter(Boolean).sort((a, b) => b.s - a.s || a.r.hay.length - b.r.hay.length)
      .slice(0, 40).map(x => x.r);
  }

  async function renderPalette() {
    const q = $('#palette-q'); if (!q) return;
    const rows = paletteMatch(await paletteIndex(), q.value);
    paletteRows = rows;
    paletteIdx = Math.min(paletteIdx, Math.max(0, rows.length - 1));
    const ul = $('#palette-list'); if (!ul) return;
    ul.innerHTML = rows.length ? rows.map((r, i) =>
      `<li data-i="${i}" class="${i === paletteIdx ? 'sel' : ''}">`
      + `<b>${esc(r.label)}</b><span>${esc(String(r.sub).slice(0, 70))}</span>`
      + `<span class="pk lvl-${esc(String(r.tag).toLowerCase())}">${esc(r.tag)}</span></li>`).join('')
      : '<li><b>לא נמצא דבר</b><span>נסה מילה אחרת — הכלים, המצבים והפקודות כולם כאן</span></li>';
    const c = $('#palette-count'); if (c) c.textContent = `${rows.length} תוצאות`;
    ul.querySelectorAll('li').forEach(li => li.addEventListener('click', () => {
      paletteIdx = parseInt(li.dataset.i || '0', 10); paletteRun();
    }));
  }

  async function paletteRun() {
    const r = paletteRows[paletteIdx];
    closePalette();
    if (!r) return;
    paletteNoteRecent(r.id);
    if (r.kind === 'mode') { setMode(r.id); return; }
    toast(`מפעיל ${r.label} דרך חומת האש…`, 'good', 3000);
    try {
      const d = await post({ type: 'invoke', skill: r.id, args: {} });
      if (d && d.ok) toast(d.value ? String(d.value).slice(0, 140) : `${r.label}: בוצע`, 'good', 6000);
      else if (d) toast(d.error || d.message || `${r.label}: נחסם או נכשל`, 'warn', 7000);
    } catch (e) { toast(`ההפעלה נכשלה: ${e.message}`, 'err', 6000); }
  }

  function openPalette() {
    const p = $('#palette'); if (!p) return;
    p.classList.remove('hidden');
    paletteIdx = 0;
    const q = $('#palette-q'); if (q) { q.value = ''; q.focus(); }
    renderPalette();
  }
  function closePalette() { const p = $('#palette'); if (p) p.classList.add('hidden'); }

  // ── transcript actions ──────────────────────────────────────────────────
  function wireTranscript() {
    const t = $('#transcript'); if (!t) return;
    t.addEventListener('scroll', updateJump);
    t.addEventListener('click', async e => {
      const btn = e.target.closest('.ma'); if (!btn) return;
      const msg = btn.closest('.msg'); if (!msg) return;
      const body = msg.querySelector('.m-body');
      const text = body ? body.innerText.trim() : '';
      const act = btn.dataset.act;
      if (act === 'del') { msg.remove(); countTurns(); updateJump(); return; }
      if (!text) return;
      if (act === 'copy') {
        try {
          await navigator.clipboard.writeText(text);
          toast('הועתק ללוח', 'good', 2200);
        } catch (_) { toast('הדפדפן לא אישר העתקה', 'warn', 4000); }
        return;
      }
      if (act === 'speak') { send({ type: 'speak', text }); return; }
    });
    const b = $('#jump-latest');
    if (b) b.addEventListener('click', () => {
      const el = $('#transcript');
      if (el) el.scrollTop = el.scrollHeight;
      updateJump();
    });
  }

  // ── collapsible panels ──────────────────────────────────────────────────
  // Nine panels on two rails is more instrumentation than any single moment
  // needs. Folding one away is reversible and remembered per HUD, so the layout
  // a person builds survives a reload.
  const PANEL_KEY = 'jarvis.panels.collapsed';

  function readCollapsed() {
    try { return JSON.parse(localStorage.getItem(PANEL_KEY) || '[]'); } catch (_) { return []; }
  }
  function writeCollapsed(list) {
    try { localStorage.setItem(PANEL_KEY, JSON.stringify(list)); } catch (_) {}
  }

  function wirePanels() {
    $$('#grid .panel > header').forEach(h => {
      h.classList.add('collapsible');
      h.title = 'לחיצה מקפלת או פותחת את הפאנל';
    });
    const folded = readCollapsed();
    $$('#grid .panel').forEach(p => { if (p.id && folded.indexOf(p.id) >= 0) p.classList.add('collapsed'); });
    document.addEventListener('click', e => {
      const hdr = e.target.closest('#grid .panel > header'); if (!hdr) return;
      if (e.target.closest('button, select, input, label, a')) return;
      const p = hdr.parentElement; if (!p) return;
      p.classList.toggle('collapsed');
      const list = readCollapsed().filter(x => x !== p.id);
      if (p.classList.contains('collapsed')) list.push(p.id);
      writeCollapsed(list);
    });
  }

  function wirePalette() {
    document.addEventListener('keydown', e => {
      if ((e.ctrlKey || e.metaKey) && String(e.key).toLowerCase() === 'k') {
        e.preventDefault();
        const p = $('#palette');
        if (p && !p.classList.contains('hidden')) closePalette(); else openPalette();
        return;
      }
      const p = $('#palette');
      if (!p || p.classList.contains('hidden')) return;
      if (e.key === 'Escape') { closePalette(); return; }
      if (e.key === 'ArrowDown') { e.preventDefault(); paletteIdx = Math.min(paletteRows.length - 1, paletteIdx + 1); renderPalette(); }
      if (e.key === 'ArrowUp') { e.preventDefault(); paletteIdx = Math.max(0, paletteIdx - 1); renderPalette(); }
      if (e.key === 'Enter') { e.preventDefault(); paletteRun(); }
    });
    const q = $('#palette-q'); if (q) q.addEventListener('input', () => { paletteIdx = 0; renderPalette(); });
    const p = $('#palette');
    if (p) p.addEventListener('mousedown', e => { if (e.target === p) closePalette(); });
  }

  function wireSecurityPanels() {
    const ba = $('#btn-refresh-audit'); if (ba) ba.addEventListener('click', refreshAudit);
    const bs = $('#btn-refresh-skills'); if (bs) bs.addEventListener('click', refreshSkills);
    // clicking the stats row cycles the risk filter on the catalogue
    const ss = $('#skills-stats');
    if (ss) ss.addEventListener('click', () => {
      skillsFilter = skillsFilter === null ? 'CRITICAL'
                   : skillsFilter === 'CRITICAL' ? 'WRITE'
                   : skillsFilter === 'WRITE' ? 'SAFE' : null;
      renderSkills({ count: skillsCache.length, skills: skillsCache });
      toast(skillsFilter ? `מסנן: ${skillsFilter}` : 'המסנן הוסר', 'good', 1800);
    });
    refreshAudit(); refreshSkills(); refreshSentinel();
    setInterval(refreshAudit, 9000);
    setInterval(refreshSentinel, 6000);
  }

  async function syncMemory() {
    try {
      const r = await api('/api/memory');
      const ul = $('#memory-list'); if (!ul) return;
      const items = [];
      (r.recent || []).slice(-9).reverse().forEach(e => {
        const txt = typeof e === 'string' ? e : (e.summary || e.text || JSON.stringify(e));
        items.push(`<li><span class="t-ts">${e.ts ? hhmmss(e.ts) : '•'}</span>${esc(String(txt)).slice(0, 130)}</li>`);
      });
      (r.facts || []).slice(0, 4).forEach(f => {
        const txt = typeof f === 'string' ? f : `${f.subject || f.key || ''} — ${f.value || ''}`;
        items.push(`<li><span class="t-ts">fact</span>${esc(txt).slice(0, 130)}</li>`);
      });
      ul.innerHTML = items.join('') || '<li>הארמון ריק — הזיכרון נבנה תוך כדי שיחה</li>';
      if (r.stats) $('#memory-stats').innerHTML = kv([
        ['אירועים', r.stats.episodes || 0], ['עובדות', r.stats.facts || 0], ['מיומנויות', r.stats.skills || 0]]);
    } catch (_) {}
  }

  // ══════════════════════ local-file retrieval (RAG) ══════════════════════
  // The panel's whole job is to make the evidence visible. Every answer is
  // rendered with its citations as separate, clickable rows, and the verdict the
  // server computed (did each quote really appear in the file it names?) is
  // shown rather than trusted silently.
  async function syncRag() {
    try {
      const r = await api('/api/rag/status');
      const s = r.status || {};
      const el = $('#rag-stats');
      if (el) el.innerHTML = kv([
        ['קבצים', s.docs || 0], ['קטעים', s.chunks || 0],
        ['מונחים', s.distinct_terms || 0], ['נפח', fmtBytes(s.indexed_bytes || 0)],
      ]);
      const roots = $('#rag-roots');
      if (roots && !roots.value && (s.roots || []).length) roots.value = s.roots.join(', ');
      if (!(s.docs > 0)) {
        const a = $('#rag-answer');
        if (a && !a.dataset.userSet) {
          a.innerHTML = '<span class="rag-none">עדיין לא אינדקסתי אף קובץ. '
            + 'הכנס תיקייה ולחץ «אנדקס».</span>';
        }
      }
    } catch (_) {}
  }

  function fmtBytes(n) {
    if (!n) return '0B';
    const u = ['B', 'KB', 'MB', 'GB']; let i = 0;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return (i === 0 ? n : n.toFixed(1)) + u[i];
  }

  function renderRagAnswer(r) {
    const a = $('#rag-answer'), ul = $('#rag-cites');
    if (!a) return;
    a.dataset.userSet = '1';
    if (!r || r.answer_type === 'none' || !r.grounded) {
      a.innerHTML = '<span class="rag-none">' + esc(r && r.text ? r.text : 'לא מצאתי תשובה בקבצים.')
        + '</span>';
      if (ul) ul.innerHTML = '';
      return;
    }
    const badge = r.answer_type === 'grounded'
      ? '<span class="rag-badge ok">מעוגן</span>'
      : '<span class="rag-badge weak">חלש</span>';
    const verdict = r.verified === false
      ? '<span class="rag-badge bad">הציטוט לא אומת</span>' : '';
    a.innerHTML = badge + verdict
      + '<span class="rag-conf">ביטחון ' + Math.round((r.confidence || 0) * 100) + '%</span>'
      + '<div class="rag-body">' + esc(r.text || '').replace(/\n/g, '<br>') + '</div>';
    if (ul) {
      ul.innerHTML = (r.citations || []).map(c =>
        '<li><span class="t-ts">' + esc(c.start_line) + '–' + esc(c.end_line) + '</span>'
        + esc(c.file) + ' · ' + esc((c.quote || '').replace(/\s+/g, ' ').slice(0, 90)) + '</li>'
      ).join('') || '<li>בלי ציטוטים</li>';
    }
  }

  async function ragAsk() {
    const q = $('#rag-query');
    const query = q && q.value.trim();
    if (!query) { toast('מה לחפש בקבצים?', 'warn'); return; }
    const a = $('#rag-answer');
    if (a) { a.dataset.userSet = '1'; a.innerHTML = '<span class="rag-none">מחפש בקבצים…</span>'; }
    try {
      const r = await api('/api/rag/ask', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, k: 6 }),
      });
      renderRagAnswer(r);
      if (r && r.verified === false) toast('הציטוט לא אומת מול הקובץ', 'err', 6000);
    } catch (e) {
      if (a) a.innerHTML = '<span class="rag-none">החיפוש נכשל: ' + esc(String(e.message || e)) + '</span>';
    }
  }

  async function ragIndex() {
    const field = $('#rag-roots');
    const roots = (field && field.value || '').split(/[,\n]/).map(s => s.trim()).filter(Boolean);
    if (!roots.length) { toast('אין תיקייה לאינדוקס', 'warn'); return; }
    const a = $('#rag-answer');
    if (a) { a.dataset.userSet = '1'; a.innerHTML = '<span class="rag-none">סורק את התיקייה…</span>'; }
    try {
      const r = await api('/api/rag/index', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ roots }),
      });
      if (r && r.ok === false) {
        if (a) a.innerHTML = '<span class="rag-none">' + esc(r.error || 'האינדוקס נחסם') + '</span>';
        toast(r.error || 'האינדוקס נחסם', 'err', 6000);
        return;
      }
      const secs = (r && r.seconds) || 0;
      if (a) a.innerHTML = '<span class="rag-ok">אינדקסתי ' + (r.added + r.updated || 0)
        + ' קבצים · ' + (r.chunks || 0) + ' קטעים · ' + secs + ' שניות'
        + ((r.skipped ? ' · דילגתי על ' + r.skipped : '') + '</span>');
      toast('האינדקס מוכן', 'ok');
      syncRag();
    } catch (e) {
      if (a) a.innerHTML = '<span class="rag-none">האינדוקס נכשל: ' + esc(String(e.message || e)) + '</span>';
    }
  }

  function wireRag() {
    const ask = $('#rag-ask-btn'), idx = $('#rag-index-btn');
    if (ask) ask.addEventListener('click', ragAsk);
    if (idx) idx.addEventListener('click', ragIndex);
    const q = $('#rag-query');
    if (q) q.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); ragAsk(); } });
    const r = $('#rag-roots');
    if (r) r.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); ragIndex(); } });
  }

  // ══════════════════════════ boot sequence ══════════════════════════
  // If the measurement sequence is still running after this long, open the
  // interface anyway and say so. A slow machine must never read as a dead one.
  const BOOT_WATCHDOG_MS = 40000;

  // Skip is a control, not decoration inside the ceremony. The overlay is on
  // screen from the first paint, but the measurements only begin once the brain
  // answers — on a cold machine that wait runs to a minute and a half. A button
  // wired at the *end* of that wait is dead exactly when it is wanted, which is
  // what made it look broken. So it is wired before anything else, it acts on
  // shared state, and it works whether or not the boot sequence has started.
  let bootSkipped = false;

  function skipBoot() {
    if (bootSkipped) return;
    bootSkipped = true;
    const overlay = $('#boot-overlay');
    if (overlay) overlay.classList.add('hidden');
    appState = 'idle';
    recomputeState();
    toast('הממשק נפתח — המדידות ממשיכות ברקע', 'good', 5000);
  }

  function wireBootSkip() {
    const b = $('#boot-skip');
    if (b && !b.dataset.wired) { b.dataset.wired = '1'; b.onclick = () => skipBoot(); }
  }

  // Wired the moment this script is parsed — the overlay is already visible and
  // the brain may be a minute away, so waiting for init() would leave the button
  // dead for exactly the stretch where it matters.
  wireBootSkip();
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', wireBootSkip);
  }

  async function runBoot() {
    const overlay = $('#boot-overlay'), log = $('#boot-log'), fill = $('#boot-bar-fill');
    wireBootSkip();
    if (bootSkipped) {
      // Already opened by hand: measure in the background and report failures as
      // toasts, never as a black screen.
      overlay.classList.add('hidden');
      api('/api/boot').then(rep => {
        const bad = ((rep && rep.report) || []).filter(r => !r.ok);
        if (bad.length) toast(`${bad.length} תתי־מערכת לא תקינות: ` + bad.map(f => f.key).join(', '), 'warn', 9000);
      }).catch(() => {});
      return;
    }
    overlay.classList.remove('hidden');
    log.innerHTML = '';
    const typeLine = (text, cls) => new Promise(res => {
      const line = document.createElement('div');
      if (cls) line.className = cls;
      log.appendChild(line);
      if (bootSkipped) { line.textContent = text; res(); return; }
      let i = 0;
      const step = () => {
        line.textContent = text.slice(0, ++i);
        if (i < text.length) setTimeout(step, 7);
        else { log.scrollTop = log.scrollHeight; res(); }
      };
      step();
    });

    await typeLine('אדיאל — POST SEQUENCE', 'dim');
    await typeLine('Mark VII · offline · zero cloud · zero API keys', 'dim');
    await new Promise(r => setTimeout(r, 180));

    // Kick the measurement off in the background; results stream in through the
    // progress endpoint so the bar fills per completed check instead of the
    // overlay sitting empty until the whole (minutes-long) POST returns.
    const bootPromise = api('/api/boot')
      .catch(err => ({ ok: false, error: String((err && err.message) || err) }));

    const started = Date.now();
    let seen = 0, done = false, polled = 0, progressUsable = false, watchdogFired = false;
    // The operator outranks the ceremony: skipping reveals the interface now and
    // leaves the measurements running in the background.
    wireBootSkip();
    const typeItem = r => typeLine(
      `${r.ok ? '[ OK ]' : '[FAIL]'}  ${String(r.key).padEnd(20, '.')}  ${r.label} · ${String(r.detail).slice(0, 90)} (${r.ms}ms)`,
      r.ok ? 'ok' : 'warn');

    while (!done && !bootSkipped) {
      let prog = null;
      try { prog = await api('/api/boot/progress'); progressUsable = true; }
      catch (_) { prog = null; }
      polled++;

      if (prog) {
        const items = prog.items || [];
        for (let i = seen; i < items.length; i++) {
          await typeItem(items[i]);
          if (!items[i].ok) toast(`${items[i].key}: ${items[i].detail}`, 'warn', 7000);
        }
        seen = items.length;
        const total = Math.max(1, prog.total || items.length || 15);
        fill.style.width = Math.round((seen / total) * 100) + '%';
        done = !!prog.done;
      }

      // A server without the progress endpoint: fall back to the single
      // response rather than polling forever.
      if (!progressUsable && polled >= 3) {
        const rep = await bootPromise;
        const report = (rep && rep.report) || [];
        for (let i = seen; i < report.length; i++) {
          await typeItem(report[i]);
          if (!report[i].ok) toast(`${report[i].key}: ${report[i].detail}`, 'warn', 7000);
        }
        seen = report.length;
        fill.style.width = '100%';
        if (!report.length) {
          await typeLine('✗ no link to the brain: ' + ((rep && rep.error) || 'unknown'), 'warn');
        }
        done = true;
        break;
      }

      if (!done && bootSkipped) {
        watchdogFired = true;
        await typeLine(`— ${seen} מדידות הושלמו; הממשק נפתח לפי בקשתך, השאר רץ ברקע —`, 'warn');
        break;
      }
      if (!done && Date.now() - started > BOOT_WATCHDOG_MS) {
        watchdogFired = true;
        await typeLine(`— המדידות עדיין רצות ברקע (${seen} הושלמו); פותח את הממשק עכשיו —`, 'warn');
        break;
      }
      if (!done) await new Promise(r => setTimeout(r, 350));
    }

    if (watchdogFired) {
      // Keep watching after the reveal so a late failure is still reported, just
      // never again at the cost of a black screen.
      (async () => {
        try {
          const rep = await bootPromise;
          const failed = ((rep && rep.report) || []).filter(r => !r.ok);
          if (failed.length) {
            toast(`${failed.length} תתי־מערכת לא תקינות: ` + failed.map(f => f.key).join(', '), 'warn', 9000);
          }
        } catch (_) {}
      })();
    } else {
      const failed = seen; // failures were already toasted as they arrived
      await new Promise(r => setTimeout(r, 140));
      await typeLine(failed ? `— ${failed} subsystem(s) degraded; JARVIS continues with fallbacks —`
                            : '— כל המערכות תקינות. שלום לך, אדוני. —', failed ? 'warn' : 'ok');
      await new Promise(r => setTimeout(r, 420));
    }
    overlay.classList.add('hidden');
    appState = 'idle';
    recomputeState();
  }

  // ══════════════════════════ websocket ══════════════════════════
  function send(obj) {
    if (ws && ws.readyState === 1) { ws.send(JSON.stringify(obj)); return true; }
    // No socket (some proxies refuse to upgrade) — the REST twin of the WS runs
    // the very same dispatcher on the server, so behaviour is identical.
    post(obj).then(msg => { if (msg) handle(msg); })
             .catch(e => toast('הפקודה נכשלה: ' + e.message, 'err'));
    return true;
  }

  async function post(obj) {
    const r = await fetch(HTTP + '/api/command', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(obj)
    });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    restOk = true;
    recomputeState();
    return r.json();
  }

  function connect() {
    try { ws = new WebSocket(WS); } catch (e) { return scheduleReconnect(); }
    ws.onopen = () => {
      wsAlive = true; retry = 600; recomputeState();
      pushEvent({ topic: 'ui.link', data: 'connected to ' + WS, ts: Date.now() / 1000 });
      syncPermissions(); syncMemory(); syncRag();
    };
    ws.onclose = () => { wsAlive = false; recomputeState(); scheduleReconnect(); };
    ws.onerror = () => { wsAlive = false; recomputeState(); };
    ws.onmessage = m => {
      let msg; try { msg = JSON.parse(m.data); } catch (_) { return; }
      handle(msg);
    };
  }

  function scheduleReconnect() {
    setTimeout(connect, retry);
    retry = Math.min(8000, retry * 1.6);
  }

  function handle(msg) {
    switch (msg.type) {
      case 'hello':
        renderStatus(msg.status);
        if (msg.status && msg.status.telemetry) renderTelemetry(msg.status.telemetry);
        break;

      case 'event': {
        const ev = msg.event || {};
        pushEvent(ev);
        const t = String(ev.topic || '');
        if (t === 'security.permission.request') renderPermission(ev.data || {});
        if (t === 'security.permission.answer' || t === 'security.permission.deny' ||
            t === 'security.permission.timeout') dropPermission((ev.data || {}).id);
        if (t === 'voice.speak.start') { busy.speaking = Date.now(); }
        if (t === 'brain.think.start') { busy.thinking++; markBusy('nano'); }
        if (t === 'brain.answer' || t === 'brain.think.end') { busy.thinking = Math.max(0, busy.thinking - 1); recomputeState(); }
        if (t === 'security.revive') { busy.killed = false; $('#kill-overlay').classList.add('hidden'); recomputeState(); }
        if (t === 'security.kill') { busy.killed = true; $('#kill-reason').textContent = JSON.stringify(ev.data || {}); recomputeState(); }
        if (t === 'memory.write') syncMemory();
        if (ev.data && ev.data.telemetry) renderTelemetry(ev.data.telemetry);
        break;
      }

      case 'telemetry': renderTelemetry(msg.data); break;

      case 'voice.state':
        Talk.setState(msg.state, msg.reason);
        if (msg.reason === 'idle timeout') toast('השיחה נסגרה — שקט מדי זמן רב', 'warn', 3000);
        if (msg.reason === 'still listening') Talk.hint('עדיין מקשיב — פשוט דבר');
        if (msg.wake_required === true) Talk.hint('אמור <b>ג׳רוויס</b> ואחר כך דבר');
        break;

      case 'voice.level':
        Talk.setLevel(msg.db, msg.rms, msg.wake, msg.wake_threshold);
        break;

      case 'voice.heard':
        addMsg('user', esc(msg.text),
               `🗣 ${esc(msg.label || 'זיהוי')} · ${Math.round((msg.confidence || 0) * 100)}% · ${Math.round(msg.ms || 0)}ms`);
        break;

      case 'voice.muted':
        Talk.setState('speaking', 'JARVIS is talking — microphone muted');
        Talk.hint('🔇 הוא מדבר עכשיו — המיקרופון ייפתח מיד כשיסיים');
        break;

      case 'voice.unmuted':
        Talk.hint('המיקרופון פתוח — <b>פשוט דבר</b>');
        break;

      case 'voice.unheard':
        if (msg.error) addMsg('err', esc(msg.error), '🗣 STT');
        else if (msg.wav_b64 && msg.teachable) Train.offerTeach(msg);
        else toast('שמעתי משהו אבל לא זיהיתי פקודה', 'warn', 2200);
        break;

      case 'audio':
        if (msg.wav_b64) {
          busy.speaking = Date.now();
          markBusy('vox', 4000);
          Voice.playB64(msg.wav_b64);
          recomputeState();
        }
        break;

      case 'thinking':
        busy.thinking++; recomputeState();
        break;

      case 'answer': {
        busy.thinking = Math.max(0, busy.thinking - 1);
        recomputeState();
        (msg.audio || []).forEach(fr => { if (fr && fr.wav_b64) { markBusy('vox', 4000); Voice.playB64(fr.wav_b64); } });
        const typing = $('#transcript .msg:last-child .typing');
        if (typing) typing.closest('.msg').remove();
        renderTurn(msg);
        if (msg.answer && msg.answer.needs_confirmation) toast('JARVIS ממתין לאישור שלך', 'warn');
        break;
      }

      case 'boot': break;
      case 'control':
        if (msg.killed === true) { busy.killed = true; $('#kill-reason').textContent = msg.reason || '—'; }
        if (msg.killed === false) { busy.killed = false; $('#kill-overlay').classList.add('hidden'); }
        api('/api/status').then(renderStatus).catch(() => {});
        recomputeState();
        break;

      case 'skill_result':
        addMsg(msg.ok ? 'sys' : 'err', esc(msg.ok ? JSON.stringify(msg.value, null, 2) : msg.error),
               'SKILL ' + (msg.skill || ''), { chips: [{ t: (msg.ms || 0).toFixed(0) + 'ms', k: 'skill' }] });
        break;

      case 'speak_result':
        toast(msg.ok ? `דובר (${(msg.seconds || 0).toFixed(1)}s · ${msg.engine || ''})` : 'הדיבור נכשל',
              msg.ok ? 'good' : 'err');
        break;

      case 'pong': break;
      case 'error': toast(msg.message || 'שגיאה', 'err'); break;
      default: break;
    }
  }

  // ══════════════════════════ local speech capture ══════════════════════════
  // ══════════════════════ microphone selection ══════════════════════
  // A desk usually has several inputs (headset, webcam mic, line-in) and the
  // browser picks one silently — JARVIS then listens to the wrong room. The
  // choice is explicit, remembered across runs, and falls back to the default
  // when the remembered device is unplugged. Device labels are blank until
  // permission has been granted once, so acquire() refreshes the list.
  const Mic = (() => {
    const KEY = 'jarvis.mic.deviceId';
    const CAMKEY = 'jarvis.cam.deviceId';
    let devices = [], chosen = '';
    let cams = [], camChosen = '';
    try { chosen = localStorage.getItem(KEY) || ''; } catch (_) {}
    try { camChosen = localStorage.getItem(CAMKEY) || ''; } catch (_) {}

    function persist() {
      try { chosen ? localStorage.setItem(KEY, chosen) : localStorage.removeItem(KEY); } catch (_) {}
    }

    function constraints() {
      const audio = { channelCount: 1, echoCancellation: true,
                      noiseSuppression: true, autoGainControl: true };
      if (chosen) audio.deviceId = devices.some(d => d.deviceId === chosen)
        ? { exact: chosen } : { ideal: chosen };
      return { audio };
    }

    async function acquire() {
      const gum = c => navigator.mediaDevices.getUserMedia(c);
      let stream;
      try {
        stream = await gum(constraints());
      } catch (e) {
        if (chosen && /Overconstrained|NotFound|NotReadable/.test(e.name || '')) {
          chosen = ''; persist(); render();
          toast('המיקרופון שנבחר אינו מחובר — עברתי לברירת המחדל', 'warn');
          stream = await gum(constraints());
        } else throw e;
      }
      refresh();
      return stream;
    }

    function render() {
      const sel = document.getElementById('mic-select');
      if (!sel) return;
      sel.innerHTML = '';
      (devices.length ? devices : [{ deviceId: '', label: 'ברירת מחדל' }]).forEach((d, i) => {
        const o = document.createElement('option');
        o.value = d.deviceId || '';
        o.textContent = d.label || `מיקרופון ${i + 1}`;
        sel.appendChild(o);
      });
      sel.value = devices.some(d => d.deviceId === chosen) ? chosen : '';
      sel.disabled = devices.length < 2;
      sel.title = devices.length < 2
        ? 'מיקרופון אחד מזוהה. חבר מכשיר נוסף כדי לבחור ביניהם.'
        : 'המיקרופון לדיבור, לשיחה החופשית ולאישון הקול';
    }

    function renderCams() {
      const sel = document.getElementById('cam-select');
      if (!sel) return;
      sel.innerHTML = '';
      (cams.length ? cams : [{ deviceId: '', label: 'ברירת מחדל' }]).forEach((d, i) => {
        const o = document.createElement('option');
        o.value = d.deviceId || '';
        o.textContent = d.label || `מצלמה ${i + 1}`;
        sel.appendChild(o);
      });
      sel.value = cams.some(d => d.deviceId === camChosen) ? camChosen : '';
      sel.disabled = cams.length < 2;
      sel.title = cams.length < 2 ? 'מצלמה אחת מזוהה' : 'המצלמה לזיהוי פנים';
    }

    function videoConstraints() {
      const video = { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' };
      if (camChosen) video.deviceId = cams.some(d => d.deviceId === camChosen)
        ? { exact: camChosen } : { ideal: camChosen };
      return { video, audio: false };
    }

    async function refresh() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return devices;
      try {
        const all = await navigator.mediaDevices.enumerateDevices();
        devices = all.filter(d => d.kind === 'audioinput' && d.deviceId);
        cams = all.filter(d => d.kind === 'videoinput' && d.deviceId);
      } catch (_) { devices = []; cams = []; }
      render();
      renderCams();
      return devices;
    }

    function init() {
      const sel = document.getElementById('mic-select');
      if (!sel) return;
      sel.onchange = () => {
        chosen = sel.value; persist();
        toast(chosen ? 'המיקרופון נשמר — ההקלטה הבאה תשתמש בו' : 'חזרתי למיקרופון ברירת המחדל', 'ok');
      };
      const cam = document.getElementById('cam-select');
      if (cam) cam.onchange = () => {
        camChosen = cam.value;
        try { camChosen ? localStorage.setItem(CAMKEY, camChosen) : localStorage.removeItem(CAMKEY); } catch (_) {}
        toast(camChosen ? 'המצלמה נשמרה' : 'מצלמת ברירת המחדל', 'ok');
        if (typeof Vision !== 'undefined' && Vision.on) Vision.restart();
      };
      if (navigator.mediaDevices && navigator.mediaDevices.addEventListener)
        navigator.mediaDevices.addEventListener('devicechange', () => refresh());
      refresh();
    }

    return { init, refresh, acquire, constraints, videoConstraints,
             get chosen() { return chosen; }, get count() { return devices.length; },
             get camCount() { return cams.length; } };
  })();

  // ══════════════════ sight — who is sitting at the machine ══════════════════
  // Frames go to the brain as raw RGB pixels read straight off a canvas, so there
  // is no image codec anywhere in the path. The brain detects, aligns on the eyes,
  // matches against the enrolled gallery and answers with the permission level
  // that person holds — a grant that expires the moment they leave the frame.
  const Vision = (() => {
    const GRAB_W = 320;        // what we ship; enough for a face, small to send
    const INTERVAL = 1200;     // ms between looks
    let stream = null, timer = null, poller = null, busy = false, on = false, last = null;

    const el = id => document.getElementById(id);

    function stat(rows) {
      const box = el('vision-stats');
      if (box) box.innerHTML = rows.map(([k, v, good]) =>
        `<div class="kv-row"><span>${k}</span><b class="${good === true ? 'ok' : good === false ? 'bad' : ''}">${v}</b></div>`).join('');
    }

    function badge(match, gate) {
      const b = el('who-badge'), nm = el('who-name'), lv = el('who-level'), bar = el('bar-who');
      if (!b) return;
      const known = match && match.known;
      const owner = !!(match && match.is_owner);
      b.className = 'who ' + (known ? 'known lvl-' + (match.level || 'SAFE').toLowerCase()
                                    + (owner ? ' is-owner' : '')
                                    : (match && match.faces ? 'stranger' : 'empty'));
      nm.textContent = known
        ? (owner ? '★ ' + match.name : match.name)
        : (match && match.faces ? 'לא מזוהה' : 'אין איש מול המצלמה');
      lv.textContent = (gate && gate.level) || (match && match.level) || 'SAFE';
      const conf = (match && match.confidence) || 0;
      if (bar) bar.style.width = Math.round(Math.max(0, Math.min(1, conf)) * 100) + '%';
      b.title = known
        ? (owner
            ? `${match.name} — היוצר. הרשאות מלאות (CRITICAL). רק הוא יכול להגיע לרמה הזו.`
            : `זוהה בוודאות ${(conf * 100).toFixed(0)}% · הרשאה ${match.level}`)
        : `ביטחון ${(conf * 100).toFixed(0)}% — מתחת לסף, ולכן SAFE`;
    }

    // What the camera is actually looking at — not just whose face it is. The
    // summary is written server-side in Hebrew so the HUD never has to guess at
    // the thresholds, and the warnings are the actionable part: "it's too dark"
    // is something the user can fix, "confidence 61%" is not.
    function scene(sc, gate) {
      const sum = el('scene-summary'), warn = el('scene-warns');
      if (sum) sum.textContent = (sc && sc.summary_he) || '—';
      if (warn) {
        const w = (sc && sc.warnings) || [];
        const withheld = gate && gate.withheld === 'liveness';
        const all = withheld ? ['הפריים נראה כמו מסך או תמונה — ההרשאה הושהתה'].concat(w) : w;
        warn.innerHTML = all.map(x => `<span class="scene-warn">${x}</span>`).join('');
        warn.className = 'scene-warns' + (all.length ? ' active' : '');
      }
      const q = el('bar-quality'), qv = el('val-quality');
      if (q && sc) q.style.width = Math.round((sc.quality || 0) * 100) + '%';
      if (qv && sc) qv.textContent = Math.round((sc.quality || 0) * 100) + '%';
      const lb = el('bar-light'), lv2 = el('val-light');
      const L = sc && sc.lighting;
      if (lb && L) lb.style.width = Math.round(Math.max(0, Math.min(1, L.mean / 255)) * 100) + '%';
      if (lv2 && L) {
        const names = { good: 'טובה', dim: 'חלשה', dark: 'חשוך', blown: 'שרוף', flat: 'שטוחה' };
        lv2.textContent = names[L.verdict] || L.verdict;
      }
    }

    function drawBoxes(res) {
      const cv = el('cam-boxes'), v = el('cam-view');
      if (!cv || !v) return;
      const w = v.clientWidth || 320, h = v.clientHeight || 240;
      if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
      const g = cv.getContext('2d');
      g.clearRect(0, 0, w, h);
      const fw = (res.frame && res.frame.w) || GRAB_W;
      const fh = (res.frame && res.frame.h) || 1;
      const sx = w / fw, sy = h / fh;
      const known = res.match && res.match.known;
      (res.boxes || []).forEach((b, i) => {
        g.strokeStyle = i === 0 ? (known ? '#5dffb0' : '#ffb45d') : 'rgba(127,233,255,.5)';
        g.lineWidth = i === 0 ? 2 : 1;
        g.strokeRect(b.x * sx, b.y * sy, b.w * sx, b.h * sy);
      });
    }

    function grab() {
      const v = el('cam-view');
      if (!v || !v.videoWidth) return null;
      const w = GRAB_W, h = Math.max(1, Math.round(GRAB_W * v.videoHeight / v.videoWidth));
      const cv = document.createElement('canvas');
      cv.width = w; cv.height = h;
      const g = cv.getContext('2d', { willReadFrequently: true });
      g.drawImage(v, 0, 0, w, h);
      const d = g.getImageData(0, 0, w, h).data;
      const rgb = new Uint8Array(w * h * 3);
      for (let i = 0, j = 0; i < d.length; i += 4) { rgb[j++] = d[i]; rgb[j++] = d[i + 1]; rgb[j++] = d[i + 2]; }
      let bin = '';
      const CH = 0x8000;
      for (let i = 0; i < rgb.length; i += CH) bin += String.fromCharCode.apply(null, rgb.subarray(i, i + CH));
      return { w, h, rgb: btoa(bin) };
    }

    async function look(silent) {
      if (busy || !on) return;
      const frame = grab();
      if (!frame) return;
      busy = true;
      try {
        const r = await fetch('/api/faces/recognize', {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(frame)
        });
        const res = await r.json();
        if (!res.ok) { if (!silent) toast(res.error || 'הזיהוי נכשל', 'err'); return; }
        last = res;
        badge(res.match, res.gate);
        scene(res.scene, res.gate);
        drawBoxes(res);
        stat([['זוהו', `${res.match.faces} פנים`, res.match.faces > 0],
              ['ביטחון', `${Math.round((res.match.confidence || 0) * 100)}%`, res.match.known],
              ['הרשאה פעילה', res.gate.level, res.gate.level !== 'SAFE'],
              ['מנוע', res.gate.armed ? 'מחובר לחומת האש' : 'מנותק', res.gate.armed]]);
        if (res.gate.identity && res.match.known) {
          const who = res.match.name;
          if (!Vision._said || Vision._said !== who) {
            Vision._said = who;
            pushEvent({ topic: 'vision.identity', data: { name: who, level: res.gate.level,
                        confidence: res.match.confidence }, ts: Date.now() / 1000 });
          }
        } else Vision._said = null;
      } catch (e) {
        if (!silent) toast('אין חיבור למוח לזיהוי פנים', 'err');
      } finally { busy = false; }
    }

    async function refresh() {
      try {
        const res = await (await fetch('/api/faces')).json();
        const gate = res.gate || {};
        const chk = el('chk-gate');
        if (chk) chk.checked = !!gate.armed;
        stat([['אנשים רשומים', res.people ? res.people.length : 0, (res.people || []).length > 0],
              ['סף זיהוי', res.threshold || '—', null],
              ['הרשאה פעילה', gate.level || 'SAFE', (gate.level || 'SAFE') !== 'SAFE'],
              ['מנוע', gate.armed ? 'מחובר לחומת האש' : 'מנותק', gate.armed]]);
        const list = el('face-list');
        if (list) {
          list.innerHTML = (res.people || []).map(p =>
            `<li${p.role === 'owner' ? ' class="owner"' : ''}>` +
            `<span class="fname">${p.role === 'owner' ? '★ ' : ''}${p.name}</span>` +
            `<select class="flevel" data-id="${p.id}"${p.role === 'owner' ? ' title="הבעלים — ההרשאה מוגנת"' : ''}>` +
            ['SAFE', 'WRITE', 'CRITICAL'].map(l =>
              `<option value="${l}"${l === p.level ? ' selected' : ''}>${l}</option>`).join('') +
            `</select><span class="fsamp">${p.samples} דגימות</span>` +
            `<button class="fdel" data-id="${p.id}" title="מחק">✕</button></li>`).join('')
            || '<li class="dim">אף אחד לא רשום — ההרשמה הראשונה תהפוך לבעלים עם הרשאות מלאות</li>';
          list.querySelectorAll('.flevel').forEach(sel => sel.onchange = async () => {
            const r = await admin('set_level', { id: sel.dataset.id, level: sel.value });
            toast(r.ok ? `ההרשאה עודכנה ל־${sel.value}`
                       : (r.error || 'ההרשאה לא שונתה — הרשאת הבעלים מוגנת'),
                  r.ok ? 'ok' : 'err');
            refresh();
          });
          list.querySelectorAll('.fdel').forEach(b => b.onclick = async () => {
            const r = await admin('remove', { id: b.dataset.id });
            toast(r.ok ? 'הפנים נמחקו' : (r.error || 'נכשל'), r.ok ? 'ok' : 'err');
            refresh();
          });
        }
      } catch (e) { stat([['ראייה', 'לא זמינה', false]]); }
    }

    async function admin(action, extra) {
      try {
        const r = await fetch('/api/faces/admin', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(Object.assign({ action }, extra || {}))
        });
        return await r.json();
      } catch (e) { return { ok: false, error: String(e) }; }
    }

    async function start() {
      if (on) return stop();
      const v = el('cam-view');
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        toast('אין גישה למצלמה בסביבה זו', 'err'); return false;
      }
      try {
        stream = await navigator.mediaDevices.getUserMedia(Mic.videoConstraints());
      } catch (e) { toast('המצלמה נדחתה: ' + e.message, 'err'); return false; }
      if (v) { v.srcObject = stream; try { await v.play(); } catch (_) {} }
      const off = el('cam-off'); if (off) off.classList.add('hidden');
      const btn = el('btn-cam'); if (btn) { btn.textContent = '⏹ כבה מצלמה'; btn.classList.add('live'); }
      on = true;
      Mic.refresh();                       // permission granted → real camera names
      await look(true);
      timer = setInterval(() => look(true), INTERVAL);
      // The grant decays even between frames. Kept in a variable and cleared on
      // stop: an anonymous interval here survived every off/on cycle and piled up.
      if (poller) clearInterval(poller);
      poller = setInterval(() => {
        if (!on) return;
        admin('poll').then(r => { if (r && r.gate) badge((last || {}).match, r.gate); }).catch(() => {});
      }, 4000);
      toast('המצלמה פתוחה — JARVIS מסתכל', 'ok');
      return true;
    }

    async function stop() {
      on = false;
      if (timer) { clearInterval(timer); timer = null; }
      if (poller) { clearInterval(poller); poller = null; }
      try { stream && stream.getTracks().forEach(t => t.stop()); } catch (_) {}
      stream = null;
      const v = el('cam-view'); if (v) v.srcObject = null;
      const off = el('cam-off'); if (off) off.classList.remove('hidden');
      const btn = el('btn-cam'); if (btn) { btn.textContent = '📷 מצלמה'; btn.classList.remove('live'); }
      const cv = el('cam-boxes'); if (cv) cv.getContext('2d').clearRect(0, 0, cv.width, cv.height);
      badge(null, null);
      stat([['מצלמה', 'כבויה', false]]);
    }

    async function restart() { if (on) { await stop(); await start(); } }

    async function enroll() {
      const nameEl = el('enroll-name'), lvlEl = el('enroll-level');
      const name = (nameEl && nameEl.value || '').trim();
      if (!name) { toast('כתוב שם לפני שרושמים פנים', 'warn'); if (nameEl) nameEl.focus(); return; }
      if (!on) { const s = await start(); if (!s) return; }
      const frame = grab();
      if (!frame) { toast('עדיין אין תמונה מהמצלמה', 'warn'); return; }
      const cnt = el('enroll-count');
      try {
        const r = await (await fetch('/api/faces/enroll', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(Object.assign({ name, level: (lvlEl && lvlEl.value) || 'SAFE' }, frame))
        })).json();
        if (!r.ok) { toast(r.error || 'הרישום נכשל', 'err'); return; }
        toast(`הפנים של ${r.person.name} נשמרו (${r.person.samples} דגימות, הרשאה ${r.person.level})`, 'ok');
        if (r.warning) toast(r.warning, 'warn', 6000);
        if (cnt) cnt.textContent = r.person.samples;
        refresh(); look(true);
      } catch (e) { toast('אין חיבור למוח לרישום פנים', 'err'); }
    }

    function init() {
      const b = el('btn-cam'); if (b) b.onclick = () => start();
      const s = el('btn-cam-shot'); if (s) s.onclick = () => { if (!on) start(); else look(false); };
      const e = el('btn-enroll'); if (e) e.onclick = () => enroll();
      const chk = el('chk-gate');
      if (chk) chk.onchange = async () => {
        const r = await admin(chk.checked ? 'arm' : 'disarm');
        toast(chk.checked ? 'המצלמה קובעת הרשאות' : 'המצלמה לא משנה הרשאות — הזיהוי רק מדווח',
              r.ok ? 'ok' : 'err');
        refresh();
      };
      refresh();
    }

    return { init, start, stop, restart, look, refresh, enroll, admin,
             get on() { return on; } };
  })();

  const Rec = (() => {
    let ac = null, proc = null, stream = null, chunks = [], recording = false, t0 = 0;

    function encodeWav(float32, sampleRate) {
      const n = float32.length;
      const buf = new ArrayBuffer(44 + n * 2);
      const dv = new DataView(buf);
      const wstr = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
      wstr(0, 'RIFF'); dv.setUint32(4, 36 + n * 2, true); wstr(8, 'WAVE');
      wstr(12, 'fmt '); dv.setUint32(16, 16, true); dv.setUint16(20, 1, true);
      dv.setUint16(22, 1, true); dv.setUint32(24, sampleRate, true);
      dv.setUint32(28, sampleRate * 2, true); dv.setUint16(32, 2, true); dv.setUint16(34, 16, true);
      wstr(36, 'data'); dv.setUint32(40, n * 2, true);
      let o = 44;
      for (let i = 0; i < n; i++, o += 2) {
        const v = Math.max(-1, Math.min(1, float32[i]));
        dv.setInt16(o, v < 0 ? v * 0x8000 : v * 0x7fff, true);
      }
      return buf;
    }

    async function start() {
      if (recording) return stop();
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        toast('אין גישה למיקרופון בסביבה זו', 'err'); return false;
      }
      try {
        stream = await Mic.acquire();
      } catch (e) { toast('המיקרופון נדחה: ' + e.message, 'err'); return false; }
      const AC = window.AudioContext || window.webkitAudioContext;
      ac = new AC();
      if (ac.state === 'suspended') await ac.resume();
      const src = ac.createMediaStreamSource(stream);
      proc = ac.createScriptProcessor(4096, 1, 1);
      chunks = []; t0 = Date.now();
      proc.onaudioprocess = e => chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
      src.connect(proc); proc.connect(ac.destination);
      recording = true;
      $('#btn-mic').classList.add('rec');
      $('#btn-mic').textContent = '⏹';
      pushEvent({ topic: 'voice.listen.start', data: { sr: ac.sampleRate }, ts: Date.now() / 1000 });
      toast('מקשיב… לחץ שוב לעצירה', 'warn', 3000);
      return true;
    }

    async function stop() {
      if (!recording) return false;
      recording = false;
      $('#btn-mic').classList.remove('rec');
      $('#btn-mic').textContent = '🎙';
      const sr = ac.sampleRate;
      const total = chunks.reduce((a, c) => a + c.length, 0);
      const pcm = new Float32Array(total);
      let o = 0; chunks.forEach(c => { pcm.set(c, o); o += c.length; });
      try { proc.disconnect(); } catch (_) {}
      try { stream.getTracks().forEach(t => t.stop()); } catch (_) {}
      try { ac.close(); } catch (_) {}
      const secs = total / sr;
      pushEvent({ topic: 'voice.listen.end', data: { seconds: +secs.toFixed(2) }, ts: Date.now() / 1000 });
      if (secs < 0.45) { toast('קטע קצר מדי — נסה שוב', 'warn'); return false; }

      addMsg('user', `<span class="typing"><i></i><i></i><i></i></span> מזהה דיבור מקומי…`, '🎙 מיקרופון');
      try {
        const r = await fetch(HTTP + '/api/listen', {
          method: 'POST', headers: { 'Content-Type': 'application/octet-stream' },
          body: encodeWav(pcm, sr)
        });
        const res = await r.json();
        const last = $('#transcript .msg:last-child'); if (last) last.remove();
        if (res.ok && res.text) {
          addMsg('user', esc(res.text), `🎙 ${esc(res.label || '')} · ${Math.round((res.confidence || 0) * 100)}%`);
          if (res.turn) renderTurn(res.turn);
          else if (res.reply) addMsg('bot', esc(res.reply), 'JARVIS');
        } else {
          addMsg('err', esc(res.reply || res.error || 'הזיהוי לא הצליח'), 'STT');
          toast(res.error || 'לא זיהיתי פקודה', 'warn');
        }
      } catch (e) {
        addMsg('err', esc(String(e.message || e)), 'STT');
      }
      return true;
    }

    return { start, stop, encodeWav, get recording() { return recording; } };
  })();

  // ══════════════ hands-free conversation (streaming microphone) ══════════════
  // Push-to-talk records, stops, uploads. This keeps the microphone open and lets
  // the server do the turn-taking: wake word -> listen -> answer aloud -> listen.
  let lastLoud = 0, quietRun = 0, deadWarned = false, lowWakeRun = 0, trainNudged = false;
  let gated = 0, unmuteSent = false;

  const Talk = (() => {
    const TARGET = 16000;                    // the STT templates live at 16 kHz
    const ACK = 'כן, אדוני?';
    const LABELS = { off: 'שיחה חופשית כבויה', idle: 'ממתין להשכמה — אמור "ג׳רוויס"',
                     listening: 'מקשיב…', thinking: 'מזהה…', speaking: 'מדבר' };
    const sid = 'hud-' + Math.random().toString(36).slice(2, 10);
    let on = false, ac = null, stream = null, proc = null, src = null, state = 'off';

    function resample(f32, srIn) {
      if (srIn === TARGET) return f32;
      const ratio = TARGET / srIn, n = Math.floor(f32.length * ratio);
      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const p = i / ratio, i0 = Math.floor(p), i1 = Math.min(i0 + 1, f32.length - 1), t = p - i0;
        out[i] = f32[i0] * (1 - t) + f32[i1] * t;
      }
      return out;
    }

    function toInt16(f32) {
      const b = new Int16Array(f32.length);
      for (let i = 0; i < f32.length; i++) {
        const v = Math.max(-1, Math.min(1, f32[i]));
        b[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
      }
      return b;
    }

    async function restVoice(body) {
      const r = await fetch(HTTP + '/api/voice', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(Object.assign({ sid }, body))
      });
      const res = await r.json();
      (res.events || []).forEach(handle);
      return res;
    }

    async function ship(bytes) {
      // The server mutes the microphone for as long as it *thinks* playback will
      // take, but the HUD is the thing actually making the sound and it knows
      // better: if the estimate expires early, JARVIS hears his own answer and
      // replies to himself. Gating here closes that loop, and the counter feeds
      // the hint line so the deafness is visible rather than mysterious.
      if (window.Voice && (Voice.speaking || Voice.pending > 0)) { gated++; return; }
      if (ws && ws.readyState === 1) { ws.send(bytes); return; }
      // REST twin: the session queues its events server-side and this response
      // carries them back, together with any speech the answer produced.
      try {
        const r = await fetch(HTTP + '/api/audio?sid=' + encodeURIComponent(sid), {
          method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: bytes
        });
        const res = await r.json();
        (res.events || []).forEach(handle);
        (res.audio || []).forEach(fr => {
          if (fr && fr.wav_b64) { busy.speaking = Date.now(); markBusy('vox', 4000); Voice.playB64(fr.wav_b64); }
        });
        if (res.state) setState(res.state);
      } catch (_) { /* one lost chunk must not end a conversation */ }
    }

    // Guard against a second start while one is still awaiting the microphone.
    // F5 can arrive twice for one press (the global hotkey and the renderer's own
    // keydown), and `on` only flips at the very end of the session setup — so two
    // overlapping starts used to both get through, leaving a second stream and
    // ScriptProcessor attached, or toggling the session straight back off.
    let starting = false;
    async function start() {
      if (on) return stop();
      if (starting) return false;
      starting = true;
      try { return await beginSession(); } finally { starting = false; }
    }

    async function beginSession() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        toast('אין גישה למיקרופון בסביבה זו', 'err'); return false;
      }
      try {
        stream = await Mic.acquire();
      } catch (e) { toast('המיקרופון נדחה: ' + e.message, 'err'); return false; }

      // Asking the browser for 16 kHz outright avoids resampling altogether; if it
      // refuses we interpolate down ourselves.
      const AC = window.AudioContext || window.webkitAudioContext;
      try { ac = new AC({ sampleRate: TARGET }); } catch (_) { ac = new AC(); }
      if (ac.state === 'suspended') { try { await ac.resume(); } catch (_) {} }

      try {
        if (ws && ws.readyState === 1) send({ type: 'voice_start', ack: ACK, idle_timeout: 25 });
        else await restVoice({ action: 'start', ack: ACK, idle_timeout: 25 });
      } catch (e) { toast('השרת סירב לפתוח שיחה: ' + e.message, 'err'); }

      src = ac.createMediaStreamSource(stream);
      proc = ac.createScriptProcessor(4096, 1, 1);
      proc.onaudioprocess = e => {
        if (!on) return;
        const x = resample(new Float32Array(e.inputBuffer.getChannelData(0)), ac.sampleRate);
        ship(toInt16(x).buffer);
      };
      src.connect(proc); proc.connect(ac.destination);
      on = true;
      quietRun = 0; deadWarned = false; lowWakeRun = 0; trainNudged = false;
      // The session opens straight into LISTENING (arm_on_start) — the wake word
      // is a bonus path, not a gate, because it only matches voices that have
      // templates in the bank.
      setState('listening', 'microphone open — just speak');
      hint('המיקרופון פתוח — <b>פשוט דבר</b>. מילת השכמה? לחץ 🎯 אימון קול');
      paint();
      pushEvent({ topic: 'voice.session.start', data: { sid, sr: ac.sampleRate }, ts: Date.now() / 1000 });
      toast('המיקרופון פתוח ומקשיב — פשוט דבר', 'good', 4200);
      return true;
    }

    async function stop() {
      if (!on) return false;
      on = false;
      try { proc.disconnect(); } catch (_) {}
      try { src.disconnect(); } catch (_) {}
      try { stream.getTracks().forEach(t => t.stop()); } catch (_) {}
      try { ac.close(); } catch (_) {}
      proc = src = stream = ac = null;
      try {
        if (ws && ws.readyState === 1) send({ type: 'voice_stop' });
        else await restVoice({ action: 'stop' });
      } catch (_) {}
      setState('off', 'stopped by user');
      paint();
      pushEvent({ topic: 'voice.session.stop', data: { sid }, ts: Date.now() / 1000 });
      return true;
    }

    function paint() {
      const b = $('#btn-talk');
      if (!b) return;
      b.classList.toggle('live', on);
      b.textContent = on ? '🗣 שיחה פעילה' : '🗣 שיחה';
      document.body.classList.toggle('talking', on);
    }

    function setState(next, reason) {
      state = next || 'off';
      const el = $('#talk-state'), dot = $('#talk-dot');
      if (el) el.textContent = LABELS[state] || state;
      if (el && reason) el.title = String(reason);
      if (dot) dot.className = 'st-' + state;
      const reactor = $('#reactor-state');
      if (reactor && state !== 'off') {
        reactor.textContent = state === 'listening' ? 'מקשיב'
          : state === 'thinking' ? 'מזהה' : state === 'speaking' ? 'מדבר' : 'ממתין';
      }
    }

    function hint(html) {
      const el = $('#talk-hint');
      if (el && html) el.innerHTML = html;
    }

    function setLevel(db, rms, wake, wakeThreshold) {
      const bar = $('#talk-level i');
      if (!bar) return;
      const norm = Math.max(0, Math.min(1, (Number(db) + 55) / 55));
      bar.style.width = (norm * 100).toFixed(1) + '%';
      bar.classList.toggle('hot', norm > 0.55);

      // "He isn't listening" has three possible causes and they look identical
      // from the outside. This readout tells them apart: no signal at all is a
      // microphone problem, signal with a low wake score is a voice-match
      // problem, and both are worth saying out loud instead of leaving the user
      // to guess.
      const loud = Number(rms) > 0.0008;
      if (loud) { lastLoud = Date.now(); quietRun = 0; deadWarned = false; }
      else if (on) { quietRun++; }
      const w = $('#talk-wake');
      if (w && wake !== undefined && wake !== null) {
        const thr = Number(wakeThreshold || 0.55);
        w.textContent = 'זיהוי קול ' + Number(wake).toFixed(2) + '/' + thr.toFixed(2);
        w.className = Number(wake) >= thr ? 'w-ok' : (Number(wake) >= thr * 0.5 ? 'w-mid' : 'w-low');
        w.title = 'כמה קרוב הקול שלך להפעלת מילת ההשכמה. נמוך? לחץ 🎯 אימון קול.';
      }
      if (on && !deadWarned && quietRun > 34) {          // ~4 s at 8 Hz of nothing
        deadWarned = true;
        toast('המיקרופון לא שולח קול — בדוק הרשאות או התקן קלט', 'err', 6000);
        hint('🔇 אין קול מהמיקרופון — בדוק הרשאות במערכת');
      }
      if (on && !trainNudged && loud && wake !== undefined && Number(wake) < 0.25) {
        lowWakeRun++;
        if (lowWakeRun > 24) {                            // ~3 s of speech, no match
          trainNudged = true;
          toast('שומע אותך, אבל הקול שלך לא תואם את בנק הדגימות — לחץ 🎯 אימון קול', 'warn', 7000);
        }
      } else if (!loud) { lowWakeRun = 0; }
    }

    return { start, stop, setState, setLevel, hint, paint, ship, resample, toInt16,
             get on() { return on; }, get sid() { return sid; },
             get available() { return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia); } };
  })();

  // ══════════ voice enrolment — teach JARVIS the voice in the room ══════════
  // The shipped template bank is synthesised from our own TTS. Measured: the same
  // words in a different timbre score 0.000 against it, while two recordings of
  // the listener lift an *unseen* take to ~1.000 and a spoken command from 0.08 to
  // 0.77 confidence. This panel is therefore not a refinement — it is what makes
  // the wake word answer to a human being at all.
  const Train = (() => {
    let script = [], busy = false, stopFn = null, pendingDone = null;

    function note(t) { const n = $('#train-note'); if (n) n.textContent = t; }

    async function load() {
      try {
        const res = await (await fetch(HTTP + '/api/enroll')).json();
        script = res.script || [];
        render();
        const n = res.enrolled_total || 0;
        note(n < 1 ? 'אין עדיין דגימות שלך — מילת ההשכמה מכווננת כרגע לקול המסונתז בלבד.'
           : n < 3 ? `יש ${n} דגימות. כדאי לפחות 3-4 משפטים — דגימה בודדת יכולה למשוך גם משפטים אחרים אליה.`
                   : `יש כבר ${n} דגימות שלך בבנק (${res.templates} תבניות). אפשר להוסיף עוד.`);
      } catch (e) { note('לא הצלחתי לקרוא את רשימת האימון: ' + e.message); }
    }

    function render() {
      const list = $('#train-list');
      if (!list) return;
      list.innerHTML = '';
      script.forEach(ph => {
        const row = document.createElement('div');
        row.className = 'train-row';
        row.dataset.label = ph.label;
        row.innerHTML = '<span class="tr-text">“' + esc(ph.text) + '”</span>' +
                        '<button class="tr-rec">🎙 הקלט</button>' +
                        '<span class="tr-state">' + (ph.enrolled ? '✓ ' + ph.enrolled + ' דגימות' : '—') + '</span>';
        row.querySelector('.tr-rec').onclick = () => stopFn ? stopFn() : capture(ph.label, ph.text, row);
        list.appendChild(row);
      });
    }

    function paintResult(st, res) {
      if (!res || !res.ok) {
        st.className = 'tr-state bad';
        st.textContent = '✗ ' + ((res && res.error) || 'ההעלאה נכשלה');
      } else {
        const pct = Math.round((res.score || 0) * 100);
        st.className = 'tr-state ' + (res.heard ? 'good' : 'bad');
        st.textContent = (res.heard ? '✓ ' : '⚠ ') + pct + '% — ' +
          (res.heard ? 'הוא מכיר את הקול שלך' : 'נמוך מדי, נסה שוב') +
          ' · ' + (res.frames || 0) + ' פריימים · ' + (res.bank || 0) + ' תבניות';
      }
      const d = pendingDone; pendingDone = null; if (d) d();
    }

    async function upload(label, wav) {
      try {
        const r = await fetch(HTTP + '/api/enroll?label=' + encodeURIComponent(label), {
          method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: wav
        });
        return await r.json();
      } catch (e) { return { ok: false, error: String(e.message || e) }; }
    }

    async function capture(label, text, row) {
      if (stopFn) return;
      const st = row.querySelector('.tr-state'), btn = row.querySelector('.tr-rec');
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        st.className = 'tr-state bad'; st.textContent = 'אין גישה למיקרופון';
        const d = pendingDone; pendingDone = null; if (d) d(); return;
      }
      let stream, ac, proc, src, chunks = [], sr = 16000, stopped = false;
      try {
        stream = await Mic.acquire();
      } catch (e) {
        st.className = 'tr-state bad'; st.textContent = 'המיקרופון נדחה';
        const d = pendingDone; pendingDone = null; if (d) d(); return;
      }
      const AC = window.AudioContext || window.webkitAudioContext;
      ac = new AC();
      if (ac.state === 'suspended') { try { await ac.resume(); } catch (_) {} }
      sr = ac.sampleRate;
      src = ac.createMediaStreamSource(stream);
      proc = ac.createScriptProcessor(4096, 1, 1);
      proc.onaudioprocess = e => { if (!stopped) chunks.push(new Float32Array(e.inputBuffer.getChannelData(0))); };
      src.connect(proc); proc.connect(ac.destination);

      const LIMIT = 3000, t0 = Date.now();
      btn.textContent = '⏹ עצור'; st.className = 'tr-state rec';
      const tick = setInterval(() => {
        const left = Math.max(0, (LIMIT - (Date.now() - t0)) / 1000);
        st.textContent = '🔴 אמור “' + text + '” · ' + left.toFixed(1) + 's';
        if (left <= 0) finish();
      }, 100);

      async function finish() {
        if (stopped) return;
        stopped = true; stopFn = null; clearInterval(tick);
        btn.textContent = '🎙 הקלט';
        try { proc.disconnect(); src.disconnect(); } catch (_) {}
        try { stream.getTracks().forEach(t => t.stop()); } catch (_) {}
        try { ac.close(); } catch (_) {}
        const total = chunks.reduce((a, c) => a + c.length, 0);
        const pcm = new Float32Array(total); let o = 0;
        chunks.forEach(c => { pcm.set(c, o); o += c.length; });
        if (total / sr < 0.45) {
          st.className = 'tr-state bad'; st.textContent = 'קצר מדי — נסה שוב';
          const d = pendingDone; pendingDone = null; if (d) d(); return;
        }
        st.className = 'tr-state wait'; st.textContent = 'שומר דגימה…';
        paintResult(st, await upload(label, Rec.encodeWav(pcm, sr)));
        load();                                  // refresh counts + totals
      }
      stopFn = finish;
    }

    async function all() {
      if (busy) return;
      busy = true;
      const btn = $('#train-all');
      if (btn) { btn.disabled = true; btn.textContent = '⏳ מאמן…'; }
      for (const ph of script) {
        const row = document.querySelector('#train-list .train-row[data-label="' + ph.label + '"]');
        if (!row) continue;
        note('עכשיו: אמור “' + ph.text + '” בקול רגיל');
        await new Promise(r => setTimeout(r, 800));
        await new Promise(r => { pendingDone = r; capture(ph.label, ph.text, row); });
        await new Promise(r => setTimeout(r, 350));
      }
      busy = false;
      if (btn) { btn.disabled = false; btn.textContent = '🎯 אמן הכול ברצף'; }
      note('האימון הושלם. ההשפעה מיידית — גם על שיחה שכבר פתוחה.');
      toast('אימון הקול הושלם — נסה לומר "ג׳רוויס"', 'good', 5000);
    }

    // ── teach-back: turn "he didn't understand me" into one click ──────────
    // The session ships the audio it failed on. Enrol it under the phrase the
    // listener meant, then replay it through the live session so the command
    // actually runs — the correction and the proof in the same breath.
    let vocab = null;
    async function getVocab() {
      if (vocab) return vocab;
      try {
        const res = await (await fetch(HTTP + '/api/enroll')).json();
        vocab = res.vocabulary || [];
      } catch (_) { vocab = []; }
      return vocab;
    }

    function b64ToBytes(b64) {
      const bin = atob(b64), out = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
      return out;
    }

    async function offerTeach(msg) {
      const list = await getVocab();
      if (!list.length) return;
      let best = (msg.top && msg.top[0]) || null;
      if (best && !(best.confidence > 0)) best = null;   // a 0% guess is no guess
      const guess = best ? `${best.label} · ${Math.round(best.confidence * 100)}%` : '';
      const opts = list.map(v =>
        `<option value="${esc(v.label)}"${best && best.label === v.label ? ' selected' : ''}>${esc(v.text)}</option>`).join('');
      const el = addMsg('err',
        `<div class="teach">לא הבנתי אותך${msg.confidence ? ` (הקרוב ביותר: ${esc(guess)}, ` +
          `${Math.round((msg.confidence || 0) * 100)}%)` : ''}.<br>` +
        `<span class="teach-q">מה אמרת?</span> <select class="teach-sel">${opts}</select>` +
        `<button class="teach-go">🎯 למד אותי והפעל</button></div>`,
        '🗣 זיהוי קול');
      if (!el) return;
      const go = el.querySelector('.teach-go'), sel = el.querySelector('.teach-sel');
      if (go) go.onclick = async () => {
        go.disabled = true; go.textContent = '⏳ לומד…';
        const r = await teach(sel.value, msg.wav_b64, el);
        go.textContent = r ? '✓ נלמד — מבצע שוב' : '✗ לא הצלחתי ללמוד';
        if (!r) go.disabled = false;
      };
    }

    async function teach(label, b64, el) {
      try {
        const bytes = b64ToBytes(b64);
        const res = await upload(label, bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
        if (!res || !res.ok) { toast('האימון נכשל: ' + ((res && res.error) || 'שגיאה'), 'err'); return false; }
        toast(res.advise && res.score >= 0.55 ? res.advise
              : `נלמד “${label}” — ציון ${Math.round((res.score || 0) * 100)}%`,
              res.advise && res.score >= 0.55 ? 'warn' : 'good', 4200);
        // replay it through the open session so the command runs right away
        const dec = window.Voice ? Voice.decodeWav(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)) : null;
        if (dec && Talk.on) {
          // JARVIS spoke the "didn't catch that" cue a moment ago, so the echo
          // guard may still be up — lift it explicitly before replaying.
          try {
            if (ws && ws.readyState === 1) send({ type: 'voice_unmute' });
            else await fetch(HTTP + '/api/voice', { method: 'POST',
                   headers: { 'Content-Type': 'application/json' },
                   body: JSON.stringify({ action: 'unmute', sid: Talk.sid }) });
          } catch (_) {}
          const x = Talk.resample(new Float32Array(dec.samples), dec.sampleRate);
          const tail = Math.round(16000 * 1.2);
          const all = new Float32Array(x.length + tail);
          all.set(x, 0);
          const pcm = Talk.toInt16(all).buffer;
          for (let i = 0; i < pcm.byteLength; i += 3200) {
            await Talk.ship(pcm.slice(i, i + 3200));
          }
        } else if (!Talk.on) {
          toast('פתח 🗣 שיחה כדי שהפקודה תרוץ מיד', 'warn', 4000);
        }
        load();
        return true;
      } catch (e) { toast('שגיאה באימון: ' + e.message, 'err'); return false; }
    }

    function open() {
      const el = $('#train-panel');
      if (el) el.classList.remove('hidden');
      load();
    }
    function close() {
      const el = $('#train-panel');
      if (el) el.classList.add('hidden');
      if (stopFn) stopFn();
    }

    return { open, close, load, all, offerTeach, teach, get busy() { return busy; } };
  })();

  // ══════════════════════════ interactions ══════════════════════════
  function bindUI() {
    const input = $('#input');

    // window chrome
    const win = window.jarvis;
    $('#btn-minimize').onclick = () => win ? win.minimize() : toast('מזעור זמין רק באפליקציית שולחן העבודה', 'warn');
    $('#btn-maximize').onclick = () => win ? win.toggleMaximize() : toast('הגדלה זמיין רק באפליקציית שולחן העבודה', 'warn');
    $('#btn-close').onclick = () => win ? win.close() : toast('במצב דפדפן סגור את הכרטיסייה', 'warn');
    $('#btn-always-top').onclick = e => {
      if (!win) return toast('נעיצה זמינה רק באפליקציית שולחן העבודה', 'warn');
      win.toggleAlwaysOnTop().then(on => { e.currentTarget.style.color = on ? 'var(--cyan)' : ''; });
    };
    $('#btn-fullscreen').onclick = () => {
      if (win) return win.toggleFullscreen();
      document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
    };
    if (!win) { $('#btn-minimize').style.opacity = $('#btn-maximize').style.opacity = '.45'; }

    // composer
    $('#btn-send').onclick = submit;
    $('#btn-mic').onclick = () => Rec.recording ? Rec.stop() : Rec.start();
    if ($('#btn-talk')) $('#btn-talk').onclick = () => Talk.on ? Talk.stop() : Talk.start();
    if ($('#btn-train')) $('#btn-train').onclick = () => Train.open();
    if ($('#train-close')) $('#train-close').onclick = () => Train.close();
    if ($('#train-all')) $('#train-all').onclick = () => Train.all();
    if ($('#btn-clear-chat')) $('#btn-clear-chat').onclick = () => {
      send({ type: 'clear_chat' });
      $('#transcript').innerHTML = '';
      countTurns(0);
      toast('היסטוריית השיחה נמחקה', 'warn');
    };
    input.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }
    });
    document.addEventListener('keydown', e => {
      // e.code, not e.key — on a Hebrew layout physical V reports 'ה' (and 'ו'
      // lives on U), so the old key-based match silently failed for the very
      // users this Hebrew voice assistant is for.  Accelerators in Electron are
      // already physical-key based; this brings the in-page handler in line.
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && !e.altKey && e.code === 'KeyV') {
        e.preventDefault();
        Talk.on ? Talk.stop() : Talk.start();
      }
    });
    input.addEventListener('input', () => { input.style.opacity = '1'; });

    $$('.mode').forEach(b => b.onclick = () => {
      $$('.mode').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      mode = b.dataset.mode;
      input.placeholder = MODES[mode].ph;
      input.focus();
    });

    $('#chk-speak').onchange = e => {
      Voice.setMuted(!e.target.checked);
      toast(e.target.checked ? 'קול פעיל' : 'קול מושתק', e.target.checked ? 'good' : 'warn', 1800);
    };

    // security
    $('#sel-level').onchange = e => { send({ type: 'set_level', level: e.target.value.toUpperCase() });
      toast('רמת הרשאות: ' + e.target.value, 'warn'); };
    $('#chk-dryrun').onchange = e => send({ type: 'set_dry_run', on: e.target.checked });
    $('#btn-kill').onclick = () => doKill('הופעל על ידי המשתמש מה־HUD');
    $('#btn-revive').onclick = () => send({ type: 'revive' });
    $('#btn-kill-revive').onclick = () => send({ type: 'revive' });
    $('#btn-clear-events').onclick = () => { $('#event-stream').innerHTML = ''; };

    // F5 — the microphone, not the keyboard. Electron sends this as a global
    // hotkey too, so it works while the HUD is in the background.
    //
    // This used to focus the text composer *first* and then fire Talk.start()
    // without awaiting it. The visible result was always chat mode: the caret
    // landed in the input box, and when the microphone failed — permission never
    // granted, device unplugged, no mediaDevices in the shell — nothing said so,
    // because the promise was dropped on the floor. Speech now comes first, the
    // attempt is awaited, and the composer is a last resort that explains itself.
    let summoning = false;
    async function summonSpeech() {
      if (summoning) return false;
      summoning = true;
      try { return await doSummon(); } finally { summoning = false; }
    }

    async function doSummon() {
      try { window.focus(); } catch (_) {}
      if (Talk.on) { toast('כבר מקשיב — פשוט דבר', 'good', 2600); return true; }
      if (Rec.recording) { toast('כבר מקליט — דבר', 'good', 2600); return true; }

      if (Talk.available) {
        let ok = false;
        try { ok = await Talk.start(); } catch (_) { ok = false; }
        if (ok) return true;
      }
      // Continuous conversation refused the microphone; try one push-to-talk take
      // before giving up on speech entirely.
      let rec = false;
      try { rec = await Rec.start(); } catch (_) { rec = false; }
      if (rec) return true;

      const el = document.getElementById('input');
      if (el) { el.focus(); if (el.select) el.select(); }
      toast('המיקרופון לא נפתח — אפשר לכתוב כאן במקום', 'warn', 6000);
      return false;
    }

    Mic.init();
    Vision.init();

    // hotkeys from Electron main
    if (win && win.onHotkey) win.onHotkey(h => {
      if (h.name === 'push-to-talk') { Rec.recording ? Rec.stop() : Rec.start(); }
      if (h.name === 'summon-speech') summonSpeech();
      if (h.name === 'kill-switch') doKill('KILL SWITCH — מקש קיצור');
    });
    if (win && win.onBrainState) win.onBrainState(s => {
      if (!s.up) toast('המוח נפל — מנסה להתחבר מחדש', 'err', 8000);
    });

    document.addEventListener('keydown', e => {
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.code === 'Escape') { e.preventDefault(); doKill('KILL SWITCH'); }
      if (e.key === 'F5' && !e.ctrlKey && !e.metaKey && !e.altKey) { e.preventDefault(); summonSpeech(); }
      if (e.key === 'Escape' && Rec.recording) Rec.stop();
      if (e.key === 'Escape') { const tp = $('#train-panel'); if (tp && !tp.classList.contains('hidden')) Train.close(); }
    });

    // unlock audio on first gesture (browser autoplay policy)
    const unlock = () => { Voice.unlock().then(() => {}); document.removeEventListener('pointerdown', unlock); };
    document.addEventListener('pointerdown', unlock);
  }

  // ══════════════════════════ view: מערכות ⇄ צ׳אט ══════════════════════════
  // A pure *view* switch.  Chat mode hides the instrument rails and gives the
  // conversation the whole window; it does not touch the voice pipeline, the
  // composer, the WebSocket or any shortcut.  Everything that already works in
  // the HUD keeps working here — the mic, the wake word, spoken answers.
  const View = (() => {
    const KEY = 'jarvis.view';
    let cur = 'hud';

    function label() {
      const el = $('#cs-state'); if (!el) return;
      const b = document.body, st = b.dataset.state || 'idle';
      if (st === 'killed')   { el.textContent = 'KILL — כל הפעולות חסומות'; return; }
      if (st === 'offline')  { el.textContent = 'אין קשר למוח — מנסה להתחבר מחדש'; return; }
      if (st === 'booting')  { el.textContent = 'מאתחל…'; return; }
      if (b.classList.contains('talking')) {
        el.textContent = st === 'speaking' ? 'מדבר — המיקרופון מושתק בזמן התשובה'
                       : st === 'thinking' ? 'שומע אותך, מעבד…'
                       : 'מקשיב — פשוט דבר, לא צריך ללחוץ';
        return;
      }
      if (st === 'speaking') { el.textContent = 'מדבר'; return; }
      if (st === 'thinking') { el.textContent = 'חושב…'; return; }
      el.textContent = 'ממתין — כתוב למטה, או לחץ 🗣 שיחה קולית ודבר';
    }

    function apply(next, opts) {
      cur = (next === 'chat') ? 'chat' : 'hud';
      const b = document.body;
      b.classList.toggle('chat-mode', cur === 'chat');

      const vh = $('#view-hud'), vc = $('#view-chat');
      if (vh) { vh.classList.toggle('active', cur === 'hud');  vh.setAttribute('aria-selected', String(cur === 'hud')); }
      if (vc) { vc.classList.toggle('active', cur === 'chat'); vc.setAttribute('aria-selected', String(cur === 'chat')); }
      if (b.dataset) b.dataset.view = cur;

      const cs = $('#cs-talk');
      if (cs) cs.classList.toggle('on', b.classList.contains('talking'));

      try { localStorage.setItem(KEY, cur); } catch (_) {}

      // the transcript is now the main event — keep the latest line in view
      const t = $('#transcript');
      if (t) requestAnimationFrame(() => { t.scrollTop = t.scrollHeight; });
      if (cur === 'chat' && !(opts && opts.silent)) { const i = $('#input'); if (i) i.focus(); }
      label();
    }

    function toggle() { apply(cur === 'chat' ? 'hud' : 'chat'); }

    function init() {
      let saved = null;
      try { saved = localStorage.getItem(KEY); } catch (_) {}
      apply(saved === 'chat' ? 'chat' : 'hud', { silent: true });

      const vh = $('#view-hud'), vc = $('#view-chat');
      if (vh) vh.onclick = () => apply('hud');
      if (vc) vc.onclick = () => apply('chat');

      const xt = $('#cs-exit'); if (xt) xt.onclick = () => apply('hud');
      const tk = $('#cs-talk');
      if (tk) tk.onclick = () => {
        if (typeof Talk !== 'undefined' && Talk.available) { Talk.on ? Talk.stop() : Talk.start(); }
        else { toast('הדפדפן לא נותן גישה למיקרופון', 'err'); }
      };

      // Keep the strip honest without touching the state machine: JARVIS
      // already writes body.talking (Talk.paint) and body[data-state]
      // (recomputeState), so just watch those two attributes.
      try {
        new MutationObserver(() => {
          const cs = $('#cs-talk');
          if (cs) cs.classList.toggle('on', document.body.classList.contains('talking'));
          label();
        }).observe(document.body, { attributes: true, attributeFilter: ['class', 'data-state'] });
      } catch (_) { setInterval(label, 900); }

      // Match on e.code, never e.key: on a Hebrew layout the physical C key
      // reports 'ב' and H reports 'י', so a key-based match would break for
      // exactly the audience this app is written for.  e.code is the physical
      // key and is layout-independent.
      document.addEventListener('keydown', e => {
        if (!(e.ctrlKey || e.metaKey) || !e.shiftKey || e.altKey) return;
        if (e.code === 'KeyC') { e.preventDefault(); apply('chat'); }
        else if (e.code === 'KeyH') { e.preventDefault(); apply('hud'); }
      });
    }

    return { init, apply, toggle, label, get current() { return cur; } };
  })();

  async function doKill(reason) {
    busy.killed = true;
    $('#kill-reason').textContent = reason;
    $('#kill-overlay').classList.remove('hidden');
    recomputeState();
    send({ type: 'kill', reason });
    toast('KILL SWITCH — כל הפעולות חסומות', 'err', 7000);
  }

  function submit() {
    const input = $('#input');
    const raw = input.value.trim();
    if (!raw) return;
    if (!linkUp()) { toast('אין קשר למוח — מנסה שוב…', 'err'); }
    const prefix = MODES[mode].prefix || '';
    const text = prefix ? (raw.startsWith(prefix) ? raw : prefix + raw) : raw;
    addMsg('user', esc(raw), mode === 'chat' ? 'אדוני' : mode.toUpperCase());
    addTyping();
    busy.thinking++; recomputeState(); markBusy('jarvis', 9000);
    send({ type: 'user_text', text, speak: $('#chk-speak').checked });
    input.value = '';
    input.focus();
  }

  // ══════════════════════════ clocks & keepalive ══════════════════════════
  function startClocks() {
    const heb = new Intl.DateTimeFormat('he-IL', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' });
    const tick = () => {
      const d = new Date();
      $('#clock-time').textContent = d.toTimeString().slice(0, 8);
      $('#clock-date').textContent = heb.format(d);
    };
    tick(); setInterval(tick, 1000);
    setInterval(tickAgents, 900);
    setInterval(() => { send({ type: 'ping' }); }, 12000);
    setInterval(() => { api('/api/status').then(renderStatus).catch(() => {}); }, 7000);
    setInterval(syncMemory, 25000);
    // The index only changes when the user asks for a scan, so this is a
    // slow poll for coverage numbers, not a freshness mechanism.
    setInterval(syncRag, 30000);
    syncRag();
    setInterval(pollFallback, 2000);
    setInterval(() => {
      // uptime pill
      api('/api/status').then(s => { $('#pill-uptime').querySelector('b').textContent = clockFmt(s.uptime_s || 0); }).catch(() => {});
    }, 5000);
  }

  async function pollFallback() {
    if (wsAlive) return;                      // the socket already streams all this
    try {
      const r = await api('/api/events?n=60');
      restOk = true;
      (r.events || []).forEach(ev => {
        if ((ev.id || 0) > lastEventId) { lastEventId = ev.id || lastEventId; pushEvent(ev); }
      });
      const p = await api('/api/permissions');
      (p.pending || []).forEach(renderPermission);
      recomputeState();
    } catch (_) { /* brain still starting */ }
  }

  // ══════════════════════════ boot ══════════════════════════
  async function main() {
    HUD.init();
    View.init();
    Voice.startMeter();
    Voice.onState(on => {
      busy.speaking = on ? Date.now() : 0;
      recomputeState();
      if (on) { unmuteSent = false; }
      else if (!unmuteSent && Talk.on) {
        // Playback is over for real — lift the server-side echo mute now instead
        // of waiting out its wall-clock guess.
        unmuteSent = true;
        try {
          if (ws && ws.readyState === 1) send({ type: 'voice_unmute' });
          else fetch(HTTP + '/api/voice', { method: 'POST',
                 headers: { 'Content-Type': 'application/json' },
                 body: JSON.stringify({ action: 'unmute', sid: Talk.sid }) }).catch(() => {});
        } catch (_) {}
      }
    });
    bindUI();
    renderAgents();
    $('#perm-queue').innerHTML = '<div class="perm-empty">אין בקשות ממתינות — המערכת במצב מאובטח</div>';

    // where is the brain?
    if (window.jarvis && window.jarvis.getBrainUrl) {
      try { HTTP = await window.jarvis.getBrainUrl(); } catch (_) { HTTP = ''; }
      try { $('#build-tag').textContent = 'MK-VII · ' + (await window.jarvis.getAppVersion()); } catch (_) {}
    }
    if (!HTTP) HTTP = (location.protocol === 'file:') ? 'http://127.0.0.1:8756' : location.origin;
    WS = HTTP.replace(/^http/, 'ws') + '/ws';

    // The bundle is served by the brain, not shipped next to the page: in
    // Electron the HUD is loaded from disk, where a relative link points nowhere,
    // and a static zip nobody built was a 404 wearing a feature's clothes.
    const dl = $('#btn-download');
    if (dl) {
      dl.href = HTTP + '/api/download';
      dl.onclick = () => {
        dl.classList.add('busy');
        toast('בונה את JARVIS-full.zip…', 'good', 4000);
        setTimeout(() => dl.classList.remove('busy'), 3000);
      };
      // Say the truth about the size before a byte moves.
      api('/api/bundle').then(b => {
        if (!b || !b.ok) return;
        const areas = (b.by_area || []).slice(0, 4)
          .map(a => `${a.area} ${a.mb}MB`).join(' · ');
        dl.title = `JARVIS-full.zip · ${b.files} קבצים · ${b.mb}MB מקור`
                 + `\n${areas}\nלא כולל data/ (המצב הפרטי של המכונה הזאת)`;
      }).catch(() => {});
    }

    startClocks();
    wireSecurityPanels();
    wireScreenRead();
    wireModes();
    wirePalette();
    wireTranscript();
    wireRag();
    wirePanels();
    wireBootSkip();
    recomputeState();
    connect();

    // wait for the brain, then run the POST sequence — unless the operator has
    // already opened the interface, in which case this must not hold it hostage
    for (let i = 0; i < 120 && !bootSkipped; i++) {
      try { const s = await api('/api/status'); renderStatus(s); if (s.telemetry) renderTelemetry(s.telemetry); break; }
      catch (_) { await new Promise(r => setTimeout(r, 800)); }
    }
    await runBoot();
    try {
      const h = await api('/api/history?n=60');
      (h.turns || []).forEach(t => {
        addMsg('user', esc(t.user), 'אדוני');
        renderTurn(t);
      });
    } catch (_) {}
    try {
      const ev = await api('/api/events?n=60');
      (ev.events || []).forEach(pushEvent);
    } catch (_) {}
    $('#input').focus();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', main);
  else main();
})();
