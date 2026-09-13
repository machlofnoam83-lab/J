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
    $('#perm-stats').innerHTML = kv([
      ['רמה', sec.level || '—'],
      ['dry-run', sec.dry_run ? 'כן' : 'לא'],
      ['אושר', sec.allowed || 0],
      ['נחסם', sec.blocked || 0],
      ['סה״כ', sec.total || 0]
    ]);
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
    el.innerHTML =
      `<div class="m-head"><span>${esc(head || (kind === 'user' ? 'אדוני' : 'JARVIS'))}</span>` +
      `<span>${new Date().toTimeString().slice(0, 8)}</span></div>` +
      `<div class="m-body">${body}</div>` +
      (chips ? `<div class="m-meta">${chips}</div>` : '') +
      (opts && opts.trace ? `<div class="m-trace">${opts.trace}</div>` : '');
    wrap.appendChild(el);
    while (wrap.children.length > 90) wrap.removeChild(wrap.firstChild);
    wrap.scrollTop = wrap.scrollHeight;
    return el;
  }

  function addTyping() {
    const wrap = $('#transcript'); if (!wrap) return null;
    const el = document.createElement('div');
    el.className = 'msg bot';
    el.innerHTML = `<div class="m-head"><span>JARVIS</span><span>מעבד…</span></div>` +
                   `<div class="m-body"><span class="typing"><i></i><i></i><i></i></span></div>`;
    wrap.appendChild(el);
    wrap.scrollTop = wrap.scrollHeight;
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
    addMsg(a.grounded ? 'bot' : 'err', esc(a.text || '…'), 'JARVIS',
           { chips, trace: traceHtml(a.trace) });
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

  // ══════════════════════════ boot sequence ══════════════════════════
  async function runBoot() {
    const overlay = $('#boot-overlay'), log = $('#boot-log'), fill = $('#boot-bar-fill');
    overlay.classList.remove('hidden');
    log.innerHTML = '';
    const typeLine = (text, cls) => new Promise(res => {
      const line = document.createElement('div');
      if (cls) line.className = cls;
      log.appendChild(line);
      let i = 0;
      const step = () => {
        line.textContent = text.slice(0, ++i);
        if (i < text.length) setTimeout(step, 7);
        else { log.scrollTop = log.scrollHeight; res(); }
      };
      step();
    });

    await typeLine('J.A.R.V.I.S. — POST SEQUENCE', 'dim');
    await typeLine('Mark VII · offline · zero cloud · zero API keys', 'dim');
    await new Promise(r => setTimeout(r, 180));

    let report = [];
    try { report = (await api('/api/boot')).report || []; }
    catch (err) { await typeLine('✗ no link to the brain: ' + err.message, 'warn'); }

    for (let i = 0; i < report.length; i++) {
      const r = report[i];
      const tag = r.ok ? '[ OK ]' : '[FAIL]';
      await typeLine(`${tag}  ${r.key.padEnd(20, '.')}  ${r.label} · ${String(r.detail).slice(0, 90)} (${r.ms}ms)`,
                     r.ok ? 'ok' : 'warn');
      fill.style.width = Math.round(((i + 1) / report.length) * 100) + '%';
      if (!r.ok) toast(`${r.key}: ${r.detail}`, 'warn', 7000);
    }
    const failed = report.filter(r => !r.ok).length;
    await new Promise(r => setTimeout(r, 140));
    await typeLine(failed ? `— ${failed} subsystem(s) degraded; JARVIS continues with fallbacks —`
                          : '— all subsystems nominal. welcome home, sir. —', failed ? 'warn' : 'ok');
    await new Promise(r => setTimeout(r, 420));
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
      syncPermissions(); syncMemory();
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
        stream = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true }
        });
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

    return { start, stop, get recording() { return recording; } };
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
    input.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }
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

    // hotkeys from Electron main
    if (win && win.onHotkey) win.onHotkey(h => {
      if (h.name === 'push-to-talk') { Rec.recording ? Rec.stop() : Rec.start(); }
      if (h.name === 'kill-switch') doKill('KILL SWITCH — מקש קיצור');
    });
    if (win && win.onBrainState) win.onBrainState(s => {
      if (!s.up) toast('המוח נפל — מנסה להתחבר מחדש', 'err', 8000);
    });

    document.addEventListener('keydown', e => {
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.code === 'Escape') { e.preventDefault(); doKill('KILL SWITCH'); }
      if (e.key === 'Escape' && Rec.recording) Rec.stop();
    });

    // unlock audio on first gesture (browser autoplay policy)
    const unlock = () => { Voice.unlock().then(() => {}); document.removeEventListener('pointerdown', unlock); };
    document.addEventListener('pointerdown', unlock);
  }

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
    Voice.startMeter();
    Voice.onState(on => {
      busy.speaking = on ? Date.now() : 0;
      if (!on) recomputeState();
      else { recomputeState(); }
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

    startClocks();
    recomputeState();
    connect();

    // wait for the brain, then run the POST sequence
    for (let i = 0; i < 120; i++) {
      try { const s = await api('/api/status'); renderStatus(s); if (s.telemetry) renderTelemetry(s.telemetry); break; }
      catch (_) { await new Promise(r => setTimeout(r, 800)); }
    }
    await runBoot();
    try {
      const h = await api('/api/history?n=12');
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
