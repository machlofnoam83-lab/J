/**
 * Advanced HUD - v2.2
 * מיושר במלואו ל-index.html - מצבי חלון (center/side/orb), טאב מודל, בלי קריסות
 */

class AdvancedHUD {
    constructor() {
        this.ws = null;
        this.wsUrl = this.detectWSUrl();
        this.connected = false;
        this.proposals = [];
        this.currentAudio = null;
        this.reconnectAttempts = 0;
        this._thinkingEl = null;

        this.initElements();
        this.bindEvents();
        this.connectWS();
        this.startClocks();
        this.startMetrics();
        this.initMode();
    }

    /** מצב חלון (center/side/orb) - מסנכרן את ה־CSS עם Electron */
    setMode(mode) {
        if (!mode) return;
        document.body.classList.remove('mode-center', 'mode-side', 'mode-orb');
        document.body.classList.add(`mode-${mode}`);
        console.log('[HUD] Mode:', mode);
    }

    initMode() {
        if (window.adielAPI?.getCurrentMode) {
            window.adielAPI.getCurrentMode().then(m => m && this.setMode(m)).catch(() => {});
        }
    }

    $(id) { return document.getElementById(id); }

    detectWSUrl() {
        const host = window.location.hostname;
        const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        if (!host) return 'ws://localhost:8765/ws'; // file:// mode (Electron loadFile)
        if (host.includes('.e2b.app') && host.includes('-')) {
            const parts = host.split('-');
            const rest = parts.slice(1).join('-');
            return `${proto}//8765-${rest}/ws`;
        }
        return `${proto}//${host}:8765/ws`;
    }

    backendUrl() {
        return this.wsUrl
            .replace('ws://', 'http://')
            .replace('wss://', 'https://')
            .replace(/\/ws$/, '');
    }

    initElements() {
        this.els = {
            mainStatusDot: this.$('mainDot'),
            mainStatusText: this.$('mainStatus'),
            memoryCount: this.$('memoryCount'),
            vocabCount: this.$('vocabCount'),
            aiVal: this.$('aiVal'),
            statusLabel: this.$('statusLabel'),
            visualizer: this.$('visualizer'),
            conversationInner: this.$('conversationInner'),
            convBadge: this.$('convBadge'),
            interactionsBadge: this.$('interactionsBadge'),
            screenPreview: this.$('screenPreview'),
            previewImage: this.$('previewImage'),
            previewContext: this.$('previewContext'),
            screenPlaceholder: this.$('screenPlaceholder'),
            proposalsList: this.$('proposalsList'),
            proposalsBadge: this.$('proposalsBadge'),
            tasksMiniList: this.$('tasksMiniList'),
            framesGrid: this.$('framesGrid'),
            dictCount: this.$('dictCount'),
            micList: this.$('micList'),
            profileInfo: this.$('profileInfo'),
            profileName: this.$('profileName'),
            projectsCount: this.$('projectsCount'),
            interactionsCount: this.$('interactionsCount'),
            memoryContext: this.$('memoryContext'),
            backendStatus: this.$('backendStatus'),
            trueAIStatus: this.$('trueAIStatus'),
            healthInfo: this.$('healthInfo'),
            textInput: this.$('textInput'),
            dictSearch: this.$('dictSearch'),
            dictResult: this.$('dictResult'),
            connDot: this.$('connDot'),
            connText: this.$('connText'),
            avatar: this.$('avatarImg'),
            orbView: this.$('orbView'),
            modelWords: this.$('modelWords'),
            modelVocab: this.$('modelVocab'),
            modelIntents: this.$('modelIntents'),
            memoryConvs: this.$('memoryConvs'),
            nbIntents: this.$('nbIntents'),
            nbFeatures: this.$('nbFeatures'),
        };
    }

    bindEvents() {
        // Window controls
        this.$('btnSide')?.addEventListener('click', () => this.switchMode('side'));
        this.$('btnCenter')?.addEventListener('click', () => this.switchMode('center'));
        this.$('btnOrb')?.addEventListener('click', () => this.switchMode('orb'));
        this.$('btnMin')?.addEventListener('click', () => window.adielAPI ? window.adielAPI.minimize() : null);
        this.$('btnClose')?.addEventListener('click', () => window.adielAPI ? window.adielAPI.close() : window.close());

        // Input
        this.$('btnSend')?.addEventListener('click', () => this.sendText());
        this.els.textInput?.addEventListener('keydown', (e) => { if (e.key === 'Enter') this.sendText(); });

        // Quick actions (כולל גרסת מצב-צד עם ה־2)
        this.$('btnWake')?.addEventListener('click', () => this.triggerWake());
        this.$('btnWake2')?.addEventListener('click', () => this.triggerWake());
        this.$('btnScreen')?.addEventListener('click', () => this.askScreen());
        this.$('btnScreen2')?.addEventListener('click', () => this.askScreen());
        this.$('btnTrain')?.addEventListener('click', () => this.trainAI());
        this.$('btnTrain2')?.addEventListener('click', () => this.trainAI());
        this.$('btnFix')?.addEventListener('click', () => this.fixSystem());
        this.$('btnClear')?.addEventListener('click', () => this.clearConversation());
        this.$('btnClear2')?.addEventListener('click', () => this.clearConversation());
        this.$('btnFrames')?.addEventListener('click', () => this.switchTab('tasks'));
        this.$('btnCenter2')?.addEventListener('click', () => this.switchMode('center'));
        this.$('btnTrainModel')?.addEventListener('click', () => this.trainAI());

        // Orb - לחיצה מחזירה למרכז ומעירה
        this.els.orbView?.addEventListener('click', () => {
            this.switchMode('center');
            this.triggerWake();
        });

        // Dictionary search
        this.$('btnDictSearch')?.addEventListener('click', () => this.searchDictionary());
        this.els.dictSearch?.addEventListener('keydown', (e) => { if (e.key === 'Enter') this.searchDictionary(); });

        // Tabs (v2: .tab / .pane)
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                const tabName = e.currentTarget.getAttribute('data-tab');
                if (tabName) this.switchTab(tabName);
            });
        });

        // Task cards
        document.querySelectorAll('.task-card[data-task]').forEach(card => {
            card.addEventListener('click', (e) => {
                const task = e.currentTarget.getAttribute('data-task');
                if (task) this.executeTask(task);
            });
        });

        if (window.adielAPI) {
            window.adielAPI.onModeChanged((mode) => this.setMode(mode));
            window.adielAPI.onSimulateWake?.(() => this.onWake('קיצור מקלדת'));
            window.adielAPI.onBackendStatus?.((st) => {
                if (st && st.running === false) this.updateConn('disconnected');
            });
        }

        // טעינת פריימים אחרי שה-DOM מוכן
        setTimeout(() => this.loadFrames(), 800);
    }

    switchMode(mode) {
        if (window.adielAPI) window.adielAPI.switchMode(mode);
    }

    switchTab(tabName) {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelector(`.tab[data-tab="${tabName}"]`)?.classList.add('active');
        document.querySelectorAll('.pane').forEach(p => p.classList.remove('active'));
        this.$(`pane-${tabName}`)?.classList.add('active');
    }

    startClocks() {
        // השעון הראשי מנוהל ב-index.html; כאן רק AI metric
    }

    startMetrics() {
        setInterval(() => {
            if (this.els.aiVal) {
                this.els.aiVal.textContent = (92 + Math.floor(Math.random() * 7)) + '%';
            }
        }, 2500);
    }

    // ================= WebSocket =================
    connectWS() {
        console.log('[AdvancedHUD] Connecting to', this.wsUrl);
        this.updateConn('connecting');
        try {
            this.ws = new WebSocket(this.wsUrl);
                this.ws.onopen = () => {
                console.log('[AdvancedHUD] Connected');
                this.connected = true;
                this.reconnectAttempts = 0;
                this.updateConn('connected');
                this.addMessage('assistant', '🚀 v2.2 מחובר! AdielMind — מודל שפה ביתי שלומד מכל שיחה, מסווג כוונות נלמד, וזיכרון BM25. הכול מאפס, בלי ענן ובלי API keys. איך אני, בוס?');
                this.loadMics();
                this.loadProfile();
                this.loadHealth();
                this.loadFrames();
                this.loadModelStatus();
                if (this._pingInterval) clearInterval(this._pingInterval);
                this._pingInterval = setInterval(() => {
                    if (this.ws?.readyState === 1) this.ws.send(JSON.stringify({ type: 'ping' }));
                }, 30000);
            };
            this.ws.onmessage = (e) => {
                try { this.handleMessage(JSON.parse(e.data)); } catch (err) { console.error('[HUD] Bad message', err); }
            };
            this.ws.onclose = () => {
                this.connected = false;
                this.updateConn('disconnected');
                this.scheduleReconnect();
            };
            this.ws.onerror = () => this.updateConn('disconnected');
        } catch (e) {
            console.error('[AdvancedHUD] WS create failed', e);
            this.updateConn('disconnected');
            this.scheduleReconnect();
        }
    }

    scheduleReconnect() {
        this.reconnectAttempts++;
        const delay = Math.min(3000 * this.reconnectAttempts, 15000);
        setTimeout(() => this.connectWS(), delay);
    }

    updateConn(state) {
        const colors = { connected: 'var(--green)', connecting: 'var(--orange)', disconnected: 'var(--red)' };
        if (this.els.connDot) {
            const c = colors[state] || 'var(--red)';
            this.els.connDot.style.background = c;
            this.els.connDot.style.boxShadow = `0 0 8px ${c}`;
        }
        if (this.els.connText) this.els.connText.textContent = state === 'connected' ? 'ONLINE' : state === 'connecting' ? 'מתחברת...' : 'מנותק';
        if (this.els.backendStatus) {
            this.els.backendStatus.textContent = state === 'connected' ? '● מחובר' : '○ מנותק';
            this.els.backendStatus.style.color = state === 'connected' ? 'var(--green)' : 'var(--red)';
        }
    }

    handleMessage(msg) {
        console.log('[AdvancedHUD] MSG', msg.type);
        switch (msg.type) {
            case 'connected': this.updateConn('connected'); break;
            case 'wake_detected': this.onWake(msg.text); break;
            case 'listening': this.setListening(true, msg.message); break;
            case 'stt_state': if (msg.state === 'recording') this.setListening(true, 'מקשיבה... דבר'); break;
            case 'stt_result':
                if (!msg.empty && msg.text) { this.addMessage('user', msg.text); this.setListening(false); }
                else { this.setListening(false, 'לא שמעתי'); setTimeout(() => this.setIdle(), 2000); }
                break;
            case 'brain_response':
                this.hideThinking();
                if (msg.assistant_text) {
                    this.addMessage('assistant', msg.assistant_text, this.modelLabel(msg.model_used));
                    if (msg.screen_image) this.showScreen(msg.screen_image, msg.screen_context);
                    if (msg.proposals) msg.proposals.forEach(p => this.addProposal(p));
                    if (msg.self_updates) msg.self_updates.forEach(p => this.addProposal(p));
                    if (msg.user_profile) this.updateProfileUI(msg.user_profile);
                }
                break;
            case 'model_trained':
                this.addMessage('assistant', `🧠 ${msg.message || 'המודל אומן מחדש!'}`);
                this.loadModelStatus();
                break;
            case 'assistant_speaking':
                this.hideThinking();
                this.setSpeaking(true, msg.text, msg.audio_base64);
                if (!msg.audio_base64) setTimeout(() => this.setSpeaking(false), Math.max(2000, (msg.text || '').length * 70));
                break;
            case 'learning_proposal':
            case 'self_update_proposal': this.addProposal(msg.proposal); break;
            case 'proposal_approved': this.addMessage('assistant', `✅ ${msg.message || 'אושר'}`); this.removeProposal(msg.proposal_id); break;
            case 'proposal_rejected': this.addMessage('assistant', `❌ ${msg.message || 'נדחה'}`); this.removeProposal(msg.proposal_id); break;
            case 'self_update_approved': this.addMessage('assistant', `🚀 ${msg.message || 'עודכנתי!'}`); this.removeProposal(msg.update_id); break;
            case 'self_update_rejected': this.addMessage('assistant', `👌 ${msg.message || 'נדחה'}`); this.removeProposal(msg.update_id); break;
            case 'hud_command': this.handleHUDCommand(msg.command); break;
            case 'mic_changed': this.addMessage('assistant', `🎙️ ${msg.message || 'מיקרופון הוחלף'}`); break;
            case 'task_result': this.addTaskResult(msg); break;
            case 'routine_result': this.addMessage('assistant', `⏰ ${msg.message || ''}`); break;
            case 'screen_data': if (msg.image) this.showScreen(msg.image, msg.context); break;
            case 'profile_data': this.updateProfileUI(msg.profile); break;
            case 'proposal_approved_ack': break;
            case 'pong': break;
            case 'error': this.hideThinking(); this.addMessage('assistant', `⚠️ ${msg.message || 'שגיאה'}`); break;
        }
    }

    handleHUDCommand(cmd) {
        const action = typeof cmd === 'string' ? cmd : (cmd && cmd.action);
        if (action === 'dock' || action === 'side') this.switchMode('side');
        else if (action === 'center' || action === 'show') this.switchMode('center');
        else if (action === 'hide' || action === 'orb') this.switchMode('orb');
    }

    // ================= מצבי מערכת =================
    onWake(text) {
        this.setListening(true, `כן בוס? שמעתי "${text || ''}"`);
    }

    setListening(isListening, label) {
        const sl = this.els.statusLabel;
        if (isListening) {
            sl?.classList.add('listening');
            if (sl) sl.textContent = label || 'מקשיבה...';
            this.els.visualizer?.classList.add('active');
            if (this.els.mainStatusText) this.els.mainStatusText.textContent = 'LISTENING';
            this.els.mainStatusDot?.classList.add('listening');
            if (this.els.avatar) { this.els.avatar.classList.remove('speaking'); this.els.avatar.classList.add('listening'); }
            if (this.els.orbView) { this.els.orbView.classList.remove('speaking'); this.els.orbView.classList.add('listening'); }
        } else {
            sl?.classList.remove('listening');
            this.els.visualizer?.classList.remove('active');
            this.els.mainStatusDot?.classList.remove('listening');
            if (label && sl) sl.textContent = label;
            if (!sl?.classList.contains('speaking')) this.setIdle();
            this.els.avatar?.classList.remove('listening');
            this.els.orbView?.classList.remove('listening');
        }
    }

    setSpeaking(isSpeaking, text, audioB64) {
        const sl = this.els.statusLabel;
        if (isSpeaking) {
            sl?.classList.add('speaking');
            if (sl) sl.textContent = text ? `מדברת: ${text.slice(0, 35)}...` : 'מדברת...';
            this.els.visualizer?.classList.add('active');
            if (this.els.mainStatusText) this.els.mainStatusText.textContent = 'SPEAKING';
            if (audioB64) this.playAudio(audioB64);
            if (this.els.avatar) { this.els.avatar.classList.remove('listening'); this.els.avatar.classList.add('speaking'); }
            if (this.els.orbView) { this.els.orbView.classList.remove('listening'); this.els.orbView.classList.add('speaking'); }
        } else {
            sl?.classList.remove('speaking');
            this.els.visualizer?.classList.remove('active');
            this.els.avatar?.classList.remove('speaking');
            this.els.orbView?.classList.remove('speaking');
            this.setIdle();
        }
    }

    setIdle() {
        const sl = this.els.statusLabel;
        if (sl && !sl.classList.contains('listening') && !sl.classList.contains('speaking')) {
            sl.textContent = 'מאזינה למילת הפעלה...';
            if (this.els.mainStatusText) this.els.mainStatusText.textContent = 'SYSTEMS NOMINAL';
        }
    }

    playAudio(b64) {
        try {
            if (this.currentAudio) { this.currentAudio.pause(); this.currentAudio = null; }
            const src = b64.startsWith('data:') ? b64 : `data:audio/mpeg;base64,${b64}`;
            const audio = new Audio(src);
            this.currentAudio = audio;
            audio.onended = () => this.setSpeaking(false);
            audio.onerror = () => this.setSpeaking(false);
            const p = audio.play();
            if (p) p.catch(() => this.setSpeaking(false));
        } catch (e) {
            console.error('[HUD] playAudio failed', e);
            this.setSpeaking(false);
        }
    }

    // ================= שיחה =================
    modelLabel(modelUsed) {
        const map = {
            'adielmind-lm': '🧠 AdielMind · נוצר בזמן אמת',
            'template-varied': '🧠 AdielMind · תבנית חכמה',
            'local-rules': '⚡ כלל מקומי · מיידי',
            'jarvis-team': '🤖 צוות JARVIS',
            'identity': '🧠 AdielMind · זהות',
            'screen-context': '🖥️ ניתוח מסך חי',
        };
        return map[modelUsed] || (modelUsed ? `🧠 ${modelUsed}` : '');
    }

    addMessage(role, text, meta) {
        const inner = this.els.conversationInner;
        if (!inner || !text) return;
        const time = new Date().toLocaleTimeString('he-IL', { hour: '2-digit', minute: '2-digit' });
        const div = document.createElement('div');
        div.className = `msg ${role}`;
        const avatarLabel = role === 'user' ? 'אתה' : 'AJ';
        const metaHtml = meta ? `<div class="msg-meta">${this.escapeHtml(meta)}</div>` : '';
        div.innerHTML = `<div class="avatar">${avatarLabel}</div><div class="bubble"><div class="msg-text">${this.escapeHtml(text)}</div>${metaHtml}<div class="msg-time">${time}</div></div>`;
        inner.appendChild(div);
        inner.scrollTop = inner.scrollHeight;
        this.updateMsgBadges();
    }

    updateMsgBadges() {
        const inner = this.els.conversationInner;
        if (!inner) return;
        if (this.els.convBadge) this.els.convBadge.textContent = inner.children.length;
        while (inner.children.length > 60) inner.removeChild(inner.firstChild);
    }

    showThinking() {
        this.hideThinking();
        const inner = this.els.conversationInner;
        if (!inner) return;
        const div = document.createElement('div');
        div.className = 'msg assistant';
        div.id = 'thinkingBubble';
        div.innerHTML = `<div class="avatar">AJ</div><div class="bubble"><div class="thinking-dots"><span></span><span></span><span></span></div></div>`;
        inner.appendChild(div);
        inner.scrollTop = inner.scrollHeight;
        this._thinkingEl = div;
    }

    hideThinking() {
        if (this._thinkingEl) { this._thinkingEl.remove(); this._thinkingEl = null; }
        else this.$('thinkingBubble')?.remove();
        this.updateMsgBadges();
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = String(text);
        return div.innerHTML.replace(/\n/g, '<br>');
    }

    sendText() {
        const input = this.els.textInput;
        if (!input) return;
        const text = input.value.trim();
        if (!text) return;

        const screenKeywords = ['מסך', 'רואה', 'שגיאה', 'קוד', 'מה פתוח', 'תסרוק'];
        const needsScreen = screenKeywords.some(kw => text.includes(kw));

        if (this.ws?.readyState === 1) {
            const taskKeywords = ['תקנה', 'תזמין', 'תחקור', 'תמלא', 'תבדוק מייל', 'תמצא זמן', 'תארגן', 'תכין דוח', 'תריץ שגרה'];
            this.addMessage('user', text);
            this.showThinking();
            if (taskKeywords.some(kw => text.includes(kw))) {
                this.executeTask(text);
            } else {
                this.ws.send(JSON.stringify({ type: 'text', text: text, with_screen: needsScreen }));
            }
            input.value = '';
        } else {
            this.addMessage('assistant', '⚠️ לא מחוברת ל-Backend. תפעיל את main.py ואנסה שוב.');
        }
    }

    triggerWake() {
        if (this.ws?.readyState === 1) this.ws.send(JSON.stringify({ type: 'manual_wake' }));
        if (window.adielAPI) window.adielAPI.simulateWake();
        this.onWake('ידני');
    }

    askScreen() {
        const q = 'מה את רואה במסך?';
        if (this.ws?.readyState === 1) {
            this.addMessage('user', q);
            this.showThinking();
            this.ws.send(JSON.stringify({ type: 'text', text: q, with_screen: true }));
        }
    }

    // ================= משימות (Super Agent) =================
    async executeTask(taskText) {
        try {
            const res = await fetch(`${this.backendUrl()}/tasks/execute`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: taskText })
            });
            const data = await res.json();
            this.hideThinking();
            this.addMessage('assistant', `🚀 ${data.message || 'מבצעת משימה...'}`);
            if (data.result) {
                const details = JSON.stringify(data.result, null, 2).substring(0, 600);
                this.addMessage('assistant', `📋 ${details}...`);
            }
            this.addTaskToMini(taskText, data);
        } catch (e) {
            console.error('[HUD] executeTask failed', e);
            if (this.ws?.readyState === 1) {
                this.ws.send(JSON.stringify({ type: 'text', text: taskText, with_screen: false }));
            } else {
                this.hideThinking();
                this.addMessage('assistant', `❌ המשימה נכשלה: ${e.message}`);
            }
        }
    }

    addTaskToMini(text, result) {
        if (!this.els.tasksMiniList) return;
        const div = document.createElement('div');
        div.className = 'task-history-item';
        div.innerHTML = `<div class="task-history-text">🚀 ${this.escapeHtml(text)}</div><div class="task-history-result">${this.escapeHtml((result.message || '').substring(0, 80))}</div><div class="task-history-time">${result.success ? '✅' : '❌'} ${new Date().toLocaleTimeString('he-IL')}</div>`;
        this.els.tasksMiniList.insertBefore(div, this.els.tasksMiniList.firstChild);
        while (this.els.tasksMiniList.children.length > 10) {
            this.els.tasksMiniList.removeChild(this.els.tasksMiniList.lastChild);
        }
    }

    addTaskResult(msg) {
        this.hideThinking();
        this.addMessage('assistant', `🚀 ${msg.message || ''}`);
        this.addTaskToMini(msg.task_type || 'משימה', { message: msg.message, success: msg.success });
    }

    // ================= הצעות למידה =================
    addProposal(proposal) {
        if (!this.els.proposalsList || !proposal) return;
        if (proposal.id && this.$(`proposal-${proposal.id}`)) return;

        const placeholder = this.els.proposalsList.querySelector('.placeholder-holo');
        if (placeholder) this.els.proposalsList.innerHTML = '';

        const card = document.createElement('div');
        card.className = 'proposal-card';
        card.id = `proposal-${proposal.id || Math.random().toString(36).slice(2)}`;
        const risk = proposal.risk || 'low';
        const riskLabel = { low: 'נמוך', medium: 'בינוני', high: 'גבוה' }[risk] || risk;

        card.innerHTML = `
            <div class="proposal-title">${this.escapeHtml(proposal.title || (proposal.description_he || 'הצעה').substring(0, 40))}</div>
            <div class="proposal-description">${this.escapeHtml(proposal.description_he || '')}</div>
            <div class="proposal-meta"><span class="proposal-tag">${this.escapeHtml(proposal.type || 'learning')}</span><span class="proposal-tag risk-${risk}">${riskLabel}</span></div>
            ${proposal.before_example ? `<div class="proposal-explanation">לפני: ${this.escapeHtml(proposal.before_example)}<br>אחרי: ${this.escapeHtml(proposal.after_example || '')}</div>` : ''}
            ${proposal.explanation_for_user ? `<div class="proposal-explanation">${this.escapeHtml(String(proposal.explanation_for_user).substring(0, 200))}</div>` : ''}
            <div class="proposal-actions">
                <button class="proposal-btn approve">✅ אשר</button>
                <button class="proposal-btn reject">❌ דחה</button>
            </div>
        `;

        const isSelfUpdate = (proposal.type || '').includes('self_update');
        card.querySelector('.approve')?.addEventListener('click', () => this.approveProposal(proposal.id, isSelfUpdate ? 'self_update' : 'learning'));
        card.querySelector('.reject')?.addEventListener('click', () => this.rejectProposal(proposal.id, isSelfUpdate ? 'self_update' : 'learning'));

        this.els.proposalsList.appendChild(card);
        this.proposals.push(proposal);
        if (this.els.proposalsBadge) this.els.proposalsBadge.textContent = this.proposals.length;

        this.switchTab('proposals');
        this.addMessage('assistant', `📚 ${proposal.description_he || 'הצעה חדשה'} - ממתין לאישורך בטאב "הצעות", בוס`);
    }

    removeProposal(id) {
        this.$(`proposal-${id}`)?.remove();
        this.proposals = this.proposals.filter(p => p.id !== id);
        if (this.els.proposalsBadge) this.els.proposalsBadge.textContent = this.proposals.length;
        if (this.els.proposalsList && this.proposals.length === 0) {
            this.els.proposalsList.innerHTML = '<div class="placeholder-holo">אין הצעות ממתינות ✨</div>';
        }
    }

    approveProposal(id, type) {
        if (this.ws?.readyState !== 1) return;
        const msgType = type === 'self_update' ? 'approve_self_update' : 'approve_proposal';
        this.ws.send(JSON.stringify({ type: msgType, id: id }));
        const card = this.$(`proposal-${id}`);
        if (card) {
            card.querySelectorAll('button').forEach(b => b.disabled = true);
            const btn = card.querySelector('.approve');
            if (btn) btn.textContent = '⏳ מאשר...';
        }
    }

    rejectProposal(id, type) {
        if (this.ws?.readyState !== 1) return;
        const msgType = type === 'self_update' ? 'reject_self_update' : 'reject_proposal';
        this.ws.send(JSON.stringify({ type: msgType, id: id }));
        const card = this.$(`proposal-${id}`);
        if (card) {
            card.querySelectorAll('button').forEach(b => b.disabled = true);
            const btn = card.querySelector('.reject');
            if (btn) btn.textContent = '⏳ דוחה...';
        }
    }

    // ================= נתונים מהשרת =================
    async apiGet(path) {
        const res = await fetch(`${this.backendUrl()}${path}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
    }

    async loadMics() {
        const listEl = this.els.micList;
        if (!listEl) return;
        listEl.innerHTML = '<div class="loading">🎙️ טוען מיקרופונים...</div>';
        try {
            const data = await this.apiGet('/audio/devices');
            if (!data.inputs || data.inputs.length === 0) {
                listEl.innerHTML = '<div class="loading">לא נמצאו מיקרופונים. אפשר לכתוב במקלדת!</div>';
                return;
            }
            listEl.innerHTML = '';
            data.inputs.forEach(mic => {
                const div = document.createElement('div');
                div.className = `mic-option ${mic.selected ? 'selected' : ''}`;
                div.innerHTML = `<div class="mic-name">${this.escapeHtml(mic.name || 'מיקרופון')}</div><div class="mic-details">ערוצים: ${mic.input_channels ?? '-'} ${mic.is_default_input ? '<span class="mic-tag default">ברירת מחדל</span>' : ''} ${mic.selected ? '<span class="mic-tag">נבחר ✓</span>' : ''}</div>`;
                div.addEventListener('click', () => this.selectMic(mic.id));
                listEl.appendChild(div);
            });
        } catch (e) {
            listEl.innerHTML = `<div class="loading">🎙️ זמין רק כשה-Backend רץ עם sounddevice</div>`;
        }
    }

    async selectMic(id) {
        try {
            const data = await this.apiPost(`/audio/devices/input/${id}`);
            if (data.success) {
                this.addMessage('assistant', `✅ ${data.message}`);
                this.loadMics();
            } else {
                this.addMessage('assistant', `❌ ${data.error || 'נכשל'}`);
            }
        } catch (e) {
            this.addMessage('assistant', `❌ שגיאה: ${e.message}`);
        }
    }

    async apiPost(path) {
        const res = await fetch(`${this.backendUrl()}${path}`, { method: 'POST' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
    }

    async loadProfile() {
        try {
            const data = await this.apiGet('/profile');
            const profile = data.profile || {};
            if (this.els.profileName) this.els.profileName.textContent = profile.name || 'בוס';
            if (this.els.projectsCount) this.els.projectsCount.textContent = (profile.projects || []).length;
            if (this.els.interactionsCount) this.els.interactionsCount.textContent = profile.interaction_count || 0;
            if (this.els.interactionsBadge) this.els.interactionsBadge.textContent = profile.interaction_count || 0;
            if (this.els.memoryContext) this.els.memoryContext.textContent = data.smart_context || 'דיברנו 0 פעמים';
            if (this.els.memoryCount) this.els.memoryCount.textContent = profile.facts_count || 0;
            if (this.els.vocabCount) this.els.vocabCount.textContent = (data.vocabulary && data.vocabulary.count) || 0;
            if (this.els.trueAIStatus) this.els.trueAIStatus.textContent = `🧠 Vocab ${(data.vocabulary && data.vocabulary.count) || 0}`;

            if (this.els.profileInfo) {
                this.els.profileInfo.innerHTML = `
                    <div>👤 שם: <b>${this.escapeHtml(profile.name || 'לא ידוע')}</b></div>
                    <div>💬 שיחות: <b>${profile.interaction_count || 0}</b></div>
                    <div>📚 מילים שנלמדו: <b>${(data.vocabulary && data.vocabulary.count) || 0}</b></div>
                    <div>🧠 עובדות: <b>${profile.facts_count || 0}</b></div>
                    <div style="margin-top:6px; font-size:10px; opacity:0.7;">${this.escapeHtml(data.smart_context || '')}</div>
                `;
            }
        } catch (e) {
            if (this.els.profileInfo) this.els.profileInfo.innerHTML = '<div class="loading">פרופיל יופיע כשהמוח יעלה</div>';
            console.log('[HUD] Profile load failed', e);
        }
    }

    async loadHealth() {
        try {
            const data = await this.apiGet('/health');
            if (this.els.healthInfo) {
                const checks = data.checks || {};
                let html = '';
                for (const [k, v] of Object.entries(checks)) {
                    html += `<div class="sys-metric"><label>${this.escapeHtml(k)}</label><span class="${v ? 'status-ok' : ''}" style="color:${v ? 'var(--green)' : 'var(--red)'}">${v ? '● OK' : '○ FAIL'}</span></div>`;
                }
                html += `<div class="sys-metric"><label>status</label><span style="color:var(--accent)">${this.escapeHtml(data.status || '?')}</span></div>`;
                this.els.healthInfo.innerHTML = html || 'טוען...';
            }
        } catch (e) {
            if (this.els.healthInfo) this.els.healthInfo.innerHTML = '<div class="loading">Health יופיע כשה-Backend יעלה</div>';
            console.log('[HUD] Health load failed', e);
        }
    }

    // ================= מילון =================
    async searchDictionary() {
        const w = (this.els.dictSearch?.value || '').trim();
        const resEl = this.els.dictResult;
        if (!w || !resEl) return;
        resEl.innerHTML = '🔍 מחפש במילון...';
        try {
            const d = await this.apiGet(`/dictionary/lookup?word=${encodeURIComponent(w)}`);
            if (d.found) {
                resEl.innerHTML = `<b style="color:var(--accent);font-size:14px">${this.escapeHtml(d.word)}</b><br><br><b>פירוש:</b> ${this.escapeHtml(d.meaning || '')}<br><br><b>סוג:</b> ${this.escapeHtml(d.type || '')}<br><b>דוגמה:</b> ${this.escapeHtml(d.example || '')}`;
            } else {
                resEl.innerHTML = `לא נמצא "${this.escapeHtml(w)}". תגיד "תלמד את המילה ${this.escapeHtml(w)}"`;
            }
        } catch (e) {
            resEl.innerHTML = `שגיאה: ${this.escapeHtml(e.message)}`;
        }
    }

    // ================= פריימים (חדש! /frames חי) =================
    async loadFrames() {
        const grid = this.els.framesGrid;
        if (!grid) return;
        try {
            const data = await this.apiGet('/frames');
            const frames = data.frames || [];
            if (!frames.length) throw new Error('no frames');
            grid.innerHTML = '';
            frames.slice(0, 24).forEach(f => this.addFrameCard(grid, f));
            if (this.els.dictCount) this.els.dictCount.textContent = frames.length;
            console.log(`[HUD] Loaded ${frames.length} live frames from /frames`);
        } catch (e) {
            console.log('[HUD] Frames load failed, using fallback', e);
            this.loadFallbackFrames(grid);
        }
    }

    addFrameCard(grid, f) {
        const div = document.createElement('div');
        div.className = 'task-card';
        div.style.textAlign = 'center';
        const dataLine = f.data ? Object.values(f.data).slice(0, 2).join(' · ') : (f.type || '');
        div.innerHTML = `<div style="font-size:24px; margin-bottom:4px;">${this.escapeHtml(f.image || f.icon || '📦')}</div><b style="font-size:11px;">${this.escapeHtml(f.title || '')}</b><small style="font-size:9px; color:var(--dim); display:block; margin-top:2px;">${this.escapeHtml((f.description || '').substring(0, 35))}</small><small style="font-size:8px; color:var(--accent); margin-top:4px; background:rgba(0,212,255,0.1); padding:2px 6px; border-radius:10px; display:inline-block;">${this.escapeHtml(String(dataLine).substring(0, 35))}</small>`;
        div.addEventListener('click', () => {
            const input = this.els.textInput;
            if (input) { input.value = `תראה לי ${f.title}`; input.focus(); }
            if (this.ws?.readyState === 1) {
                this.addMessage('user', `תראה לי ${f.title}`);
                this.showThinking();
                this.ws.send(JSON.stringify({ type: 'text', text: `תראה לי ${f.title}`, with_screen: false }));
            }
        });
        grid.appendChild(div);
    }

    loadFallbackFrames(grid) {
        const fallback = [
            { icon: '♟️', title: 'שעון שחמט', desc: 'טיימר לבן/שחור - אוכל לשחק איתך!' },
            { icon: '📅', title: 'לוח זמנים', desc: 'פגישות היום + חלונות פנויים' },
            { icon: '📧', title: 'מייל', desc: 'לא נקראו ודחופים' },
            { icon: '🕐', title: 'שעון עולם', desc: 'אזורי זמן' },
            { icon: '🌤️', title: 'מזג אוויר', desc: 'תל אביב' },
            { icon: '📰', title: 'חדשות', desc: 'חדשות טק' },
            { icon: '✅', title: 'משימות', desc: 'ToDo היום' },
            { icon: '📆', title: 'יומן', desc: 'פגישות קרובות' },
            { icon: '🔢', title: 'מחשבון', desc: 'חישוב מהיר' },
            { icon: '⏲️', title: 'טיימר', desc: '25 דק פומודורו' },
            { icon: '⏱️', title: 'סטופר', desc: 'מדידת זמן' },
            { icon: '📝', title: 'פתקים', desc: 'פתקים מהירים' },
            { icon: '📁', title: 'קבצים', desc: 'הורדות אחרונות' },
            { icon: '🎵', title: 'מוזיקה', desc: 'Spotify' },
            { icon: '▶️', title: 'יוטיוב', desc: 'סרטונים' },
            { icon: '💻', title: 'קוד', desc: 'VS Code' },
            { icon: '🖥️', title: 'טרמינל', desc: 'פקודות' },
            { icon: '🌐', title: 'דפדפן', desc: 'כרום' },
            { icon: '🗺️', title: 'מפות', desc: 'ניווט' },
            { icon: '🌍', title: 'תרגום', desc: 'עברית ↔ אנגלית' },
            { icon: '📚', title: 'מילון', desc: 'מילים בעברית' },
            { icon: '🛒', title: 'קניות', desc: 'השוואת מחירים' },
            { icon: '✈️', title: 'חופשות', desc: 'דיל משתלם' },
            { icon: '🔬', title: 'מחקר', desc: 'סריקת רשת' },
        ];
        grid.innerHTML = '';
        fallback.forEach(f => {
            const div = document.createElement('div');
            div.className = 'task-card';
            div.style.textAlign = 'center';
            div.innerHTML = `<div style="font-size:22px">${f.icon}</div><b style="font-size:11px">${f.title}</b><small style="font-size:9px;display:block;color:var(--dim)">${f.desc}</small>`;
            grid.appendChild(div);
        });
    }

    // ================= פעולות מערכת =================
    async trainAI() {
        const btn = this.$('btnTrainModel');
        if (btn) { btn.disabled = true; btn.textContent = '⏳ מאמנת... שנייה בוס'; }
        this.addMessage('assistant', '🧠 מאמנת את AdielMind מחדש על כל השיחות והידע... שנייה, בוס');
        try {
            const data = await this.apiPost('/model/train');
            const s = data.stats || {};
            this.addMessage('assistant', `✅ אימון הושלם! ${(s.total_words || 0).toLocaleString('he-IL')} מילים, אוצר ${(s.vocab_size || 0).toLocaleString('he-IL')} מילים, ${s.intent_count || 0} כוונות. ${s.corpus_total ? `(${s.corpus_total} משפטים בקורפוס)` : ''}`);
            this.loadModelStatus();
        } catch (e) {
            this.addMessage('assistant', `❌ האימון נכשל: ${e.message}`);
        } finally {
            if (btn) { btn.disabled = false; btn.textContent = '🧠 אמן אותי מחדש על כל השיחות'; }
        }
    }

    async loadModelStatus() {
        try {
            const data = await this.apiGet('/model/status');
            if (this.els.trueAIStatus) {
                const k = ((data.total_words || 0) / 1000).toFixed(1);
                this.els.trueAIStatus.textContent = `🧠 AdielMind ${k}K מילים`;
            }
            // טאב המודל החדש
            const fmt = (n) => (n || 0).toLocaleString('he-IL');
            if (this.els.modelWords) this.els.modelWords.textContent = fmt(data.total_words);
            if (this.els.modelVocab) this.els.modelVocab.textContent = fmt(data.vocab_size);
            if (this.els.modelIntents) this.els.modelIntents.textContent = fmt(data.intent_count || (data.intents || []).length);
            if (this.els.memoryConvs) this.els.memoryConvs.textContent = fmt(data.memory_conversations);
            if (this.els.nbIntents) this.els.nbIntents.textContent = fmt(data.nb_intents);
            if (this.els.nbFeatures) this.els.nbFeatures.textContent = fmt(data.nb_features);
        } catch (e) {
            console.log('[HUD] Model status failed', e);
        }
    }

    fixSystem() {
        fetch(`${this.backendUrl()}/fix`, { method: 'POST' })
            .then(r => r.json())
            .then(data => {
                this.addMessage('assistant', `🛠️ תיקון: ${(data.fixed || []).length} דברים תוקנו - ${(data.log || []).join(', ') || 'הכל תקין'}`);
                this.loadHealth();
            })
            .catch(e => this.addMessage('assistant', `🛠️ לא הצלחתי להריץ תיקון: ${e.message}`));
    }

    clearConversation() {
        const inner = this.els.conversationInner;
        if (inner) {
            inner.innerHTML = `<div class="msg assistant"><div class="avatar">AJ</div><div class="bubble"><div class="msg-text">שיחה נוקתה, אבל אני זוכרת הכל! v2.2 עם AdielMind מוכנה 🧠</div><div class="msg-time">עכשיו</div></div></div>`;
            this.updateMsgBadges();
        }
    }

    showScreen(b64, context) {
        if (!this.els.previewImage || !b64) return;
        this.els.previewImage.src = b64.startsWith('data:') ? b64 : `data:image/jpeg;base64,${b64}`;
        if (this.els.previewContext && context) this.els.previewContext.textContent = context;
        this.els.screenPreview?.classList.remove('hidden');
        this.els.screenPlaceholder?.classList.add('hidden');
        this.switchTab('screen');
    }

    updateProfileUI(profile) {
        if (!profile) return;
        if (this.els.profileName) this.els.profileName.textContent = profile.name || 'בוס';
        if (this.els.projectsCount) this.els.projectsCount.textContent = (profile.projects || []).length;
        if (this.els.interactionsCount) this.els.interactionsCount.textContent = profile.interaction_count || 0;
        if (this.els.interactionsBadge) this.els.interactionsBadge.textContent = profile.interaction_count || 0;
        if (this.els.memoryCount) this.els.memoryCount.textContent = profile.facts_count || 0;
        if (this.els.vocabCount) this.els.vocabCount.textContent = profile.vocab_learned || 0;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.advancedHUD = new AdvancedHUD();
    console.log('[Advanced HUD] v2.2 initialized - modes, model tab, port-safe backend');
});
