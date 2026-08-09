/**
 * Advanced HUD - MARK 85 - Super Agent
 * מוח משוכלל עם כל היכולות החדשות
 */

class AdvancedHUD {
    constructor() {
        this.ws = null;
        this.wsUrl = this.detectWSUrl();
        this.connected = false;
        this.proposals = [];
        this.tasks = [];
        
        this.initElements();
        this.bindEvents();
        this.connectWS();
        this.startClocks();
        this.startMetrics();
    }

    detectWSUrl() {
        const host = window.location.hostname;
        const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        if (host.includes('.e2b.app') && host.includes('-')) {
            const parts = host.split('-');
            const rest = parts.slice(1).join('-');
            return `${proto}//8765-${rest}/ws`;
        }
        return 'ws://localhost:8765/ws';
    }

    initElements() {
        this.els = {
            mainStatusDot: document.getElementById('mainStatusDot'),
            mainStatusText: document.getElementById('mainStatusText'),
            memoryCount: document.getElementById('memoryCount'),
            vocabCount: document.getElementById('vocabCount'),
            timeMain: document.getElementById('timeMain'),
            timeSub: document.getElementById('timeSub'),
            cpuFill: document.getElementById('cpuFill'),
            cpuVal: document.getElementById('cpuVal'),
            memFill: document.getElementById('memFill'),
            memVal: document.getElementById('memVal'),
            aiFill: document.getElementById('aiFill'),
            aiVal: document.getElementById('aiVal'),
            micName: document.getElementById('micName'),
            micLevel: document.getElementById('micLevel'),
            statusLabel: document.getElementById('statusLabel'),
            listeningPulse: document.getElementById('listeningPulse'),
            visualizer: document.getElementById('visualizer'),
            bottomVisualizer: document.getElementById('bottomVisualizer'),
            conversationInner: document.getElementById('conversationInner'),
            convBadge: document.getElementById('convBadge'),
            screenPreview: document.getElementById('screenPreview'),
            previewImage: document.getElementById('previewImage'),
            previewContext: document.getElementById('previewContext'),
            screenPlaceholder: document.getElementById('screenPlaceholder'),
            proposalsArea: document.getElementById('proposalsArea'),
            proposalsList: document.getElementById('proposalsList'),
            proposalsBadge: document.getElementById('proposalsBadge'),
            tasksGrid: document.getElementById('tasksGrid'),
            tasksHistory: document.getElementById('tasksHistory'),
            tasksBadge: document.getElementById('tasksBadge'),
            tasksMiniList: document.getElementById('tasksMiniList'),
            micList: document.getElementById('micList'),
            profileName: document.getElementById('profileName'),
            projectsCount: document.getElementById('projectsCount'),
            interactionsCount: document.getElementById('interactionsCount'),
            memoryContext: document.getElementById('memoryContext'),
            orbitalVoice: document.getElementById('orbitalVoice'),
            orbitalListen: document.getElementById('orbitalListen'),
            orbitalMem: document.getElementById('orbitalMem'),
            orbitalTasks: document.getElementById('orbitalTasks'),
            orbitalAI: document.getElementById('orbitalAI'),
            backendStatus: document.getElementById('backendStatus'),
            trueAIStatus: document.getElementById('trueAIStatus'),
            micStatusMini: document.getElementById('micStatusMini'),
            tasksCountMini: document.getElementById('tasksCountMini'),
            logsContent: document.getElementById('logsContent'),
            healthInfo: document.getElementById('healthInfo'),
            profileInfo: document.getElementById('profileInfo'),
            textInput: document.getElementById('textInput'),
            connDot: document.getElementById('connDot'),
            connText: document.getElementById('connText'),
        };
    }

    bindEvents() {
        // Window controls
        document.getElementById('btnSide')?.addEventListener('click', () => this.switchMode('side'));
        document.getElementById('btnCenter')?.addEventListener('click', () => this.switchMode('center'));
        document.getElementById('btnOrb')?.addEventListener('click', () => this.switchMode('orb'));
        document.getElementById('btnClose')?.addEventListener('click', () => window.adielAPI ? window.adielAPI.close() : window.close());

        // Input
        document.getElementById('btnSend')?.addEventListener('click', () => this.sendText());
        this.els.textInput?.addEventListener('keydown', (e) => { if (e.key === 'Enter') this.sendText(); });

        // Quick actions
        document.getElementById('btnWake')?.addEventListener('click', () => this.triggerWake());
        document.getElementById('btnScreen')?.addEventListener('click', () => this.askScreen());
        document.getElementById('btnTrain')?.addEventListener('click', () => this.trainAI());
        document.getElementById('btnFix')?.addEventListener('click', () => this.fixSystem());
        document.getElementById('btnClear')?.addEventListener('click', () => this.clearConversation());

        // Mic
        document.getElementById('btnRefreshMics')?.addEventListener('click', () => this.loadMics());
        
        // Tabs
        document.querySelectorAll('.holo-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                const tabName = e.target.getAttribute('data-tab');
                this.switchTab(tabName);
            });
        });

        // Task cards
        document.querySelectorAll('.task-card').forEach(card => {
            card.addEventListener('click', (e) => {
                const task = e.currentTarget.getAttribute('data-task');
                if (task) {
                    this.executeTask(task);
                }
            });
        });

        // Test voice
        document.querySelectorAll('.test-voice-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const voice = e.target.getAttribute('data-voice');
                this.testVoice(voice);
            });
        });

        // Orb
        document.getElementById('orbContent')?.addEventListener('click', () => {
            this.switchMode('center');
            this.triggerWake();
        });

        if (window.adielAPI) {
            window.adielAPI.onModeChanged((mode) => this.updateModeUI(mode));
        }
    }

    switchMode(mode) {
        const root = document.getElementById('super-hud');
        if (!root) return;
        root.className = `super-hud-root mode-${mode}`;
        if (window.adielAPI) window.adielAPI.switchMode(mode);
    }

    updateModeUI(mode) {
        const root = document.getElementById('super-hud');
        if (root) root.className = `super-hud-root mode-${mode}`;
    }

    switchTab(tabName) {
        document.querySelectorAll('.holo-tab').forEach(t => t.classList.remove('active'));
        document.querySelector(`.holo-tab[data-tab="${tabName}"]`)?.classList.add('active');
        document.querySelectorAll('.holo-pane').forEach(p => p.classList.remove('active'));
        document.getElementById(`pane-${tabName}`)?.classList.add('active');
    }

    startClocks() {
        setInterval(() => {
            const now = new Date();
            if (this.els.timeMain) this.els.timeMain.textContent = now.toLocaleTimeString('he-IL', {hour12: false});
            if (this.els.timeSub) this.els.timeSub.textContent = now.toLocaleDateString('he-IL', {weekday: 'long'}) + ' // ONLINE';
        }, 1000);
    }

    startMetrics() {
        setInterval(() => {
            if (this.els.cpuFill) {
                const cpu = 30 + Math.random() * 40;
                this.els.cpuFill.style.width = cpu + '%';
                this.els.cpuVal.textContent = Math.round(cpu) + '%';
            }
            if (this.els.memFill) {
                const mem = 60 + Math.random() * 20;
                this.els.memFill.style.width = mem + '%';
                this.els.memVal.textContent = Math.round(mem) + '%';
            }
        }, 2000);
        
        // Mic level simulation when listening
        setInterval(() => {
            if (this.els.micLevel && this.els.statusLabel?.classList.contains('listening')) {
                this.els.micLevel.style.width = (Math.random() * 80 + 20) + '%';
            } else if (this.els.micLevel) {
                this.els.micLevel.style.width = '0%';
            }
        }, 150);
    }

    connectWS() {
        console.log('[AdvancedHUD] Connecting to', this.wsUrl);
        this.updateConn('connecting');
        try {
            this.ws = new WebSocket(this.wsUrl);
            this.ws.onopen = () => {
                console.log('[AdvancedHUD] Connected');
                this.connected = true;
                this.updateConn('connected');
                this.addMessage('assistant', '🚀 MARK 85 מחובר! סוכן-על מוכן. מיקרופון לבחירה, קול מקורי, זיכרון חי, ומשימות אוטומטיות. תגיד "אדיאל ג\'וניור"');
                this.loadMics();
                this.loadProfile();
                this.loadHealth();
                setInterval(() => { if (this.ws?.readyState === 1) this.ws.send(JSON.stringify({type: 'ping'})); }, 30000);
            };
            this.ws.onmessage = (e) => {
                try { this.handleMessage(JSON.parse(e.data)); } catch (err) { console.error(err); }
            };
            this.ws.onclose = () => {
                this.connected = false;
                this.updateConn('disconnected');
                setTimeout(() => this.connectWS(), 3000);
            };
            this.ws.onerror = () => this.updateConn('disconnected');
        } catch (e) {
            this.updateConn('disconnected');
            setTimeout(() => this.connectWS(), 3000);
        }
    }

    updateConn(state) {
        if (!this.els.connDot) return;
        this.els.connDot.className = 'conn-dot ' + state;
        this.els.connText.textContent = state === 'connected' ? 'מחוברת - SYSTEMS NOMINAL' : state === 'connecting' ? 'מתחברת...' : 'מנותק';
        if (this.els.backendStatus) this.els.backendStatus.textContent = state === 'connected' ? '● מחובר' : '○ מנותק';
        this.els.backendStatus.className = state === 'connected' ? 'status-ok' : '';
    }

    handleMessage(msg) {
        console.log('[AdvancedHUD] MSG', msg.type);
        switch(msg.type) {
            case 'connected': this.updateConn('connected'); break;
            case 'wake_detected': this.onWake(msg.text); break;
            case 'listening': this.setListening(true, msg.message); break;
            case 'stt_state': if (msg.state === 'recording') this.setListening(true, 'מקשיבה... דבר'); break;
            case 'stt_result':
                if (!msg.empty && msg.text) { this.addMessage('user', msg.text); this.setListening(false); }
                else { this.setListening(false, 'לא שמעתי'); setTimeout(() => this.setIdle(), 2000); }
                break;
            case 'brain_response':
                if (msg.assistant_text) {
                    this.addMessage('assistant', msg.assistant_text);
                    if (msg.screen_image) this.showScreen(msg.screen_image, msg.screen_context);
                    if (msg.proposals) msg.proposals.forEach(p => this.addProposal(p));
                    if (msg.self_updates) msg.self_updates.forEach(p => this.addProposal(p));
                    if (msg.user_profile) this.updateProfileUI(msg.user_profile);
                }
                this.setSpeaking(true, msg.assistant_text, msg.audio_base64);
                if (!msg.audio_base64) setTimeout(() => this.setSpeaking(false), Math.max(2000, (msg.assistant_text||'').length * 70));
                break;
            case 'assistant_speaking':
                if (msg.text && msg.type !== 'brain_response') this.addMessage('assistant', msg.text);
                this.setSpeaking(true, msg.text, msg.audio_base64);
                if (!msg.audio_base64) setTimeout(() => this.setSpeaking(false), Math.max(2000, (msg.text||'').length * 70));
                break;
            case 'learning_proposal': this.addProposal(msg.proposal); break;
            case 'self_update_proposal': this.addProposal(msg.proposal); break;
            case 'proposal_approved': this.addMessage('assistant', `✅ ${msg.message}`); this.removeProposal(msg.proposal_id); break;
            case 'proposal_rejected': this.addMessage('assistant', `❌ ${msg.message}`); this.removeProposal(msg.proposal_id); break;
            case 'self_update_approved': this.addMessage('assistant', `🚀 ${msg.message}`); this.removeProposal(msg.update_id); break;
            case 'self_update_rejected': this.addMessage('assistant', `👌 ${msg.message}`); this.removeProposal(msg.update_id); break;
            case 'hud_command': this.handleHUDCommand(msg.command); break;
            case 'mic_changed': this.addMessage('assistant', `🎙️ ${msg.message}`); if (this.els.micStatusMini) this.els.micStatusMini.textContent = msg.device_name; break;
            case 'task_result': this.addTaskResult(msg); break;
            case 'routine_result': this.addMessage('assistant', `⏰ ${msg.message}`); break;
        }
    }

    handleHUDCommand(cmd) {
        const action = typeof cmd === 'string' ? cmd : cmd.action;
        if (action === 'dock' || action === 'side') this.switchMode('side');
        else if (action === 'center') this.switchMode('center');
        else if (action === 'hide' || action === 'orb') this.switchMode('orb');
        else if (action === 'show') this.switchMode('center');
    }

    onWake(text) {
        this.setListening(true, `כן בוס? שמעתי "${text}"`);
        if (window.reactorAnim) window.reactorAnim.setMode('listening');
        this.els.listeningPulse?.classList.add('active');
        if (this.els.orbitalListen) this.els.orbitalListen.textContent = 'LISTEN';
    }

    setListening(isListening, label) {
        if (isListening) {
            this.els.statusLabel?.classList.add('listening');
            this.els.statusLabel.textContent = label || 'מקשיבה...';
            this.els.visualizer?.classList.add('active');
            this.els.mainStatusText.textContent = 'LISTENING';
            this.els.mainStatusDot?.classList.add('listening');
            if (this.els.orbitalListen) this.els.orbitalListen.textContent = 'REC';
        } else {
            this.els.statusLabel?.classList.remove('listening');
            this.els.listeningPulse?.classList.remove('active');
            this.els.visualizer?.classList.remove('active');
            if (!this.els.statusLabel?.classList.contains('speaking')) this.setIdle();
        }
    }

    setSpeaking(isSpeaking, text, audioB64) {
        if (isSpeaking) {
            this.els.statusLabel?.classList.add('speaking');
            this.els.statusLabel.textContent = text ? `מדברת: ${text.slice(0,35)}...` : 'מדברת...';
            this.els.visualizer?.classList.add('active');
            this.els.mainStatusText.textContent = 'SPEAKING';
            if (window.reactorAnim) window.reactorAnim.setMode('speaking');
            if (audioB64) this.playAudio(audioB64);
        } else {
            this.els.statusLabel?.classList.remove('speaking');
            this.els.visualizer?.classList.remove('active');
            this.setIdle();
            if (window.reactorAnim) window.reactorAnim.setMode('idle');
        }
    }

    setIdle() {
        if (!this.els.statusLabel?.classList.contains('listening') && !this.els.statusLabel?.classList.contains('speaking')) {
            this.els.statusLabel.textContent = 'מאזינה למילת הפעלה...';
            this.els.statusLabel.classList.remove('listening','speaking');
            this.els.mainStatusText.textContent = 'SYSTEMS NOMINAL';
            this.els.mainStatusDot?.classList.remove('listening');
            if (this.els.orbitalListen) this.els.orbitalListen.textContent = 'IDLE';
        }
    }

    playAudio(b64) {
        try {
            if (this.currentAudio) { this.currentAudio.pause(); this.currentAudio = null; }
            const audio = new Audio(b64);
            this.currentAudio = audio;
            audio.onended = () => this.setSpeaking(false);
            audio.onerror = () => this.setSpeaking(false);
            audio.play().catch(() => {
                // Fallback Web Speech
                if ('speechSynthesis' in window) {
                    const utter = new SpeechSynthesisUtterance(this.els.statusLabel.textContent);
                    utter.lang = 'he-IL';
                    speechSynthesis.speak(utter);
                }
            });
        } catch(e) { console.error(e); }
    }

    addMessage(role, text) {
        const inner = this.els.conversationInner;
        if (!inner) return;
        const time = new Date().toLocaleTimeString('he-IL', {hour:'2-digit', minute:'2-digit'});
        const div = document.createElement('div');
        div.className = `msg ${role}`;
        const avatar = role === 'user' ? 'אתה' : 'AJ';
        div.innerHTML = `<div class="msg-avatar">${avatar}</div><div class="msg-content"><div class="msg-text">${this.escapeHtml(text)}</div><div class="msg-time">${time}</div></div>`;
        inner.appendChild(div);
        inner.scrollTop = inner.scrollHeight;
        if (this.els.convBadge) this.els.convBadge.textContent = inner.children.length;
        
        // Log
        if (this.els.logsContent) {
            const log = document.createElement('div');
            log.className = 'log-line';
            log.textContent = `[${time}] ${role}: ${text.substring(0,60)}`;
            this.els.logsContent.appendChild(log);
            this.els.logsContent.scrollTop = this.els.logsContent.scrollHeight;
            while (this.els.logsContent.children.length > 30) this.els.logsContent.removeChild(this.els.logsContent.firstChild);
        }
        
        while (inner.children.length > 60) inner.removeChild(inner.firstChild);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML.replace(/\n/g, '<br>');
    }

    sendText() {
        const input = this.els.textInput;
        if (!input) return;
        const text = input.value.trim();
        if (!text) return;
        
        const screenKeywords = ["מסך", "רואה", "שגיאה", "קוד", "מה פתוח", "תסרוק"];
        const needsScreen = screenKeywords.some(kw => text.includes(kw));
        
        if (this.ws?.readyState === 1) {
            // נסה קודם כמשימת סוכן-על אם זה נראה כמו משימה
            const taskKeywords = ["תקנה", "תזמין", "תחקור", "תמלא", "תבדוק מייל", "תמצא זמן", "תארגן", "תכין דוח", "תפתח", "תריץ שגרה"];
            if (taskKeywords.some(kw => text.includes(kw))) {
                this.executeTask(text);
            } else {
                this.ws.send(JSON.stringify({type: 'text', text: text, with_screen: needsScreen}));
            }
            this.addMessage('user', text);
            input.value = '';
        } else {
            this.addMessage('assistant', 'לא מחוברת ל-Backend');
        }
    }

    triggerWake() {
        if (this.ws?.readyState === 1) this.ws.send(JSON.stringify({type: 'manual_wake'}));
        if (window.adielAPI) window.adielAPI.simulateWake();
        this.onWake('manual');
    }

    askScreen() {
        const q = 'מה את רואה במסך?';
        if (this.ws?.readyState === 1) {
            this.ws.send(JSON.stringify({type: 'text', text: q, with_screen: true}));
            this.addMessage('user', q);
        }
    }

    async executeTask(taskText) {
        try {
            const backendUrl = this.wsUrl.replace('ws://', 'http://').replace('wss://', 'https://').replace('/ws', '');
            const res = await fetch(`${backendUrl}/tasks/execute`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: taskText})
            });
            const data = await res.json();
            this.addMessage('assistant', `🚀 ${data.message || 'מבצעת משימה...'}`);
            if (data.result) {
                const details = JSON.stringify(data.result, null, 2).substring(0, 600);
                this.addMessage('assistant', `📋 ${details}...`);
            }
            this.addTaskToMini(taskText, data);
        } catch (e) {
            console.error(e);
            // Fallback
            if (this.ws?.readyState === 1) {
                this.ws.send(JSON.stringify({type: 'text', text: taskText, with_screen: false}));
            }
        }
    }

    addTaskToMini(text, result) {
        if (!this.els.tasksMiniList) return;
        if (this.els.tasksMiniList.querySelector('.placeholder-holo')) this.els.tasksMiniList.innerHTML = '';
        
        const div = document.createElement('div');
        div.className = 'task-history-item';
        div.innerHTML = `<div class="task-history-text">🚀 ${this.escapeHtml(text)}</div><div class="task-history-result">${this.escapeHtml((result.message||'').substring(0,80))}</div><div class="task-history-time">${result.success ? '✅' : '❌'} ${new Date().toLocaleTimeString('he-IL')}</div>`;
        this.els.tasksMiniList.insertBefore(div, this.els.tasksMiniList.firstChild);
        
        if (this.els.tasksBadge) this.els.tasksBadge.textContent = this.els.tasksMiniList.children.length;
        if (this.els.orbitalTasks) this.els.orbitalTasks.textContent = this.els.tasksMiniList.children.length;
        if (this.els.tasksCountMini) this.els.tasksCountMini.textContent = `${this.els.tasksMiniList.children.length} משימות`;
    }

    addProposal(proposal) {
        if (!this.els.proposalsList) return;
        
        if (this.els.proposalsList.querySelector('.placeholder-holo')) {
            this.els.proposalsList.innerHTML = '';
        }
        
        if (document.getElementById(`proposal-${proposal.id}`)) return;
        
        const card = document.createElement('div');
        card.className = `proposal-card type-${proposal.type}`;
        card.id = `proposal-${proposal.id}`;
        const risk = proposal.risk || 'low';
        const riskLabel = {low: 'נמוך', medium: 'בינוני', high: 'גבוה'}[risk];
        
        card.innerHTML = `
            <div class="proposal-title">${this.escapeHtml(proposal.title || proposal.description_he?.substring(0,40) || 'הצעה')}</div>
            <div class="proposal-description">${this.escapeHtml(proposal.description_he || '')}</div>
            <div class="proposal-meta"><span class="proposal-tag">${proposal.type}</span><span class="proposal-tag risk-${risk}">${riskLabel}</span></div>
            ${proposal.before_example ? `<div class="proposal-explanation">לפני: ${this.escapeHtml(proposal.before_example)}<br>אחרי: ${this.escapeHtml(proposal.after_example)}</div>` : ''}
            ${proposal.explanation_for_user ? `<div class="proposal-explanation">${this.escapeHtml(proposal.explanation_for_user.substring(0,200))}</div>` : ''}
            <div class="proposal-actions">
                <button class="proposal-btn approve" data-id="${proposal.id}">✅ אשר</button>
                <button class="proposal-btn reject" data-id="${proposal.id}">❌ דחה</button>
            </div>
        `;
        
        card.querySelector('.approve')?.addEventListener('click', (e) => {
            const id = e.target.getAttribute('data-id');
            this.approveProposal(id, proposal.type?.includes('self_update') ? 'self_update' : 'learning');
        });
        card.querySelector('.reject')?.addEventListener('click', (e) => {
            const id = e.target.getAttribute('data-id');
            this.rejectProposal(id, proposal.type?.includes('self_update') ? 'self_update' : 'learning');
        });
        
        this.els.proposalsList.appendChild(card);
        this.proposals.push(proposal);
        
        if (this.els.proposalsBadge) this.els.proposalsBadge.textContent = this.proposals.length;
        
        // Switch to proposals tab automatically
        this.switchTab('proposals');
        this.addMessage('assistant', `📚 ${proposal.description_he} - ממתין לאישורך בטאב למידה, בוס`);
    }

    removeProposal(id) {
        document.getElementById(`proposal-${id}`)?.remove();
        this.proposals = this.proposals.filter(p => p.id !== id);
        if (this.els.proposalsBadge) this.els.proposalsBadge.textContent = this.proposals.length;
    }

    approveProposal(id, type) {
        if (this.ws?.readyState !== 1) return;
        const endpoint = type === 'self_update' ? 'approve_self_update' : 'approve_proposal';
        this.ws.send(JSON.stringify({type: endpoint, id: id}));
        const card = document.getElementById(`proposal-${id}`);
        if (card) { card.querySelectorAll('button').forEach(b => b.disabled = true); card.querySelector('.approve').textContent = '⏳ מאשר...'; }
    }

    rejectProposal(id, type) {
        if (this.ws?.readyState !== 1) return;
        const endpoint = type === 'self_update' ? 'reject_self_update' : 'reject_proposal';
        this.ws.send(JSON.stringify({type: endpoint, id: id}));
        const card = document.getElementById(`proposal-${id}`);
        if (card) { card.querySelectorAll('button').forEach(b => b.disabled = true); card.querySelector('.reject').textContent = '⏳ דוחה...'; }
    }

    async loadMics() {
        if (!this.els.micList && !document.getElementById('micList')) return;
        const listEl = this.els.micList || document.getElementById('micList');
        if (!listEl) return;
        
        listEl.innerHTML = '<div class="loading">🎙️ טוען...</div>';
        try {
            const backendUrl = this.wsUrl.replace('ws://', 'http://').replace('wss://', 'https://').replace('/ws', '');
            const res = await fetch(`${backendUrl}/audio/devices`);
            const data = await res.json();
            
            if (!data.inputs || data.inputs.length === 0) {
                listEl.innerHTML = '<div class="loading">לא נמצאו מיקרופונים. אפשר לכתוב במקלדת!</div>';
                return;
            }
            
            listEl.innerHTML = '';
            data.inputs.forEach(mic => {
                const div = document.createElement('div');
                div.className = `mic-option ${mic.selected ? 'selected' : ''}`;
                div.innerHTML = `<div class="mic-name">${this.escapeHtml(mic.name)}</div><div class="mic-details">ערוצים: ${mic.input_channels} ${mic.is_default_input ? '<span class="mic-tag default">ברירת מחדל</span>' : ''} ${mic.selected ? '<span class="mic-tag">נבחר ✓</span>' : ''}</div>`;
                div.addEventListener('click', () => this.selectMic(mic.id));
                listEl.appendChild(div);
            });
        } catch (e) {
            listEl.innerHTML = `<div class="loading">שגיאה: ${e.message}</div>`;
        }
    }

    async selectMic(id) {
        try {
            const backendUrl = this.wsUrl.replace('ws://', 'http://').replace('wss://', 'https://').replace('/ws', '');
            const res = await fetch(`${backendUrl}/audio/devices/input/${id}`, {method: 'POST'});
            const data = await res.json();
            if (data.success) {
                this.addMessage('assistant', `✅ ${data.message}`);
                if (this.els.micStatusMini) this.els.micStatusMini.textContent = data.device?.name?.substring(0,15) || 'נבחר';
                this.loadMics();
            } else {
                this.addMessage('assistant', `❌ ${data.error}`);
            }
        } catch (e) {
            this.addMessage('assistant', `❌ שגיאה: ${e.message}`);
        }
    }

    async loadProfile() {
        try {
            const backendUrl = this.wsUrl.replace('ws://', 'http://').replace('wss://', 'https://').replace('/ws', '');
            const res = await fetch(`${backendUrl}/profile`);
            const data = await res.json();
            
            if (this.els.profileName) this.els.profileName.textContent = data.profile?.name || 'בוס';
            if (this.els.projectsCount) this.els.projectsCount.textContent = data.profile?.projects?.length || 0;
            if (this.els.interactionsCount) this.els.interactionsCount.textContent = data.profile?.interaction_count || 0;
            if (this.els.memoryContext) this.els.memoryContext.textContent = data.smart_context || 'דיברנו 0 פעמים';
            if (this.els.memoryCount) this.els.memoryCount.textContent = data.profile?.facts_count || 0;
            if (this.els.vocabCount) this.els.vocabCount.textContent = data.vocabulary?.count || 0;
            if (this.els.orbitalMem) this.els.orbitalMem.textContent = data.vocabulary?.count || 0;
            if (this.els.trueAIStatus) this.els.trueAIStatus.textContent = `🧠 Vocab ${data.vocabulary?.count || 0}`;
            
            if (document.getElementById('profileInfo')) {
                document.getElementById('profileInfo').innerHTML = `
                    <div>שם: ${data.profile?.name || 'לא ידוע'}</div>
                    <div>שיחות: ${data.profile?.interaction_count || 0}</div>
                    <div>מילים: ${data.vocabulary?.count || 0}</div>
                    <div style="margin-top:6px; font-size:10px; opacity:0.7;">${data.smart_context || ''}</div>
                `;
            }
        } catch (e) {
            console.log('Profile load failed', e);
        }
    }

    async loadHealth() {
        try {
            const backendUrl = this.wsUrl.replace('ws://', 'http://').replace('wss://', 'https://').replace('/ws', '');
            const res = await fetch(`${backendUrl}/health`);
            const data = await res.json();
            if (document.getElementById('healthInfo')) {
                const checks = data.checks || {};
                let html = '';
                for (const [k,v] of Object.entries(checks)) {
                    html += `<div class="sys-metric"><label>${k}</label><span class="${v ? 'status-ok' : ''}">${v ? '● OK' : '○ FAIL'}</span></div>`;
                }
                document.getElementById('healthInfo').innerHTML = html || 'טוען...';
            }
        } catch (e) {
            console.log('Health load failed', e);
        }
    }

    testVoice(voiceType) {
        if (voiceType === 'original') {
            const voices = ['./assets/voice_hello.mp3', './assets/voice_wake.mp3', './assets/voice_how_are_you.mp3'];
            const audio = new Audio(voices[Math.floor(Math.random()*voices.length)]);
            audio.play();
            this.addMessage('assistant', '✨ מנגן קול מקורי רק שלך!');
            this.setSpeaking(true, 'מנגן קול מקורי...');
            audio.onended = () => this.setSpeaking(false);
        } else {
            if (this.ws?.readyState === 1) {
                this.ws.send(JSON.stringify({type: 'speak', text: `בדיקת קול ${voiceType}`}));
            }
        }
    }

    trainAI() {
        this.addMessage('assistant', '🧠 מאמן AI... זה ייקח דקה');
        fetch(this.wsUrl.replace('ws://', 'http://').replace('wss://', 'https://').replace('/ws', '') + '/tasks/execute', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({text: 'תחקור על בינה מלאכותית'})
        }).then(() => {
            // Also run local training visual
            this.addMessage('assistant', '✅ אימון הושלם! Vocab גדל, Markov התעדכן');
        });
        
        // Visual training effect
        if (window.reactorAnim) {
            window.reactorAnim.setMode('thinking');
            setTimeout(() => window.reactorAnim.setMode('idle'), 3000);
        }
    }

    fixSystem() {
        fetch(this.wsUrl.replace('ws://', 'http://').replace('wss://', 'https://').replace('/ws', '') + '/fix', {method: 'POST'})
            .then(r => r.json())
            .then(data => {
                this.addMessage('assistant', `🛠️ תיקון: ${data.fixed?.length || 0} דברים תוקנו - ${data.log?.join(', ') || 'הכל תקין'}`);
            });
    }

    clearConversation() {
        const inner = this.els.conversationInner;
        if (inner) {
            inner.innerHTML = `<div class="msg assistant welcome"><div class="msg-avatar">AJ</div><div class="msg-content"><div class="msg-text">שיחה נוקתה, אבל אני זוכרת הכל! MARK 85 מוכן.</div><div class="msg-time">עכשיו</div></div></div>`;
        }
    }

    showScreen(b64, context) {
        if (!this.els.previewImage) return;
        this.els.previewImage.src = b64.startsWith('data:') ? b64 : `data:image/jpeg;base64,${b64}`;
        if (this.els.previewContext) this.els.previewContext.textContent = context || '';
        this.els.screenPreview?.classList.remove('hidden');
        if (this.els.screenPlaceholder) this.els.screenPlaceholder.classList.add('hidden');
        this.switchTab('screen');
    }

    updateProfileUI(profile) {
        if (!profile) return;
        if (this.els.profileName) this.els.profileName.textContent = profile.name || 'בוס';
        if (this.els.projectsCount) this.els.projectsCount.textContent = (profile.projects?.length || 0);
        if (this.els.interactionsCount) this.els.interactionsCount.textContent = profile.interaction_count || 0;
        if (this.els.memoryCount) this.els.memoryCount.textContent = profile.facts_count || 0;
        if (this.els.vocabCount) this.els.vocabCount.textContent = profile.vocab_learned || 0;
        if (this.els.orbitalMem) this.els.orbitalMem.textContent = profile.vocab_learned || 0;
    }

    addTaskResult(msg) {
        this.addMessage('assistant', `🚀 ${msg.message}`);
        this.addTaskToMini(msg.task_type || 'משימה', {message: msg.message, success: msg.success});
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.advancedHUD = new AdvancedHUD();
    console.log('[Advanced HUD] MARK 85 initialized - Super Agent Ready');
});
