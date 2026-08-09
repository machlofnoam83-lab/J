/**
 * Adiel Junior - HUD Logic
 * Frontend brain connecting to Python backend via WebSocket
 */

class AdielHUD {
    constructor() {
        this.ws = null;
        this.wsUrl = this.detectWebSocketUrl();
        this.connected = false;
        this.currentMode = 'center';
        this.isListening = false;
        this.isSpeaking = false;
        this.audioContext = null;
        
        this.elements = {};
        this.initElements();
        this.bindEvents();
        this.connectWebSocket();
        this.updateConnectionUI('connecting');
    }

    detectWebSocketUrl() {
        // Try to detect preview environment (Arena / e2b)
        // Format: https://{port}-{sandboxId}.e2b.app
        const host = window.location.hostname;
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        
        // If running on e2b preview domain
        if (host.includes('.e2b.app') && host.includes('-')) {
            // Extract sandbox id: {port}-{id}.e2b.app -> {id}.e2b.app
            const parts = host.split('-');
            if (parts.length >= 2) {
                const portPart = parts[0]; // e.g., "3000"
                const rest = parts.slice(1).join('-'); // {id}.e2b.app
                // Backend should be on 8765 with same id
                const backendHost = `8765-${rest}`;
                const wsUrl = `${protocol}//${backendHost}/ws`;
                console.log(`[HUD] Detected e2b preview, backend WS: ${wsUrl}`);
                return wsUrl;
            }
        }
        
        // If localhost:3000 dev, backend on 8765 local
        if (host === 'localhost' || host === '127.0.0.1') {
            return 'ws://localhost:8765/ws';
        }
        
        // Fallback - try same host with 8765 port or relative
        // For production Electron, localhost is correct
        return 'ws://localhost:8765/ws';
    }

    initElements() {
        this.elements = {
            app: document.getElementById('app'),
            reactorContainer: document.getElementById('reactorContainer'),
            statusDot: document.getElementById('statusDot'),
            statusLabel: document.getElementById('statusLabel'),
            listeningPulse: document.getElementById('listeningPulse'),
            visualizer: document.getElementById('visualizer'),
            conversationArea: document.getElementById('conversationArea'),
            conversationInner: document.getElementById('conversationInner'),
            textInput: document.getElementById('textInput'),
            btnSend: document.getElementById('btnSend'),
            btnSide: document.getElementById('btnSide'),
            btnCenter: document.getElementById('btnCenter'),
            btnOrb: document.getElementById('btnOrb'),
            btnClose: document.getElementById('btnClose'),
            btnWake: document.getElementById('btnWake'),
            btnScreen: document.getElementById('btnScreen'),
            btnClear: document.getElementById('btnClear'),
            screenPreview: document.getElementById('screenPreview'),
            previewImage: document.getElementById('previewImage'),
            previewContext: document.getElementById('previewContext'),
            connDot: document.getElementById('connDot'),
            connText: document.getElementById('connText'),
            orbContent: document.getElementById('orbContent'),
            proposalsArea: document.getElementById('proposalsArea'),
            proposalsList: document.getElementById('proposalsList'),
            proposalsBadge: document.getElementById('proposalsBadge'),
        };
        this.proposals = []; // כל ההצעות שממתינות
    }

    bindEvents() {
        // HUD window controls
        if (window.adielAPI) {
            this.elements.btnSide?.addEventListener('click', () => this.switchMode('side'));
            this.elements.btnCenter?.addEventListener('click', () => this.switchMode('center'));
            this.elements.btnOrb?.addEventListener('click', () => this.switchMode('orb'));
            this.elements.btnClose?.addEventListener('click', () => window.adielAPI.close());

            window.adielAPI.onModeChanged((mode) => {
                this.currentMode = mode;
                this.updateModeUI(mode);
            });

            window.adielAPI.onSimulateWake(() => {
                this.simulateWakeFromUI();
            });
        } else {
            // Browser fallback - still allow mode switching via CSS
            this.elements.btnSide?.addEventListener('click', () => this.switchMode('side'));
            this.elements.btnCenter?.addEventListener('click', () => this.switchMode('center'));
            this.elements.btnOrb?.addEventListener('click', () => this.switchMode('orb'));
            this.elements.btnClose?.addEventListener('click', () => {
                if (confirm('לסגור את אדיאל ג\'וניור?')) window.close();
            });
        }

        // Input
        this.elements.btnSend?.addEventListener('click', () => this.sendText());
        this.elements.textInput?.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') this.sendText();
        });

        // Actions
        this.elements.btnWake?.addEventListener('click', () => this.triggerWake());
        this.elements.btnScreen?.addEventListener('click', () => this.askScreen());
        this.elements.btnClear?.addEventListener('click', () => this.clearConversation());
        this.elements.btnMicSelect = document.getElementById('btnMicSelect');
        this.elements.btnVoiceSelect = document.getElementById('btnVoiceSelect');
        this.elements.micPanel = document.getElementById('micPanel');
        this.elements.voicePanel = document.getElementById('voicePanel');
        this.elements.micList = document.getElementById('micList');
        this.elements.btnCloseMic = document.getElementById('btnCloseMic');
        this.elements.btnCloseVoice = document.getElementById('btnCloseVoice');
        this.elements.btnRefreshMics = document.getElementById('btnRefreshMics');
        this.elements.btnResetMic = document.getElementById('btnResetMic');

        this.elements.btnMicSelect?.addEventListener('click', () => this.toggleMicPanel());
        this.elements.btnVoiceSelect?.addEventListener('click', () => this.toggleVoicePanel());
        this.elements.btnCloseMic?.addEventListener('click', () => this.hideMicPanel());
        this.elements.btnCloseVoice?.addEventListener('click', () => this.hideVoicePanel());
        this.elements.btnRefreshMics?.addEventListener('click', () => this.loadMicrophones());
        this.elements.btnResetMic?.addEventListener('click', () => this.resetMicrophone());

        // Voice test buttons
        document.querySelectorAll('.test-voice-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const voice = e.target.getAttribute('data-voice');
                this.testVoice(voice);
            });
        });

        // Orb click
        this.elements.orbContent?.addEventListener('click', () => {
            this.switchMode('center');
            this.triggerWake();
        });

        // Drag handle (for electron, but also works as visual)
        const dragHandle = document.getElementById('dragHandle');
        if (dragHandle && !window.adielAPI) {
            // Simple draggable for browser testing
            let isDragging = false, startX, startY, initialX, initialY;
            dragHandle.addEventListener('mousedown', (e) => {
                isDragging = true;
                startX = e.clientX;
                startY = e.clientY;
                const rect = this.elements.app.getBoundingClientRect();
                initialX = rect.left;
                initialY = rect.top;
            });
            document.addEventListener('mousemove', (e) => {
                if (!isDragging) return;
                const dx = e.clientX - startX;
                const dy = e.clientY - startY;
                this.elements.app.style.position = 'fixed';
                this.elements.app.style.left = (initialX + dx) + 'px';
                this.elements.app.style.top = (initialY + dy) + 'px';
            });
            document.addEventListener('mouseup', () => isDragging = false);
        }
    }

    switchMode(mode) {
        this.currentMode = mode;
        this.updateModeUI(mode);
        if (window.adielAPI) {
            window.adielAPI.switchMode(mode);
        }
    }

    updateModeUI(mode) {
        this.elements.app.className = `hud-container mode-${mode}`;
        // For orb mode, clicking the core should come back
        if (mode === 'orb') {
            this.elements.app.style.width = '140px';
            this.elements.app.style.height = '140px';
        } else if (mode === 'side') {
            this.elements.app.style.width = '400px';
            this.elements.app.style.height = '720px';
        } else {
            this.elements.app.style.width = '520px';
            this.elements.app.style.height = '680px';
        }
    }

    connectWebSocket() {
        console.log(`[HUD] Connecting to ${this.wsUrl}...`);
        this.updateConnectionUI('connecting');

        try {
            this.ws = new WebSocket(this.wsUrl);

            this.ws.onopen = () => {
                console.log('[HUD] WS Connected');
                this.connected = true;
                this.updateConnectionUI('connected');
                this.addMessage('assistant', 'התחברתי ל-Backend, בוס. המערכות ירוקות. אמור "אדיאל ג\'וניור" כדי להתחיל.');
                
                // Ping loop
                setInterval(() => {
                    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                        this.ws.send(JSON.stringify({ type: 'ping' }));
                    }
                }, 30000);
            };

            this.ws.onmessage = (event) => {
                try {
                    const msg = JSON.parse(event.data);
                    this.handleBackendMessage(msg);
                } catch (e) {
                    console.error('[HUD] Parse error', e);
                }
            };

            this.ws.onclose = () => {
                console.log('[HUD] WS Closed, reconnecting in 3s');
                this.connected = false;
                this.updateConnectionUI('disconnected');
                setTimeout(() => this.connectWebSocket(), 3000);
            };

            this.ws.onerror = (err) => {
                console.error('[HUD] WS Error', err);
                this.updateConnectionUI('disconnected');
            };

        } catch (e) {
            console.error('[HUD] WS Connect failed', e);
            this.updateConnectionUI('disconnected');
            setTimeout(() => this.connectWebSocket(), 3000);
        }
    }

    handleBackendMessage(msg) {
        console.log('[HUD] Message', msg.type, msg);

        switch (msg.type) {
            case 'connected':
                this.updateConnectionUI('connected');
                break;

            case 'wake_detected':
                this.onWakeDetected(msg.text);
                break;

            case 'listening':
                this.setListening(true, msg.message);
                break;

            case 'stt_state':
                if (msg.state === 'recording') {
                    this.setListening(true, 'מקשיבה... דבר עכשיו');
                }
                break;

            case 'stt_result':
                if (!msg.empty && msg.text) {
                    this.addMessage('user', msg.text);
                    this.setListening(false);
                } else {
                    this.setListening(false, 'לא שמעתי, תנסה שוב?');
                    setTimeout(() => this.setIdle(), 2000);
                }
                break;

            case 'brain_response':
            case 'assistant_speaking':
                const text = msg.assistant_text || msg.text || '';
                if (text) {
                    if (msg.type === 'brain_response') {
                        this.addMessage('assistant', text);
                        if (msg.screen_image) {
                            this.showScreenPreview(msg.screen_image, msg.screen_context);
                        }
                        if (msg.hud_command) {
                            this.handleHUDCommand(msg.hud_command);
                        }
                        if (msg.proposals && msg.proposals.length > 0) {
                            msg.proposals.forEach(p => this.addProposal(p, 'learning'));
                        }
                        if (msg.self_updates && msg.self_updates.length > 0) {
                            msg.self_updates.forEach(p => this.addProposal(p, 'self_update'));
                        }
                        if (msg.user_profile) {
                            console.log('[HUD] User profile', msg.user_profile);
                        }
                    }
                    this.setSpeaking(true, text, msg.audio_base64 || null);
                    const duration = msg.audio_base64 ? null : Math.max(2000, text.length * 80);
                    if (duration) {
                        setTimeout(() => this.setSpeaking(false), duration);
                    }
                    // אם יש base64, ה-setSpeaking יכבה לבד ב-onended
                }
                break;

            case 'learning_proposal':
                this.addProposal(msg.proposal, 'learning');
                break;

            case 'self_update_proposal':
                this.addProposal(msg.proposal, 'self_update');
                break;

            case 'proposal_approved':
                this.addMessage('assistant', `✅ ${msg.message || 'למדתי! תודה בוס, אני חכמה יותר עכשיו.'}`);
                this.removeProposalCard(msg.proposal_id);
                break;

            case 'proposal_rejected':
                this.addMessage('assistant', `❌ ${msg.message || 'סגור, לא אזכור את זה.'}`);
                this.removeProposalCard(msg.proposal_id);
                break;

            case 'self_update_approved':
                this.addMessage('assistant', `🚀 ${msg.message || 'עדכנתי את עצמי! המוח שלי השתפר.'}`);
                this.removeProposalCard(msg.update_id);
                if (msg.update && msg.update.explanation_for_user) {
                    this.addMessage('assistant', `הסבר: ${msg.update.explanation_for_user.substring(0,300)}...`);
                }
                break;

            case 'self_update_rejected':
                this.addMessage('assistant', `👌 ${msg.message || 'סגור, לא אעדכן את זה.'}`);
                this.removeProposalCard(msg.update_id);
                break;

            case 'hud_command':
                this.handleHUDCommand(msg.command);
                break;

            case 'tts':
                this.setSpeaking(true, msg.text);
                break;

            case 'screen_data':
                this.showScreenPreview(msg.image, msg.context);
                break;

            case 'error':
                this.addMessage('assistant', `אופס: ${msg.message}`);
                break;

            case 'pong':
                break;
        }
    }

    // === NEW: Proposals System - אדיאל רוצה ללמוד ומבקשת אישור ===

    addProposal(proposal, type='learning') {
        this.proposals.push(proposal);
        this.renderProposalCard(proposal, type);
        this.updateProposalsBadge();
        
        // פתח את אזור ההצעות
        if (this.elements.proposalsArea) {
            this.elements.proposalsArea.classList.remove('hidden');
        }
        
        // אם זה הצעת אוצר מילים / פרופיל חשובה, הודע בצ'אט
        if (proposal.type === 'profile_new' || proposal.type === 'vocabulary') {
            this.addMessage('assistant', `📚 ${proposal.description_he} - זה ממתין לאישורך למטה, בוס.`);
        }
    }

    renderProposalCard(proposal, type) {
        if (!this.elements.proposalsList) return;
        
        const existing = document.getElementById(`proposal-${proposal.id}`);
        if (existing) return; // כבר קיים
        
        const card = document.createElement('div');
        card.className = `proposal-card type-${proposal.type || type}`;
        card.id = `proposal-${proposal.id}`;
        
        const risk = proposal.risk || 'low';
        const riskLabel = {low: 'סיכון נמוך', medium: 'בינוני', high: 'גבוה'}[risk] || risk;
        
        const title = proposal.title || proposal.description_he?.substring(0,40) || 'הצעה חדשה';
        const description = proposal.description_he || proposal.description || '';
        const explanation = proposal.explanation_for_user || proposal.what_it_does || '';
        const beforeAfter = proposal.before_example && proposal.after_example ? 
            `<div class="proposal-explanation"><b>לפני:</b> ${this.escapeHtml(proposal.before_example)}<br><b>אחרי:</b> ${this.escapeHtml(proposal.after_example)}</div>` : 
            (explanation ? `<div class="proposal-explanation">${this.escapeHtml(explanation)}</div>` : '');
        
        card.innerHTML = `
            <div class="proposal-title">${this.escapeHtml(title)}</div>
            <div class="proposal-description">${this.escapeHtml(description)}</div>
            <div class="proposal-meta">
                <span class="proposal-tag">${this.escapeHtml(proposal.type || type)}</span>
                <span class="proposal-tag risk-${risk}">${riskLabel}</span>
                ${proposal.word ? `<span class="proposal-tag">${this.escapeHtml(proposal.word)}</span>` : ''}
            </div>
            ${beforeAfter}
            <div class="proposal-actions">
                <button class="proposal-btn approve" data-id="${proposal.id}" data-type="${type}">✅ אשר - למד/עדכן</button>
                <button class="proposal-btn reject" data-id="${proposal.id}" data-type="${type}">❌ דחה</button>
            </div>
        `;
        
        // אירועי כפתורים
        card.querySelector('.approve')?.addEventListener('click', (e) => {
            const id = e.target.getAttribute('data-id');
            const t = e.target.getAttribute('data-type');
            this.approveProposal(id, t);
        });
        card.querySelector('.reject')?.addEventListener('click', (e) => {
            const id = e.target.getAttribute('data-id');
            const t = e.target.getAttribute('data-type');
            this.rejectProposal(id, t);
        });
        
        this.elements.proposalsList.appendChild(card);
    }

    updateProposalsBadge() {
        if (this.elements.proposalsBadge) {
            this.elements.proposalsBadge.textContent = this.proposals.length;
        }
    }

    removeProposalCard(id) {
        const card = document.getElementById(`proposal-${id}`);
        if (card) {
            card.style.animation = 'msgEnter 0.3s reverse';
            setTimeout(() => card.remove(), 300);
        }
        this.proposals = this.proposals.filter(p => p.id !== id);
        this.updateProposalsBadge();
        if (this.proposals.length === 0 && this.elements.proposalsArea) {
            this.elements.proposalsArea.classList.add('hidden');
        }
    }

    approveProposal(id, type) {
        console.log(`[HUD] Approving ${type} proposal ${id}`);
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            this.addMessage('assistant', 'אני לא מחוברת ל-Backend לאישור.');
            return;
        }
        
        if (type === 'self_update') {
            // Self-update approval - דורש הסבר מלא
            fetch(`http://${window.location.hostname}:8765/self-updates/${id}/approve`, {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    console.log('Self-update approved', data);
                    if (!data.success) throw new Error(data.error);
                })
                .catch(() => {
                    // Fallback via WS
                    this.ws.send(JSON.stringify({type: 'approve_self_update', id: id}));
                });
            // גם דרך WS
            this.ws.send(JSON.stringify({type: 'approve_self_update', id: id}));
        } else {
            // Learning proposal
            fetch(`http://${window.location.hostname}:8765/proposals/${id}/approve`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({})
            }).then(r => r.json()).then(data => {
                console.log('Proposal approved', data);
            }).catch(() => {});
            // Fallback via WS
            this.ws.send(JSON.stringify({type: 'approve_proposal', id: id}));
        }
        
        // אופטימי - הסר מיד
        const card = document.getElementById(`proposal-${id}`);
        if (card) {
            card.querySelectorAll('button').forEach(b => b.disabled = true);
            card.querySelector('.approve').textContent = '⏳ מאשר...';
        }
    }

    rejectProposal(id, type) {
        console.log(`[HUD] Rejecting ${type} proposal ${id}`);
        if (type === 'self_update') {
            fetch(`http://${window.location.hostname}:8765/self-updates/${id}/reject`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({reason: 'user rejected via HUD'})
            }).catch(()=>{});
            this.ws?.send(JSON.stringify({type: 'reject_self_update', id: id}));
        } else {
            fetch(`http://${window.location.hostname}:8765/proposals/${id}/reject`, {method: 'POST'}).catch(()=>{});
            this.ws?.send(JSON.stringify({type: 'reject_proposal', id: id}));
        }
        
        const card = document.getElementById(`proposal-${id}`);
        if (card) {
            card.querySelectorAll('button').forEach(b => b.disabled = true);
            card.querySelector('.reject').textContent = '⏳ דוחה...';
        }
    }

    handleHUDCommand(cmd) {
        const action = typeof cmd === 'string' ? cmd : cmd.action;
        console.log('[HUD] HUD command', action, cmd);
        
        if (action === 'dock' || action === 'side') {
            this.switchMode('side');
            this.addMessage('assistant', 'עוברת הצידה, בוס!');
        } else if (action === 'center') {
            this.switchMode('center');
        } else if (action === 'hide' || action === 'orb') {
            this.switchMode('orb');
        } else if (action === 'show') {
            this.switchMode('center');
        }
    }

    onWakeDetected(text) {
        console.log('[HUD] Wake detected', text);
        this.setListening(true, `כן בוס? שמעתי "${text}"`);
        // Ensure visible
        if (this.currentMode === 'orb') {
            this.switchMode('center');
        }
        // Pulse animation
        this.elements.listeningPulse?.classList.add('active');
        if (window.reactorAnim) window.reactorAnim.setMode('listening');
    }

    setListening(isListening, label = null) {
        this.isListening = isListening;
        this.isSpeaking = false;

        if (isListening) {
            this.elements.statusDot?.classList.add('listening');
            this.elements.statusDot?.classList.remove('speaking');
            this.elements.statusLabel?.classList.add('listening');
            this.elements.statusLabel?.classList.remove('speaking');
            this.elements.statusLabel.textContent = label || 'מקשיבה...';
            this.elements.listeningPulse?.classList.add('active');
            this.elements.visualizer?.classList.add('active');
            if (window.reactorAnim) window.reactorAnim.setMode('listening');
        } else {
            this.elements.statusDot?.classList.remove('listening');
            this.elements.statusLabel?.classList.remove('listening');
            this.elements.listeningPulse?.classList.remove('active');
            this.elements.visualizer?.classList.remove('active');
            if (!this.isSpeaking) {
                this.setIdle();
            }
        }
    }

    setSpeaking(isSpeaking, text = null, audioBase64 = null) {
        this.isSpeaking = isSpeaking;
        if (isSpeaking) {
            this.isListening = false;
            this.elements.statusDot?.classList.add('speaking');
            this.elements.statusDot?.classList.remove('listening');
            this.elements.statusLabel?.classList.add('speaking');
            this.elements.statusLabel?.classList.remove('listening');
            this.elements.statusLabel.textContent = text ? `מדברת: ${text.slice(0, 30)}...` : 'מדברת...';
            this.elements.visualizer?.classList.add('active');
            if (window.reactorAnim) window.reactorAnim.setMode('speaking');
            
            // NEW: נגן קול מה-backend אם יש base64
            if (audioBase64) {
                this.playAudioBase64(audioBase64);
            }
        } else {
            this.elements.statusDot?.classList.remove('speaking');
            this.elements.statusLabel?.classList.remove('speaking');
            this.elements.visualizer?.classList.remove('active');
            this.setIdle();
            if (window.reactorAnim) window.reactorAnim.setMode('idle');
        }
    }

    playAudioBase64(base64Audio) {
        try {
            // עצור קודם אם מנגן
            if (this.currentAudio) {
                this.currentAudio.pause();
                this.currentAudio = null;
            }
            
            console.log('[HUD] Playing audio base64, length:', base64Audio.length);
            const audio = new Audio(base64Audio);
            this.currentAudio = audio;
            
            audio.onended = () => {
                console.log('[HUD] Audio ended');
                this.setSpeaking(false);
            };
            audio.onerror = (e) => {
                console.error('[HUD] Audio play failed:', e);
                this.setSpeaking(false);
            };
            
            audio.play().catch(e => {
                console.error('[HUD] Audio play promise failed:', e);
                // Fallback: נסה עם Web Speech API בעברית אם יש
                if ('speechSynthesis' in window && base64Audio) {
                    // חלץ טקסט מההודעה האחרונה
                    console.log('[HUD] Trying Web Speech API fallback');
                }
            });
        } catch (e) {
            console.error('[HUD] playAudioBase64 error:', e);
        }
    }

    setIdle() {
        if (!this.isListening && !this.isSpeaking) {
            this.elements.statusLabel.textContent = 'מאזינה למילת הפעלה...';
            this.elements.statusDot?.classList.remove('listening', 'speaking');
            this.elements.statusLabel?.classList.remove('listening', 'speaking');
            this.elements.listeningPulse?.classList.remove('active');
            this.elements.visualizer?.classList.remove('active');
            if (window.reactorAnim) window.reactorAnim.setMode('idle');
        }
    }

    updateConnectionUI(state) {
        const dot = this.elements.connDot;
        const text = this.elements.connText;
        if (!dot || !text) return;

        dot.className = 'conn-dot ' + state;
        if (state === 'connected') {
            text.textContent = 'מחוברת - Backend פעיל';
        } else if (state === 'connecting') {
            text.textContent = 'מתחברת ל-Backend...';
        } else {
            text.textContent = 'מנותק - מנסה להתחבר...';
        }
    }

    addMessage(role, text) {
        const inner = this.elements.conversationInner;
        if (!inner) return;

        const time = new Date().toLocaleTimeString('he-IL', { hour: '2-digit', minute: '2-digit' });
        const msgDiv = document.createElement('div');
        msgDiv.className = `msg ${role}`;

        const avatar = role === 'user' ? 'אתה' : 'AJ';
        msgDiv.innerHTML = `
            <div class="msg-avatar">${avatar}</div>
            <div class="msg-content">
                <div class="msg-text">${this.escapeHtml(text)}</div>
                <div class="msg-time">${time}</div>
            </div>
        `;

        inner.appendChild(msgDiv);
        inner.scrollTop = inner.scrollHeight;

        // Keep max 50 messages
        while (inner.children.length > 50) {
            inner.removeChild(inner.firstChild);
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML.replace(/\n/g, '<br>');
    }

    sendText() {
        const input = this.elements.textInput;
        if (!input) return;
        const text = input.value.trim();
        if (!text) return;

        // חכם: רק אם שואל על מסך, שלח עם מסך - לא על "היי"
        const screenKeywords = ["מסך", "רואה", "שגיאה", "קוד", "מה פתוח", "תסרוק", "תבדוק", "מה יש"];
        const needsScreen = screenKeywords.some(kw => text.includes(kw));
        
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'text',
                text: text,
                with_screen: needsScreen
            }));
            this.addMessage('user', text);
            input.value = '';
        } else {
            this.addMessage('assistant', 'אני לא מחוברת ל-Backend כרגע, בוס. תבדוק שהשרת רץ.');
        }
    }

    triggerWake() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'manual_wake' }));
        }
        // Also try local API
        if (window.adielAPI) {
            window.adielAPI.simulateWake();
        }
        this.onWakeDetected('manual wake (כפתור)');
    }

    simulateWakeFromUI() {
        this.onWakeDetected('אדיאל ג\'וניור (לחיצה)');
    }

    askScreen() {
        const query = 'מה את רואה במסך?';
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'text',
                text: query,
                with_screen: true
            }));
            this.addMessage('user', query);
        } else {
            this.addMessage('assistant', 'צריכה חיבור ל-Backend כדי לראות מסך.');
        }
    }

    showScreenPreview(base64Image, context) {
        if (!base64Image) return;
        try {
            this.elements.previewImage.src = base64Image.startsWith('data:') ? base64Image : `data:image/jpeg;base64,${base64Image}`;
            this.elements.previewContext.textContent = context || '';
            this.elements.screenPreview.classList.remove('hidden');
            // Auto hide after 15 sec
            setTimeout(() => {
                this.elements.screenPreview.classList.add('hidden');
            }, 15000);
        } catch (e) {
            console.error('[HUD] Preview error', e);
        }
    }

    clearConversation() {
        if (this.elements.conversationInner) {
            this.elements.conversationInner.innerHTML = `
                <div class="msg assistant welcome">
                    <div class="msg-avatar">AJ</div>
                    <div class="msg-content">
                        <div class="msg-text">שיחה נוקתה, בוס. אבל אני זוכרת הכל! מה הלאה?</div>
                        <div class="msg-time">עכשיו</div>
                    </div>
                </div>
            `;
        }
        this.elements.screenPreview?.classList.add('hidden');
    }

    // === Mic Selection ===
    toggleMicPanel() {
        if (this.elements.micPanel?.classList.contains('hidden')) {
            this.elements.micPanel.classList.remove('hidden');
            this.elements.voicePanel?.classList.add('hidden');
            this.loadMicrophones();
        } else {
            this.hideMicPanel();
        }
    }

    hideMicPanel() {
        this.elements.micPanel?.classList.add('hidden');
    }

    toggleVoicePanel() {
        if (this.elements.voicePanel?.classList.contains('hidden')) {
            this.elements.voicePanel.classList.remove('hidden');
            this.elements.micPanel?.classList.add('hidden');
        } else {
            this.hideVoicePanel();
        }
    }

    hideVoicePanel() {
        this.elements.voicePanel?.classList.add('hidden');
    }

    async loadMicrophones() {
        if (!this.elements.micList) return;
        this.elements.micList.innerHTML = '<div class="loading">🎙️ טוען מיקרופונים...</div>';
        
        try {
            const backendUrl = this.wsUrl.replace('ws://', 'http://').replace('wss://', 'https://').replace('/ws', '');
            const response = await fetch(`${backendUrl}/audio/devices`);
            const data = await response.json();
            
            if (!data.inputs || data.inputs.length === 0) {
                this.elements.micList.innerHTML = '<div class="loading">לא נמצאו מיקרופונים. חבר מיקרופון ותלחץ רענן<br>אפשר להשתמש במקלדת בינתיים!</div>';
                return;
            }
            
            this.elements.micList.innerHTML = '';
            data.inputs.forEach(mic => {
                const div = document.createElement('div');
                div.className = `mic-option ${mic.selected ? 'selected' : ''}`;
                div.innerHTML = `
                    <div class="mic-name">${this.escapeHtml(mic.name)}</div>
                    <div class="mic-details">ערוצים: ${mic.input_channels} | ${mic.samplerate}Hz
                        ${mic.is_default_input ? '<span class="mic-tag default">ברירת מחדל</span>' : ''}
                        ${mic.selected ? '<span class="mic-tag">נבחר ✓</span>' : ''}
                    </div>
                `;
                div.addEventListener('click', () => this.selectMicrophone(mic.id));
                this.elements.micList.appendChild(div);
            });
            
            this.addMessage('assistant', `מצאתי ${data.inputs.length} מיקרופונים. בחר אחד שאדיאל תקשיב דרכו, בוס.`);
        } catch (e) {
            console.error('Load mics failed', e);
            this.elements.micList.innerHTML = `<div class="loading">שגיאה בטעינת מיקרופונים: ${e.message}<br>בדוק שה-Backend רץ</div>`;
        }
    }

    async selectMicrophone(deviceId) {
        try {
            const backendUrl = this.wsUrl.replace('ws://', 'http://').replace('wss://', 'https://').replace('/ws', '');
            const response = await fetch(`${backendUrl}/audio/devices/input/${deviceId}`, {method: 'POST'});
            const result = await response.json();
            
            if (result.success) {
                this.addMessage('assistant', `✅ ${result.message} - עכשיו אני מקשיבה דרכו, בוס! תגיד "אדיאל ג'וניור"`);
                this.loadMicrophones(); // רענן
                setTimeout(() => this.hideMicPanel(), 1500);
            } else {
                this.addMessage('assistant', `❌ שגיאה בבחירת מיקרופון: ${result.error}`);
            }
        } catch (e) {
            this.addMessage('assistant', `❌ שגיאה: ${e.message}`);
        }
    }

    async resetMicrophone() {
        try {
            const backendUrl = this.wsUrl.replace('ws://', 'http://').replace('wss://', 'https://').replace('/ws', '');
            const response = await fetch(`${backendUrl}/audio/devices/reset`, {method: 'POST'});
            const result = await response.json();
            this.addMessage('assistant', `↩️ ${result.message || 'חזר לברירת מחדל'}`);
            this.loadMicrophones();
        } catch (e) {
            console.error(e);
        }
    }

    async testVoice(voiceType) {
        // מנגן קול מקורי או שולח בקשת TTS ל-backend
        if (voiceType === 'original') {
            // נגן את הקולות המקוריים שיצרנו
            const voices = [
                './assets/voice_hello.mp3',
                './assets/voice_wake.mp3',
                './assets/voice_how_are_you.mp3',
                './assets/voice_on_it.mp3'
            ];
            const randomVoice = voices[Math.floor(Math.random() * voices.length)];
            try {
                const audio = new Audio(randomVoice);
                audio.play();
                this.addMessage('assistant', `✨ מנגן קול מקורי: ${randomVoice} - זה הקול שיצרתי רק בשבילך, בוס!`);
                this.setSpeaking(true, `מנגן קול מקורי...`);
                audio.onended = () => this.setSpeaking(false);
            } catch (e) {
                this.addMessage('assistant', `מנגן קול מקורי דרך Backend...`);
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify({type: 'text', text: 'בדיקת קול מקורי', with_screen: false}));
                }
            }
        } else {
            // בדיקת קול Microsoft
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({type: 'speak', text: `בדיקת קול ${voiceType}, שלום בוס! אני אדיאל ג'וניור`}));
                this.addMessage('assistant', `🔊 בודק קול ${voiceType}...`);
            } else {
                this.addMessage('assistant', 'לא מחוברת ל-Backend לבדיקת קול');
            }
        }
        
        // סמן כ-selected
        document.querySelectorAll('.voice-option').forEach(opt => opt.classList.remove('selected'));
        document.querySelector(`.voice-option[data-voice="${voiceType}"]`)?.classList.add('selected');
    }
}

// Init when DOM ready
document.addEventListener('DOMContentLoaded', () => {
    window.adielHUD = new AdielHUD();
    console.log('[HUD] Adiel Junior HUD initialized');
});
