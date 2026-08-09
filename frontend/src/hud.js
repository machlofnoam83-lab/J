/**
 * Adiel Junior - HUD Logic
 * Frontend brain connecting to Python backend via WebSocket
 */

class AdielHUD {
    constructor() {
        this.ws = null;
        this.wsUrl = 'ws://localhost:8765/ws';
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
        };
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
                        // Show screen preview if attached
                        if (msg.screen_image) {
                            this.showScreenPreview(msg.screen_image, msg.screen_context);
                        }
                        // Handle HUD command
                        if (msg.hud_command) {
                            this.handleHUDCommand(msg.hud_command);
                        }
                    }
                    // Visualizer speaking
                    this.setSpeaking(true, text);
                    // Auto stop speaking visual after estimated duration
                    const duration = Math.max(2000, text.length * 80);
                    setTimeout(() => this.setSpeaking(false), duration);
                }
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
                // keepalive
                break;
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

    setSpeaking(isSpeaking, text = null) {
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
        } else {
            this.elements.statusDot?.classList.remove('speaking');
            this.elements.statusLabel?.classList.remove('speaking');
            this.elements.visualizer?.classList.remove('active');
            this.setIdle();
            if (window.reactorAnim) window.reactorAnim.setMode('idle');
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

        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'text',
                text: text,
                with_screen: true
            }));
            // Optimistically add user message, will be duplicated maybe but ok
            // Actually backend will echo, so don't double - wait
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
                        <div class="msg-text">שיחה נוקתה, בוס. מה הלאה?</div>
                        <div class="msg-time">עכשיו</div>
                    </div>
                </div>
            `;
        }
        this.elements.screenPreview?.classList.add('hidden');
    }
}

// Init when DOM ready
document.addEventListener('DOMContentLoaded', () => {
    window.adielHUD = new AdielHUD();
    console.log('[HUD] Adiel Junior HUD initialized');
});
