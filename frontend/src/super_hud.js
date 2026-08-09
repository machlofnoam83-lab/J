/**
 * Super HUD - ניהול משימות סוכן-על
 * מציג ומנהל את כל המשימות החדשות
 */

class SuperHUD {
    constructor(mainHUD) {
        this.mainHUD = mainHUD;
        this.tasks = [];
        this.init();
    }

    init() {
        // הוסף כפתורי משימות ל-input-actions אם לא קיימים
        const actions = document.querySelector('.input-actions');
        if (actions && !document.getElementById('btnTasks')) {
            const taskBtn = document.createElement('button');
            taskBtn.className = 'action-btn';
            taskBtn.id = 'btnTasks';
            taskBtn.innerHTML = '🚀 משימות';
            taskBtn.title = 'סוכן-על - כל המשימות';
            taskBtn.addEventListener('click', () => this.toggleTaskPanel());
            actions.appendChild(taskBtn);
            
            const routineBtn = document.createElement('button');
            routineBtn.className = 'action-btn';
            routineBtn.id = 'btnRoutines';
            routineBtn.innerHTML = '⏰ שגרה';
            routineBtn.title = 'רוטינות אוטומטיות';
            routineBtn.addEventListener('click', () => this.showRoutines());
            actions.appendChild(routineBtn);
        }

        // צור פאנל משימות אם לא קיים
        if (!document.getElementById('tasksPanel')) {
            const hudMain = document.querySelector('.hud-main');
            const inputArea = document.querySelector('.input-area');
            
            const panel = document.createElement('div');
            panel.id = 'tasksPanel';
            panel.className = 'settings-panel hidden';
            panel.innerHTML = `
                <div class="settings-header">
                    <span>🚀 סוכן-על - ניהול משימות</span>
                    <button class="icon-btn" id="btnCloseTasks">✕</button>
                </div>
                <div class="tasks-categories">
                    <button class="category-btn active" data-cat="all">הכל</button>
                    <button class="category-btn" data-cat="shopping">🛒 קניות</button>
                    <button class="category-btn" data-cat="travel">✈️ חופשות</button>
                    <button class="category-btn" data-cat="research">🔬 מחקר</button>
                    <button class="category-btn" data-cat="email">📧 מייל</button>
                    <button class="category-btn" data-cat="files">📁 קבצים</button>
                    <button class="category-btn" data-cat="computer">🖥️ מחשב</button>
                </div>
                <div class="settings-content" id="tasksList">
                    <div class="task-quick-actions">
                        <button class="quick-task-btn" data-task="תקנה לי אוזניות הכי זול">🛒 קנה הכי זול</button>
                        <button class="quick-task-btn" data-task="תזמין טיסה מתל אביב ללונדון שבוע הבא">✈️ חפש טיסה</button>
                        <button class="quick-task-btn" data-task="תחקור על בינה מלאכותית">🔬 מחקר AI</button>
                        <button class="quick-task-btn" data-task="תבדוק מיילים דחופים">📧 מיילים דחופים</button>
                        <button class="quick-task-btn" data-task="תמצא זמן לפגישה מחר">📅 מצא זמן</button>
                        <button class="quick-task-btn" data-task="תארגן קבצים בהורדות">📁 ארגן קבצים</button>
                        <button class="quick-task-btn" data-task="תפתח ספוטיפיי ותנגן מוזיקה">🎵 נגן מוזיקה</button>
                        <button class="quick-task-btn" data-task="תכין דוח מנהלים">📊 דוח</button>
                    </div>
                    <div class="tasks-history" id="tasksHistory">
                        <div class="loading">אין משימות עדיין - תגיד לאדיאל משהו כמו "תקנה לי..."</div>
                    </div>
                </div>
            `;
            
            hudMain.insertBefore(panel, inputArea);
            
            document.getElementById('btnCloseTasks')?.addEventListener('click', () => this.hideTaskPanel());
            
            // Quick tasks
            panel.querySelectorAll('.quick-task-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const task = e.target.getAttribute('data-task');
                    this.executeQuickTask(task);
                });
            });
            
            // Categories
            panel.querySelectorAll('.category-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    panel.querySelectorAll('.category-btn').forEach(b => b.classList.remove('active'));
                    e.target.classList.add('active');
                    const cat = e.target.getAttribute('data-cat');
                    this.filterTasks(cat);
                });
            });
        }
    }

    toggleTaskPanel() {
        const panel = document.getElementById('tasksPanel');
        if (!panel) return;
        
        if (panel.classList.contains('hidden')) {
            panel.classList.remove('hidden');
            document.getElementById('micPanel')?.classList.add('hidden');
            document.getElementById('voicePanel')?.classList.add('hidden');
            this.loadTaskHistory();
        } else {
            panel.classList.add('hidden');
        }
    }

    hideTaskPanel() {
        document.getElementById('tasksPanel')?.classList.add('hidden');
    }

    async executeQuickTask(taskText) {
        const input = document.getElementById('textInput');
        if (input) {
            input.value = taskText;
        }
        
        // שלח כמשימה
        if (this.mainHUD.ws && this.mainHUD.ws.readyState === WebSocket.OPEN) {
            try {
                // נסה קודם כ-task
                const backendUrl = this.mainHUD.wsUrl.replace('ws://', 'http://').replace('wss://', 'https://').replace('/ws', '');
                const response = await fetch(`${backendUrl}/tasks/execute`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text: taskText})
                });
                const result = await response.json();
                
                this.mainHUD.addMessage('assistant', `🚀 סוכן-על: ${result.message || 'מבצע משימה...'}`);
                
                if (result.result) {
                    const details = JSON.stringify(result.result, null, 2).substring(0, 500);
                    this.mainHUD.addMessage('assistant', `📋 פרטים: ${details}...`);
                }
                
                this.addToHistory(taskText, result);
                
            } catch (e) {
                // Fallback לטקסט רגיל
                this.mainHUD.sendText();
            }
        }
        
        this.hideTaskPanel();
    }

    addToHistory(taskText, result) {
        this.tasks.unshift({
            text: taskText,
            result: result,
            time: new Date().toLocaleTimeString('he-IL')
        });
        
        const history = document.getElementById('tasksHistory');
        if (!history) return;
        
        if (this.tasks.length === 1) {
            history.innerHTML = '';
        }
        
        const div = document.createElement('div');
        div.className = 'task-history-item';
        div.innerHTML = `
            <div class="task-history-text">🚀 ${this.mainHUD.escapeHtml(taskText)}</div>
            <div class="task-history-result">${this.mainHUD.escapeHtml((result.message || '').substring(0, 100))}</div>
            <div class="task-history-time">${this.tasks[0].time} - ${result.success ? '✅' : '❌'}</div>
        `;
        history.insertBefore(div, history.firstChild);
    }

    async loadTaskHistory() {
        try {
            const backendUrl = this.mainHUD.wsUrl.replace('ws://', 'http://').replace('wss://', 'https://').replace('/ws', '');
            const response = await fetch(`${backendUrl}/tasks/history`);
            const data = await response.json();
            
            if (data.tasks && data.tasks.length > 0) {
                const history = document.getElementById('tasksHistory');
                if (history) {
                    history.innerHTML = '';
                    data.tasks.slice(0, 10).forEach(task => {
                        const div = document.createElement('div');
                        div.className = 'task-history-item';
                        div.innerHTML = `
                            <div class="task-history-text">${this.mainHUD.escapeHtml(task.description)}</div>
                            <div class="task-history-time">${task.type} - ${task.status}</div>
                        `;
                        history.appendChild(div);
                    });
                }
            }
        } catch (e) {
            console.log('Load history failed', e);
        }
    }

    filterTasks(category) {
        // סינון לפי קטגוריה - פשוט מציג הכל כרגע
        console.log('Filter', category);
    }

    async showRoutines() {
        try {
            const backendUrl = this.mainHUD.wsUrl.replace('ws://', 'http://').replace('wss://', 'https://').replace('/ws', '');
            const response = await fetch(`${backendUrl}/tasks/routines`);
            const data = await response.json();
            
            let msg = `⏰ יש ${data.routines.length} רוטינות:\n`;
            data.routines.forEach(r => {
                msg += `\n• ${r.name}: ${r.description} (${r.schedule})`;
            });
            msg += `\n\nתגיד "תריץ שגרת בוקר" כדי להפעיל!`;
            
            this.mainHUD.addMessage('assistant', msg);
            
        } catch (e) {
            this.mainHUD.addMessage('assistant', 'לא הצלחתי לטעון רוטינות');
        }
    }
}

// Auto init when HUD ready
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        if (window.adielHUD) {
            window.superHUD = new SuperHUD(window.adielHUD);
            console.log('[SuperHUD] Loaded - Task agent ready');
        }
    }, 1000);
});
