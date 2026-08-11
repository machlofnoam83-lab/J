/**
 * Adiel Junior - Electron Main Process v2.2
 * חלון HUD בסגנון Iron Man - center / side / orb
 *
 * v2.2:
 *  - בדיקת Backend חי לפני spawn - אין יותר שני Backendים שרבים על פורט 8765 (WinError 10048)
 *  - PYTHONUTF8=1 לתהליך הבן - אין יותר UnicodeEncodeError באימוג'י של הלוגים
 *  - גדלי חלון מתוקנים: center רחב ונשלף-גודל, side צר בלי חיתוכים, orb קטן
 *  - לוג stderr של uvicorn (INFO) לא מסומן יותר כ־[Backend ERR] אלא אם זו שגיאה אמיתית
 */
const { app, BrowserWindow, ipcMain, screen, globalShortcut, Tray, Menu } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');
const http = require('http');

let mainWindow = null;
let tray = null;
let backendProcess = null;
let backendExternal = false; // Backend רץ מבחוץ (run.bat) - לא להרוג אותו ביציאה
let isDev = process.argv.includes('--dev');
let currentMode = 'center'; // center / side / orb

// v2.2: גדלים נכונים! center רחב ל־3 עמודות, side צר בלי חיתוך תוכן
const WINDOW_MODES = {
    center: { width: 1150, height: 780, minWidth: 1020, minHeight: 680, resizable: true },
    side:   { width: 430,  height: 800, minWidth: 380,  minHeight: 560, resizable: true },
    orb:    { width: 150,  height: 150, minWidth: 150,  minHeight: 150, resizable: false }
};

const BACKEND_PORT = 8765;

/** בדיקה אם Backend חי כבר (run.bat / עותק קודם / תהליך ידני) */
function isBackendAlive(timeoutMs = 1200) {
    return new Promise((resolve) => {
        const req = http.get({ host: '127.0.0.1', port: BACKEND_PORT, path: '/health', timeout: timeoutMs }, (res) => {
            res.resume();
            resolve(res.statusCode >= 200 && res.statusCode < 500);
        });
        req.on('timeout', () => { req.destroy(); resolve(false); });
        req.on('error', () => resolve(false));
    });
}

function getBackendPath() {
    // ב-production, backend.exe נמצא ב-resources
    const possiblePaths = [
        path.join(process.resourcesPath, 'backend', 'adiel_backend.exe'),
        path.join(process.resourcesPath, 'backend', 'adiel_backend'),
        path.join(__dirname, '..', 'backend', 'dist', 'adiel_backend.exe'),
        path.join(__dirname, '..', 'backend', 'dist', 'adiel_backend'),
        null // fallback to python
    ];

    for (const p of possiblePaths) {
        if (p && fs.existsSync(p)) {
            return { type: 'exe', path: p };
        }
    }
    return { type: 'python', path: path.join(__dirname, '..', 'backend', 'main.py') };
}

/** סיווג שורות לוג של ה-Backend: stderr של uvicorn הוא INFO רגיל, לא שגיאה */
function logBackendLine(chunk, isErr) {
    const lines = String(chunk).split(/\r?\n/).filter(l => l.trim());
    for (const line of lines) {
        const realError = /error|traceback|exception|failed|critical/i.test(line) && !/error handler|0 error/i.test(line);
        if (isErr && realError) console.error(`[Backend ERR] ${line}`);
        else console.log(`[Backend] ${line}`);
    }
}

async function startBackend() {
    // v2.2: אל תפעיל עותק שני אם כבר יש Backend חי!
    if (await isBackendAlive()) {
        console.log(`[Main] Backend already alive on :${BACKEND_PORT} - reusing it (no double-start)`);
        backendExternal = true;
        if (mainWindow) mainWindow.webContents.send('backend-status', { running: true, external: true });
        return;
    }

    const backendInfo = getBackendPath();
    console.log('[Main] Backend path:', backendInfo);

    // v2.2: UTF-8 לתהליך הבן - מונע UnicodeEncodeError באימוג'י הלוגים ב-Windows
    const childEnv = {
        ...process.env,
        PORT: String(BACKEND_PORT),
        PYTHONUTF8: '1',
        PYTHONIOENCODING: 'utf-8'
    };

    try {
        if (backendInfo.type === 'exe') {
            backendProcess = spawn(backendInfo.path, [], {
                cwd: path.dirname(backendInfo.path),
                detached: false,
                stdio: 'pipe',
                env: childEnv
            });
        } else {
            // Dev mode - python (מעדיפים venv של הפרויקט אם קיים)
            const venvPython = process.platform === 'win32'
                ? path.join(__dirname, '..', 'venv', 'Scripts', 'python.exe')
                : path.join(__dirname, '..', 'venv', 'bin', 'python');
            const pythonCmd = fs.existsSync(venvPython)
                ? venvPython
                : (process.platform === 'win32' ? 'python' : 'python3');
            backendProcess = spawn(pythonCmd, [backendInfo.path], {
                cwd: path.join(__dirname, '..', 'backend'),
                detached: false,
                stdio: 'pipe',
                env: childEnv
            });
        }

        backendProcess.stdout?.on('data', (d) => logBackendLine(d, false));
        backendProcess.stderr?.on('data', (d) => logBackendLine(d, true));
        backendProcess.on('close', (code) => {
            console.log(`[Backend] Exited with code ${code}`);
            if (mainWindow) {
                mainWindow.webContents.send('backend-status', { running: false, code });
            }
        });
        console.log('[Main] Backend process started PID', backendProcess.pid);
    } catch (e) {
        console.error('[Main] Failed to start backend:', e);
    }
}

function modeBounds(mode) {
    const primaryDisplay = screen.getPrimaryDisplay();
    const { width: screenWidth, height: screenHeight } = primaryDisplay.workAreaSize;
    const c = WINDOW_MODES[mode] || WINDOW_MODES.center;

    if (mode === 'center') {
        return { x: Math.round((screenWidth - c.width) / 2), y: Math.round((screenHeight - c.height) / 2), width: c.width, height: c.height };
    }
    if (mode === 'side') {
        return { x: screenWidth - c.width - 12, y: Math.round((screenHeight - c.height) / 2), width: c.width, height: c.height };
    }
    // orb - כדור קטן בפינה
    return { x: screenWidth - c.width - 28, y: screenHeight - c.height - 64, width: c.width, height: c.height };
}

function createWindow(mode = 'center') {
    const modeConfig = WINDOW_MODES[mode] || WINDOW_MODES.center;
    const b = modeBounds(mode);

    mainWindow = new BrowserWindow({
        width: b.width,
        height: b.height,
        x: b.x,
        y: b.y,
        frame: false,
        transparent: true,
        alwaysOnTop: true,
        resizable: modeConfig.resizable !== false,
        minWidth: modeConfig.minWidth,
        minHeight: modeConfig.minHeight,
        hasShadow: false,
        skipTaskbar: false,
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            preload: path.join(__dirname, 'preload.js'),
            backgroundThrottling: false
        },
        show: false,
        // Windows 11 Mica / Acrylic
        backgroundColor: '#00000000',
        vibrancy: 'ultra-dark',
        visualEffectState: 'active'
    });

    // לטעון HUD - נסה מתקדם קודם, אז רגיל
    const advancedPath = path.join(__dirname, 'src', 'advanced_hud.html');
    const regularPath = path.join(__dirname, 'src', 'index.html');
    const hudPath = fs.existsSync(advancedPath) ? advancedPath : regularPath;
    console.log(`[Main] Loading HUD from ${hudPath}`);
    mainWindow.loadFile(hudPath);

    mainWindow.once('ready-to-show', () => {
        mainWindow.show();
        mainWindow.setAlwaysOnTop(true, 'screen-saver');
        console.log(`[Main] HUD shown in ${mode} mode`);
        mainWindow.webContents.send('hud-mode-changed', mode);
    });

    // DevTools ב-dev
    if (isDev) {
        mainWindow.webContents.openDevTools({ mode: 'detach' });
    }

    mainWindow.setMovable(true);
    currentMode = mode;
}

function switchMode(newMode) {
    if (!mainWindow || currentMode === newMode) return;

    const config = WINDOW_MODES[newMode];
    if (!config) return;
    const b = modeBounds(newMode);

    currentMode = newMode;

    // min-size משתנה לפי מצב (orb קטן, center רחב)
    mainWindow.setMinimumSize(config.minWidth || 150, config.minHeight || 150);
    mainWindow.setResizable(config.resizable !== false);

    // אנימציה - שינוי גודל ומיקום
    mainWindow.setBounds({ x: b.x, y: b.y, width: b.width, height: b.height }, true);

    mainWindow.webContents.send('hud-mode-changed', newMode);
    console.log(`[Main] Switched to ${newMode}`);
}

// IPC Handlers
ipcMain.on('hud-switch-mode', (event, mode) => {
    if (WINDOW_MODES[mode]) {
        switchMode(mode);
    }
});

ipcMain.on('hud-minimize', () => {
    if (mainWindow) mainWindow.minimize();
});

ipcMain.on('hud-close', () => {
    app.quit();
});

ipcMain.handle('get-current-mode', () => currentMode);

ipcMain.handle('get-backend-status', async () => {
    if (backendExternal) return { running: true, external: true };
    return { running: !!(backendProcess && !backendProcess.killed) };
});

ipcMain.on('simulate-wake', () => {
    if (mainWindow) {
        mainWindow.webContents.send('simulate-wake');
        // החזר למרכז אם ב-orb
        if (currentMode === 'orb') {
            switchMode('center');
        }
        mainWindow.show();
        mainWindow.focus();
    }
});

app.whenReady().then(() => {
    // Windows: AppUserModelId להתראות/אייקון נכון
    if (process.platform === 'win32') {
        app.setAppUserModelId('com.adiel.junior');
    }

    createWindow('center');
    startBackend();

    // Tray icon
    try {
        const iconPath = path.join(__dirname, 'src', 'assets', 'icon.png');
        if (fs.existsSync(iconPath)) {
            tray = new Tray(iconPath);
            const contextMenu = Menu.buildFromTemplate([
                { label: 'Adiel Junior - מרכז', click: () => switchMode('center') },
                { label: 'צד המסך', click: () => switchMode('side') },
                { label: 'כדור קטן', click: () => switchMode('orb') },
                { type: 'separator' },
                { label: 'התעוררי', click: () => { if (mainWindow) { mainWindow.webContents.send('simulate-wake'); mainWindow.show(); } } },
                { label: 'יציאה', click: () => app.quit() }
            ]);
            tray.setToolTip('Adiel Junior - אדיאל ג\'וניור');
            tray.setContextMenu(contextMenu);
            tray.on('click', () => {
                if (mainWindow) {
                    if (mainWindow.isVisible()) mainWindow.hide();
                    else mainWindow.show();
                }
            });
        }
    } catch (e) {
        console.log('[Main] Tray not available:', e.message);
    }

    // Global shortcut - Ctrl+Shift+A להתעוררות
    try {
        globalShortcut.register('CommandOrControl+Shift+A', () => {
            console.log('[Main] Global shortcut pressed - wake');
            if (mainWindow) {
                if (currentMode === 'orb') switchMode('center');
                mainWindow.show();
                mainWindow.focus();
                mainWindow.webContents.send('simulate-wake');
            }
        });
    } catch (e) {
        console.log('[Main] Global shortcut failed:', e.message);
    }

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) createWindow(currentMode);
    });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('will-quit', () => {
    globalShortcut.unregisterAll();
    // לא הורגים Backend חיצוני (run.bat) - רק כזה שאנחנו יצרנו
    if (backendProcess && !backendExternal) {
        try {
            backendProcess.kill();
        } catch {}
    }
});

// Handle frontend HUD commands
ipcMain.on('hud-command-from-backend', (event, command) => {
    const action = command.action || command;
    if (action === 'dock' || action === 'side') switchMode('side');
    else if (action === 'center') switchMode('center');
    else if (action === 'hide' || action === 'orb') switchMode('orb');
});
