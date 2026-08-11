/**
 * Adiel Junior - Electron Main Process
 * חלון שקוף ללא מסגרת בסגנון Iron Man HUD
 */
const { app, BrowserWindow, ipcMain, screen, globalShortcut, Tray, Menu } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

let mainWindow = null;
let tray = null;
let backendProcess = null;
let isDev = process.argv.includes('--dev');
let currentMode = 'center'; // center / side / orb

const WINDOW_MODES = {
    center: { width: 520, height: 680, x: null, y: null, resizable: false },
    side: { width: 400, height: 720, x: 'right', y: 'center' },
    orb: { width: 140, height: 140, x: 'bottom-right', y: null }
};

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

function startBackend() {
    const backendInfo = getBackendPath();
    console.log('[Main] Backend path:', backendInfo);

    try {
        if (backendInfo.type === 'exe') {
            backendProcess = spawn(backendInfo.path, [], {
                cwd: path.dirname(backendInfo.path),
                detached: false,
                stdio: 'pipe'
            });
        } else {
            // Dev mode - python
            const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
            backendProcess = spawn(pythonCmd, [backendInfo.path], {
                cwd: path.join(__dirname, '..', 'backend'),
                detached: false,
                stdio: 'pipe',
                env: { ...process.env, PORT: '8765' }
            });
        }

        backendProcess.stdout?.on('data', (data) => {
            console.log(`[Backend] ${data}`);
        });
        backendProcess.stderr?.on('data', (data) => {
            console.error(`[Backend ERR] ${data}`);
        });
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

function createWindow(mode = 'center') {
    const primaryDisplay = screen.getPrimaryDisplay();
    const { width: screenWidth, height: screenHeight } = primaryDisplay.workAreaSize;

    const modeConfig = WINDOW_MODES[mode] || WINDOW_MODES.center;
    let winX, winY;

    if (mode === 'center') {
        winX = Math.round((screenWidth - modeConfig.width) / 2);
        winY = Math.round((screenHeight - modeConfig.height) / 2);
    } else if (mode === 'side') {
        winX = screenWidth - modeConfig.width - 20;
        winY = Math.round((screenHeight - modeConfig.height) / 2);
    } else if (mode === 'orb') {
        winX = screenWidth - modeConfig.width - 30;
        winY = screenHeight - modeConfig.height - 60;
    }

    mainWindow = new BrowserWindow({
        width: modeConfig.width,
        height: modeConfig.height,
        x: winX,
        y: winY,
        frame: false,
        transparent: true,
        alwaysOnTop: true,
        resizable: modeConfig.resizable || false,
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
        console.log(`[Main] HUD shown in ${mode} mode at ${winX},${winY}`);
    });

    // DevTools ב-dev
    if (isDev) {
        mainWindow.webContents.openDevTools({ mode: 'detach' });
    }

    // אפשר לגרור את החלון ע"י האזור העליון
    mainWindow.setMovable(true);

    currentMode = mode;
}

function switchMode(newMode) {
    if (!mainWindow || currentMode === newMode) return;

    const primaryDisplay = screen.getPrimaryDisplay();
    const { width: screenWidth, height: screenHeight } = primaryDisplay.workAreaSize;
    const config = WINDOW_MODES[newMode];

    let newX, newY;
    if (newMode === 'center') {
        newX = Math.round((screenWidth - config.width) / 2);
        newY = Math.round((screenHeight - config.height) / 2);
    } else if (newMode === 'side') {
        newX = screenWidth - config.width - 20;
        newY = Math.round((screenHeight - config.height) / 2);
    } else if (newMode === 'orb') {
        newX = screenWidth - config.width - 30;
        newY = screenHeight - config.height - 60;
    }

    currentMode = newMode;

    // אנימציה - שינוי גודל ומיקום
    mainWindow.setBounds({
        x: newX,
        y: newY,
        width: config.width,
        height: config.height
    }, true);

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

ipcMain.on('hud-drag-start', () => {
    // Electron לא מאפשר drag programmaticaly בקלות, אבל נשאיר API
});

ipcMain.handle('get-current-mode', () => currentMode);

ipcMain.handle('get-backend-status', async () => {
    return { running: backendProcess && !backendProcess.killed };
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
    if (backendProcess) {
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
