/*
 * JARVIS — Electron main process (desktop shell).
 *
 * Responsibilities:
 *   • keep one instance alive
 *   • own the frameless HUD window (transparent, always-on-top option, custom titlebar)
 *   • spawn & supervise the Python brain (`python -m core.server`)
 *   • bridge renderer ⇄ brain over HTTP/WebSocket on 127.0.0.1
 *   • global push-to-talk hotkey, tray icon, single-instance lock
 *
 * Nothing here reaches the internet. The only network traffic is loopback.
 */
'use strict';

const { app, BrowserWindow, ipcMain, globalShortcut, Tray, Menu, shell, screen, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn, spawnSync } = require('child_process');

// Dev: the git checkout. Packaged: the Python tree we shipped in extraResources.
const ROOT = app.isPackaged
    ? path.join(process.resourcesPath, 'pyapp')
    : path.join(__dirname, '..');
const BRAIN_HOST = '127.0.0.1';
const BRAIN_PORT = Number(process.env.JARVIS_PORT || 8765);
const BRAIN_URL = `http://${BRAIN_HOST}:${BRAIN_PORT}`;

let win = null;
let tray = null;
let brain = null;
let brainReady = false;
let shuttingDown = false;

// --------------------------------------------------------------- single lock
if (!app.requestSingleInstanceLock()) {
    app.quit();
} else {
    app.on('second-instance', () => {
        if (win) {
            if (win.isMinimized()) win.restore();
            win.show();
            win.focus();
        }
    });
}

// ------------------------------------------------------------------ python --
function pythonCandidates() {
    const winExe = process.platform === 'win32';
    return [
        process.env.JARVIS_PYTHON,
        winExe ? 'py' : null,
        'python3',
        'python',
        path.join(ROOT, '.venv', winExe ? 'Scripts/python.exe' : 'bin/python3')
    ].filter(Boolean);
}

function findPython() {
    for (const cand of pythonCandidates()) {
        try {
            const r = spawnSync(cand, ['-c', 'import torch, aiohttp; print("ok")'],
                { cwd: ROOT, timeout: 60000, encoding: 'utf8' });
            if (r.status === 0) return cand;
        } catch (_) { /* try next */ }
    }
    return null;
}

function startBrain() {
    const py = findPython();
    if (!py) {
        dialog.showErrorBox('JARVIS brain not found',
            'Python with torch + aiohttp is required.\n\n' +
            'From the project folder run:\n  pip install -r requirements.txt\n\n' +
            'Then start JARVIS again.');
        return null;
    }
    console.log(`[jarvis] starting brain with ${py}`);
    const proc = spawn(py, ['-m', 'core.server', '--host', BRAIN_HOST, '--port', String(BRAIN_PORT)], {
        cwd: ROOT,
        env: Object.assign({}, process.env, { PYTHONIOENCODING: 'utf-8', PYTHONUNBUFFERED: '1' }),
        windowsHide: true
    });
    proc.stdout.on('data', d => process.stdout.write(`[brain] ${d}`));
    proc.stderr.on('data', d => process.stderr.write(`[brain!] ${d}`));
    proc.on('exit', code => {
        console.log(`[jarvis] brain exited (${code})`);
        brain = null;
        if (!shuttingDown && win) {
            win.webContents.send('brain-state', { up: false, code });
        }
    });
    return proc;
}

function waitForBrain(timeoutMs = 90000) {
    return new Promise(resolve => {
        const t0 = Date.now();
        const tick = () => {
            fetch(`${BRAIN_URL}/api/status`)
                .then(r => r.json())
                .then(() => { brainReady = true; resolve(true); })
                .catch(() => {
                    if (Date.now() - t0 > timeoutMs) return resolve(false);
                    setTimeout(tick, 700);
                });
        };
        tick();
    });
}

// ------------------------------------------------------------------ window --
function createWindow() {
    const { workAreaSize } = screen.getPrimaryDisplay();
    const width = Math.min(1500, Math.round(workAreaSize.width * 0.92));
    const height = Math.min(980, Math.round(workAreaSize.height * 0.94));

    win = new BrowserWindow({
        width, height,
        minWidth: 1024, minHeight: 640,
        show: false,
        frame: false,
        transparent: false,
        backgroundColor: '#02050a',
        autoHideMenuBar: true,
        title: 'J.A.R.V.I.S.',
        icon: path.join(__dirname, 'assets', 'icon.png'),
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false,
            sandbox: false,
            backgroundThrottling: false
        }
    });

    win.setMenuBarVisibility(false);
    // In dev the HUD lives next to the code; packaged it ships inside the asar.
    win.loadFile(app.isPackaged ? path.join(__dirname, '..', 'ui', 'index.html')
                                : path.join(ROOT, 'ui', 'index.html'));

    win.once('ready-to-show', () => {
        win.show();
        win.focus();
    });

    win.webContents.setWindowOpenHandler(({ url }) => {
        if (url.startsWith(BRAIN_URL) || url.startsWith('file://')) return { action: 'allow' };
        shell.openExternal(url);          // never navigate the HUD away
        return { action: 'deny' };
    });

    win.on('closed', () => { win = null; });
}

function createTray() {
    const iconPath = path.join(__dirname, 'assets', 'icon.png');
    if (!fs.existsSync(iconPath)) return;   // tray is a nicety, not a requirement
    try { tray = new Tray(iconPath); } catch (_) { return; }
    tray.setToolTip('J.A.R.V.I.S. — Just A Rather Very Intelligent System');
    tray.setContextMenu(Menu.buildFromTemplate([
        { label: 'Open HUD', click: () => { if (win) { win.show(); win.focus(); } } },
        { label: 'Push-to-talk (Ctrl+Shift+Space)', enabled: false },
        { type: 'separator' },
        { label: 'Quit JARVIS', click: () => quit() }
    ]));
    tray.on('click', () => { if (win) { win.show(); win.focus(); } });
}

// --------------------------------------------------------------------- ipc --
function ipc(channel, handler) {
    ipcMain.handle(channel, async (event, ...args) => {
        try { return { ok: true, data: await handler(...args) }; }
        catch (err) { return { ok: false, error: String(err && err.message || err) }; }
    });
}

ipc('brain:url', () => BRAIN_URL);
ipc('brain:up', () => brainReady);
ipc('app:version', () => app.getVersion());
ipc('app:platform', () => ({ platform: process.platform, arch: process.arch, versions: process.versions }));

ipc('win:minimize', () => win && win.minimize());
ipc('win:maximize', () => {
    if (!win) return false;
    win.isMaximized() ? win.unmaximize() : win.maximize();
    return win.isMaximized();
});
ipc('win:close', () => win && win.hide());
ipc('win:alwaysOnTop', (_, on) => { if (win) win.setAlwaysOnTop(!!on); return win ? win.isAlwaysOnTop() : false; });
ipc('win:fullscreen', () => {
    if (!win) return false;
    win.setFullScreen(!win.isFullScreen());
    return win.isFullScreen();
});

ipc('restart:brain', async () => {
    if (brain) { brain.kill(); brain = null; }
    brainReady = false;
    brain = startBrain();
    const ok = await waitForBrain();
    if (win) win.webContents.send('brain-state', { up: ok });
    return ok;
});

ipc('app:quit', () => quit());

// ------------------------------------------------------------------ hotkey --
function registerHotkeys() {
    globalShortcut.register('CommandOrControl+Shift+Space', () => {
        if (win) { win.show(); win.focus(); win.webContents.send('hotkey', { name: 'push-to-talk' }); }
    });
    globalShortcut.register('CommandOrControl+Shift+J', () => {
        if (win) { win.isAlwaysOnTop() ? win.setAlwaysOnTop(false) : win.setAlwaysOnTop(true); }
    });
    globalShortcut.register('CommandOrControl+Shift+Escape', () => {
        if (win) win.webContents.send('hotkey', { name: 'kill-switch' });
    });
}

function quit() {
    if (shuttingDown) return;
    shuttingDown = true;
    try { globalShortcut.unregisterAll(); } catch (_) {}
    try { if (brain) { brain.kill(); brain = null; } } catch (_) {}
    try { if (tray) tray.destroy(); } catch (_) {}
    app.quit();
}

// ------------------------------------------------------------------- boot ---
app.whenReady().then(async () => {
    if (process.platform === 'win32') app.setAppUserModelId('com.jarvis.hud');
    createWindow();
    createTray();
    registerHotkeys();

    brain = startBrain();
    const ok = await waitForBrain();
    brainReady = ok;
    if (win) win.webContents.send('brain-state', { up: ok, url: BRAIN_URL });
    if (!ok) console.error('[jarvis] brain did not come up — HUD will show offline state');
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') quit(); else if (win === null) quit();
});
app.on('activate', () => { if (win === null) createWindow(); });
app.on('before-quit', () => { shuttingDown = true; try { if (brain) brain.kill(); } catch (_) {} });
process.on('SIGINT', quit);
process.on('SIGTERM', quit);
