/**
 * Preload script - Secure bridge
 */
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('adielAPI', {
    // HUD control
    switchMode: (mode) => ipcRenderer.send('hud-switch-mode', mode),
    minimize: () => ipcRenderer.send('hud-minimize'),
    close: () => ipcRenderer.send('hud-close'),
    simulateWake: () => ipcRenderer.send('simulate-wake'),
    getCurrentMode: () => ipcRenderer.invoke('get-current-mode'),
    getBackendStatus: () => ipcRenderer.invoke('get-backend-status'),
    
    // Events from main
    onModeChanged: (callback) => ipcRenderer.on('hud-mode-changed', (_, mode) => callback(mode)),
    onSimulateWake: (callback) => ipcRenderer.on('simulate-wake', () => callback()),
    onBackendStatus: (callback) => ipcRenderer.on('backend-status', (_, status) => callback(status)),

    // Platform
    platform: process.platform
});

// Version
contextBridge.exposeInMainWorld('adielVersion', '2.2.0');
