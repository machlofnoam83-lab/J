/*
 * JARVIS — Electron preload.
 * The renderer is fully sandboxed from Node; everything it needs comes through
 * this narrow, explicitly-typed bridge.
 */
'use strict';

const { contextBridge, ipcRenderer } = require('electron');

function invoke(channel, ...args) {
    return ipcRenderer.invoke(channel, ...args).then(r => {
        if (!r || r.ok === false) throw new Error(r ? r.error : `${channel} failed`);
        return r.data;
    });
}

contextBridge.exposeInMainWorld('jarvis', {
    isElectron: true,
    platform: process.platform,

    // brain location — the renderer builds ws:// and http:// URLs from this
    getBrainUrl: () => invoke('brain:url'),
    isBrainUp: () => invoke('brain:up'),
    getAppVersion: () => invoke('app:version'),
    getPlatformInfo: () => invoke('app:platform'),
    restartBrain: () => invoke('restart:brain'),

    // window chrome (frameless → the HUD draws its own buttons)
    minimize: () => invoke('win:minimize'),
    toggleMaximize: () => invoke('win:maximize'),
    close: () => invoke('win:close'),
    toggleAlwaysOnTop: (on) => invoke('win:alwaysOnTop', on),
    toggleFullscreen: () => invoke('win:fullscreen'),

    // app lifecycle
    quit: () => invoke('app:quit'),

    // events pushed from main
    onBrainState: (cb) => ipcRenderer.on('brain-state', (_e, s) => cb(s)),
    onHotkey: (cb) => ipcRenderer.on('hotkey', (_e, h) => cb(h))
});
