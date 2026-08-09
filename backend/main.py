"""
Adiel Junior - Backend Server
FastAPI + WebSockets + Audio + Vision + Brain
השרת המרכזי שמריץ את כל המערכת

Run: python main.py
Or: uvicorn main:app --host 0.0.0.0 --port 8765 --reload
"""
import os
import sys
import asyncio
import json
import base64
import time
from datetime import datetime
from typing import List, Dict, Optional

# הוסף נתיב
sys.path.insert(0, os.path.dirname(__file__))

# Auto Recovery - מערכת תיקון שגיאות אוטומטית
try:
    from core.error_recovery import get_recovery, auto_recover
    recovery = get_recovery()
    print("[Main] 🛡️  Auto Recovery System loaded - Self-Healing Active")
except Exception as e:
    print(f"[Main] Recovery system not available: {e}")
    recovery = None
    # Fallback decorator
    def auto_recover(fallback=None, context=""):
        def decorator(func):
            return func
        return decorator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Modules with auto-fix fallback
from core.brain import AdielBrain
try:
    from audio.wake_word import WakeWordDetector
except Exception as e:
    print(f"[Main] Wake word import failed, will use mock: {e}")
    WakeWordDetector = None

try:
    from audio.stt import HebrewSTT
except Exception as e:
    print(f"[Main] STT import failed: {e}")
    HebrewSTT = None

try:
    from audio.tts import get_tts_engine, HebrewTTS
except Exception as e:
    print(f"[Main] TTS import failed: {e}")
    get_tts_engine = lambda: None
    HebrewTTS = None

try:
    from vision.screen import get_vision_engine
except Exception as e:
    print(f"[Main] Vision import failed: {e}")
    get_vision_engine = lambda: None

try:
    from tools.system_tools import get_system_tools
except Exception as e:
    print(f"[Main] System tools import failed: {e}")
    get_system_tools = lambda: None

# Init FastAPI
app = FastAPI(title="Adiel Junior Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engines
brain: Optional[AdielBrain] = None
stt_engine: Optional[HebrewSTT] = None
tts_engine: Optional[HebrewTTS] = None
vision_engine = None
system_tools = None
wake_detector: Optional[WakeWordDetector] = None

# WebSocket Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[WS] Client connected, total {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"[WS] Client disconnected, total {len(self.active_connections)}")

    async def send_personal(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message, ensure_ascii=False))
        except Exception as e:
            print(f"[WS] Send personal failed: {e}")

    async def broadcast(self, message: dict):
        if not self.active_connections:
            return
        data = json.dumps(message, ensure_ascii=False)
        stale = []
        for conn in self.active_connections:
            try:
                await conn.send_text(data)
            except:
                stale.append(conn)
        for s in stale:
            self.disconnect(s)

manager = ConnectionManager()

# State
class AppState:
    def __init__(self):
        self.is_listening = False
        self.is_speaking = False
        self.last_wake_time = 0
        self.conversation_active_until = 0  # זמן שהשיחה פעילה (30 שניות אחרי wake)

    def is_conversation_active(self):
        return time.time() < self.conversation_active_until

    def activate_conversation(self, duration=30):
        self.conversation_active_until = time.time() + duration

app_state = AppState()

# Startup
@app.on_event("startup")
async def startup_event():
    global brain, stt_engine, tts_engine, vision_engine, system_tools, wake_detector

    print("="*60)
    print("  אדיאל ג'וניור - Backend מתחיל...")
    print("="*60)

    # בנה engines - lazy loading כדי לא להיתקע
    try:
        brain = AdielBrain()
        print("[Startup] Brain OK")
    except Exception as e:
        print(f"[Startup] Brain failed: {e}")
        import traceback; traceback.print_exc()

    try:
        tts_engine = get_tts_engine()
        print("[Startup] TTS OK")
    except Exception as e:
        print(f"[Startup] TTS failed: {e}")

    try:
        stt_engine = HebrewSTT(model_size=os.getenv("WHISPER_MODEL", "small"))
        print("[Startup] STT OK")
    except Exception as e:
        print(f"[Startup] STT failed: {e}")

    try:
        vision_engine = get_vision_engine()
        print("[Startup] Vision OK")
    except Exception as e:
        print(f"[Startup] Vision failed: {e}")

    try:
        system_tools = get_system_tools()
        print("[Startup] SystemTools OK")
    except Exception as e:
        print(f"[Startup] SystemTools failed: {e}")

    # Wake word detector - עם callback
    def on_wake_detected(text: str):
        print(f"[Main] Wake word callback: {text}")
        asyncio.run_coroutine_threadsafe(handle_wake_word(text), asyncio.get_event_loop())

    try:
        wake_detector = WakeWordDetector(on_wake=on_wake_detected)
        wake_detector.start()
        print("[Startup] WakeWord detector started - listening for 'אדיאל ג'וניור'")
    except Exception as e:
        print(f"[Startup] WakeWord failed: {e}")

    print("="*60)
    print("  Backend מוכן! ws://localhost:8765/ws")
    print("  HTTP: http://localhost:8765/status")
    print("="*60)

@app.on_event("shutdown")
async def shutdown_event():
    global wake_detector
    if wake_detector:
        wake_detector.stop()
    print("[Shutdown] Adiel Junior closing...")

# Models
class SpeakRequest(BaseModel):
    text: str
    play: bool = True

class TextInputRequest(BaseModel):
    text: str
    with_screen: bool = True

# Routes
@app.get("/")
async def root():
    return {"name": "Adiel Junior Backend", "status": "running", "version": "1.0", "hebrew": "אדיאל ג'וניור - עוזרת אישית"}

@app.get("/status")
async def status():
    base = {
        "brain": brain is not None,
        "stt": stt_engine is not None and getattr(stt_engine, 'model', None) is not None,
        "tts": tts_engine is not None,
        "vision": vision_engine is not None,
        "wake_detector": wake_detector.running if wake_detector else False,
        "connections": len(manager.active_connections),
        "conversation_active": app_state.is_conversation_active(),
        "time": datetime.now().isoformat()
    }
    # Add recovery health if available
    if recovery:
        try:
            base["health"] = recovery.get_health_report()
            base["auto_fix"] = "active"
        except:
            base["auto_fix"] = "error"
    else:
        base["auto_fix"] = "disabled"
    return base

@app.get("/health")
async def health_check():
    """בדיקת בריאות מפורטת עם תיקון אוטומטי"""
    report = {
        "status": "ok",
        "checks": {},
        "auto_fix_log": None
    }
    
    # בדוק כל רכיב
    checks = {
        "brain": brain is not None,
        "tts": tts_engine is not None,
        "vision": vision_engine is not None,
        "stt": stt_engine is not None,
        "wake_word": wake_detector is not None and wake_detector.running if wake_detector else False
    }
    report["checks"] = checks
    
    if not all(checks.values()):
        report["status"] = "degraded"
        report["message"] = "חלק מהרכיבים לא פעילים, אבל המערכת ממשיכה עם fallback"
    
    if recovery:
        report["auto_fix_log"] = recovery.get_health_report()
    
    return report

@app.post("/fix")
async def trigger_auto_fix():
    """טריגר ידני לתיקון אוטומטי"""
    log = []
    fixed = []
    
    # נסה לתקן רכיבים חסרים
    global brain, tts_engine, vision_engine, system_tools
    
    if brain is None:
        try:
            brain = AdielBrain()
            fixed.append("brain")
            log.append("Brain fixed")
        except Exception as e:
            log.append(f"Brain fix failed: {e}")
    
    if tts_engine is None:
        try:
            tts_engine = get_tts_engine()
            fixed.append("tts")
            log.append("TTS fixed")
        except Exception as e:
            log.append(f"TTS fix failed: {e}")
    
    return {"fixed": fixed, "log": log, "status": "fixed" if fixed else "no_fix_needed"}

@app.post("/speak")
async def speak_endpoint(req: SpeakRequest):
    """TTS ישיר"""
    if not tts_engine:
        raise HTTPException(500, "TTS not available")
    
    path = await tts_engine.synthesize(req.text, play=req.play)
    
    # Broadcast to frontend
    await manager.broadcast({
        "type": "tts",
        "text": req.text,
        "audio_path": path
    })
    
    return {"text": req.text, "audio": path}

@app.post("/wake")
async def manual_wake():
    """זימון ידני של wake word מה-UI"""
    if wake_detector:
        wake_detector.simulate_wake("אדיאל ג'וניור (manual)")
    else:
        await handle_wake_word("manual wake")
    return {"waked": True}

@app.post("/chat")
async def chat_endpoint(req: TextInputRequest):
    """צ'אט טקסטואלי"""
    result = await process_user_input(req.text, with_screen=req.with_screen)
    return result

@app.get("/screen")
async def screen_endpoint():
    """צילום מסך + context"""
    if not vision_engine:
        raise HTTPException(500, "Vision not available")
    ctx = vision_engine.analyze_screen_context(include_ocr=True)
    b64 = vision_engine.get_base64_for_api(max_size=800)
    return {"context": ctx, "image_base64": b64[:100] + "..." if b64 and len(b64)>100 else b64}

# Core logic
async def handle_wake_word(detected_text: str):
    """נקרא כשמזוהה wake word"""
    app_state.activate_conversation(duration=30)
    app_state.last_wake_time = time.time()

    print(f"[Main] WAKE handled, activating conversation for 30s")

    # Notify frontend
    await manager.broadcast({
        "type": "wake_detected",
        "text": detected_text,
        "timestamp": datetime.now().isoformat()
    })

    # TTS - acknowledgment
    if tts_engine:
        ack_texts = [
            "כן בוס?",
            "כן בוס? אני כאן.",
            "אני מקשיבה, בוס.",
            "יאללה, מה צריך?"
        ]
        import random
        ack = random.choice(ack_texts)
        
        # synthesized path but also send text
        await manager.broadcast({
            "type": "listening",
            "state": True,
            "message": ack
        })

        try:
            await tts_engine.synthesize(ack, play=True)
            # שלח גם כ-audio message
            await manager.broadcast({
                "type": "assistant_speaking",
                "text": ack
            })
        except Exception as e:
            print(f"[Main] TTS ack failed: {e}")

    # Start STT listening loop - האזן לפקודה אחת
    asyncio.create_task(listen_for_command())

async def listen_for_command():
    """האזן לפקודה אחת לאחר wake"""
    if not stt_engine or not brain:
        print("[Main] No STT/Brain, cannot listen for command")
        return

    # סמן שהתחלנו להאזין
    await manager.broadcast({
        "type": "stt_state",
        "state": "recording",
        "message": "מקשיבה..."
    })

    try:
        # הקלטה
        loop = asyncio.get_event_loop()
        user_text = await loop.run_in_executor(None, lambda: stt_engine.listen_and_transcribe(max_seconds=8))
        
        if not user_text or len(user_text.strip()) < 2:
            print("[Main] No speech detected after wake")
            await manager.broadcast({
                "type": "stt_result",
                "text": "",
                "empty": True,
                "message": "לא שמעתי, תנסה שוב?"
            })
            return

        print(f"[Main] User said after wake: {user_text}")

        await manager.broadcast({
            "type": "stt_result",
            "text": user_text,
            "empty": False
        })

        # עבד עם המוח
        await process_user_input(user_text, with_screen=True)

    except Exception as e:
        print(f"[Main] listen_for_command error: {e}")
        import traceback
        traceback.print_exc()
        await manager.broadcast({
            "type": "error",
            "message": str(e)
        })

async def process_user_input(user_text: str, with_screen=True) -> Dict:
    """עיבוד קלט משתמש - לב המוח"""
    global brain, vision_engine, system_tools, tts_engine

    if not brain:
        return {"error": "Brain not initialized"}

    screen_ctx = None
    screen_b64 = None
    if with_screen and vision_engine:
        try:
            screen_ctx = vision_engine.analyze_screen_context(include_ocr=True)
            # קבל גם base64 קטן ל-frontend preview
            screen_b64 = vision_engine.get_base64_for_api(max_size=600)
        except Exception as e:
            print(f"[Main] Screen context failed: {e}")
            screen_ctx = None

    # עבד עם המוח הפרטי
    try:
        result = await brain.process(user_text, screen_context=screen_ctx)
    except Exception as e:
        print(f"[Main] Brain process failed: {e}")
        import traceback; traceback.print_exc()
        result = {
            "text": "אופס, הייתה לי תקלה בעיבוד. תנסה שוב, בוס?",
            "intent": "error",
            "hud_command": None,
            "system_action": None
        }

    response_text = result.get("text", "...")
    hud_cmd = result.get("hud_command")
    sys_action = result.get("system_action")

    print(f"[Main] Brain response: {response_text}")
    print(f"[Main] HUD cmd: {hud_cmd} | Sys action: {sys_action}")

    # בצע system action אם יש
    if sys_action and system_tools:
        try:
            sys_result = system_tools.execute_action(sys_action)
            print(f"[Main] System action result: {sys_result}")
            result["system_result"] = sys_result
        except Exception as e:
            print(f"[Main] System action failed: {e}")

    # שלח ל-frontend
    await manager.broadcast({
        "type": "brain_response",
        "user_text": user_text,
        "assistant_text": response_text,
        "intent": result.get("intent"),
        "hud_command": hud_cmd,
        "system_action": sys_action,
        "screen_context": screen_ctx,
        "screen_image": screen_b64,
        "timestamp": datetime.now().isoformat()
    })

    # אם יש HUD command, שלח בנפרד (למקרה שהקליינט מאזין לסוג מסוים)
    if hud_cmd:
        await manager.broadcast({
            "type": "hud_command",
            "command": hud_cmd
        })

    # TTS
    if tts_engine and response_text:
        try:
            # נקה טקסט ל-TTS (הסר קוד)
            await tts_engine.synthesize(response_text, play=True)
            
            await manager.broadcast({
                "type": "assistant_speaking",
                "text": response_text
            })
        except Exception as e:
            print(f"[Main] TTS failed: {e}")

    # אם זה סוף שיחה, אחרי תשובה החזר ל-idle אבל השאר conversation active לעוד קצת
    app_state.activate_conversation(duration=15)

    return {
        "user_text": user_text,
        "assistant_text": response_text,
        "intent": result.get("intent"),
        "screen_context": screen_ctx
    }

# WebSocket
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    # שלח סטטוס ראשוני
    try:
        await manager.send_personal({
            "type": "connected",
            "message": "אדיאל ג'וניור Backend מחובר - מוכנה",
            "status": {
                "brain": brain is not None,
                "vision": vision_engine is not None
            }
        }, websocket)

        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
            except:
                msg = {"type": "text", "text": data}

            msg_type = msg.get("type", "text")

            if msg_type == "text":
                text = msg.get("text", "")
                with_screen = msg.get("with_screen", True)
                if text.strip():
                    await process_user_input(text, with_screen=with_screen)

            elif msg_type == "manual_wake":
                if wake_detector:
                    wake_detector.simulate_wake("manual from WS")
                else:
                    await handle_wake_word("manual WS")

            elif msg_type == "speak":
                text = msg.get("text", "")
                if tts_engine and text:
                    await tts_engine.synthesize(text, play=True)

            elif msg_type == "get_screen":
                if vision_engine:
                    ctx = vision_engine.analyze_screen_context()
                    b64 = vision_engine.get_base64_for_api()
                    await manager.send_personal({
                        "type": "screen_data",
                        "context": ctx,
                        "image": b64
                    }, websocket)

            elif msg_type == "ping":
                await manager.send_personal({"type": "pong", "time": datetime.now().isoformat()}, websocket)

            else:
                print(f"[WS] Unknown msg type: {msg_type}")

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"[WS] Error: {e}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    # יצירת תיקיות data
    os.makedirs(os.path.join(os.path.dirname(__file__), "data", "screenshots"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), "data", "whisper_models"), exist_ok=True)

    port = int(os.getenv("PORT", "8765"))
    print(f"Starting Adiel Junior Backend on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False, log_level="info")
