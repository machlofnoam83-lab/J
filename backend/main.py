"""
Adiel Junior - Backend Server
FastAPI + WebSockets + Audio + Vision + Brain
השרת המרכזי שמריץ את כל המערכת

Run: python main.py
Or: uvicorn main:app --host 0.0.0.0 --port 8765 --reload
"""
import os
import sys

# בדיקת תלויות קריטיות עם הודעה בעברית אם חסר
try:
    import fastapi
except ModuleNotFoundError:
    print("\n" + "="*60)
    print("  ❌ שגיאה: fastapi לא מותקן!")
    print("  אתה מריץ את main.py בלי venv או בלי התקנה")
    print("="*60)
    print("\nפתרונות:")
    print("1. הכי קל: דאבל קליק על scripts\\fix_fastapi.bat")
    print("2. או: scripts\\run_with_autofix.bat")
    print("3. או ידנית:")
    print("   venv\\Scripts\\activate")
    print("   pip install -r backend\\requirements.txt")
    print("   python backend\\main.py")
    print("\nאם אין venv:")
    print("   python -m venv venv")
    print("   venv\\Scripts\\activate")
    print("   pip install fastapi uvicorn")
    print("="*60 + "\n")
    sys.exit(1)

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

# Fast STT - חדש! לדיבור מהיר
try:
    from audio.fast_stt import HebrewFastSTT, get_fast_stt
    print("[Main] ⚡ Fast STT loaded - מבין דיבור מהיר!")
    HAS_FAST_STT = True
except Exception as e:
    print(f"[Main] Fast STT not available: {e}")
    HAS_FAST_STT = False

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

# Advanced Reader - חדש! קריאת טקסט חכמה
try:
    from vision.advanced_reader import get_advanced_reader
    print("[Main] 📖 Advanced Reader loaded - קורא טקסט חכם RTL")
    HAS_ADV_READER = True
except Exception as e:
    print(f"[Main] Advanced Reader not available: {e}")
    HAS_ADV_READER = False
    get_advanced_reader = lambda: None

try:
    from tools.system_tools import get_system_tools
except Exception as e:
    print(f"[Main] System tools import failed: {e}")
    get_system_tools = lambda: None

try:
    from audio.device_manager import get_device_manager
    print("[Main] 🎤 Device Manager loaded - Mic selection available")
except Exception as e:
    print(f"[Main] Device manager not available: {e}")
    get_device_manager = lambda: None

try:
    from core.hebrew_dictionary import get_hebrew_dictionary
    print("[Main] 📚 Hebrew Dictionary loaded - מילון עברי מלא!")
    HAS_HEBREW_DICT = True
except Exception as e:
    print(f"[Main] Hebrew Dictionary not available: {e}")
    HAS_HEBREW_DICT = False
    get_hebrew_dictionary = lambda: None

try:
    from tools.task_orchestrator import get_task_orchestrator
    print("[Main] 🚀 Task Orchestrator loaded - Super Agent Ready")
except Exception as e:
    print(f"[Main] Task Orchestrator not available: {e}")
    get_task_orchestrator = lambda: None

# AI Voice Generator - חדש! מייצר קול לכל שאלה
try:
    from audio.ai_voice import get_ai_voice
    print("[Main] 🔊 AI Voice Generator loaded - קול AI לכל שאלה!")
    HAS_AI_VOICE = True
except Exception as e:
    print(f"[Main] AI Voice not available: {e}")
    HAS_AI_VOICE = False
    get_ai_voice = lambda: None

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
        # נסה Fast STT קודם - לדיבור מהיר
        if HAS_FAST_STT:
            try:
                from audio.fast_stt import HebrewFastSTT
                stt_engine = HebrewFastSTT(model_size=os.getenv("WHISPER_MODEL", "small"))
                print(f"[Startup] Fast STT OK (fast speech) - {stt_engine.model_size}")
            except Exception as e:
                print(f"[Startup] Fast STT failed, falling back to regular: {e}")
                stt_engine = HebrewSTT(model_size=os.getenv("WHISPER_MODEL", "small"))
                print("[Startup] STT OK (regular)")
        else:
            stt_engine = HebrewSTT(model_size=os.getenv("WHISPER_MODEL", "small"))
            print("[Startup] STT OK")
    except Exception as e:
        print(f"[Startup] STT failed: {e}")
        import traceback; traceback.print_exc()

    try:
        vision_engine = get_vision_engine()
        print("[Startup] Vision (screen capture) OK")
        if HAS_ADV_READER:
            adv_reader = get_advanced_reader()
            print(f"[Startup] Advanced Reader OK - EasyOCR:{adv_reader.has_easyocr} Tesseract:{adv_reader.has_tesseract}")
    except Exception as e:
        print(f"[Startup] Vision failed: {e}")
        import traceback; traceback.print_exc()

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

# === NEW: Learning & Self-Update Endpoints ===

@app.get("/proposals")
async def get_proposals():
    """כל ההצעות שממתינות לאישור - למידה + עדכון עצמי"""
    if not brain:
        raise HTTPException(500, "Brain not initialized")
    
    proposals = []
    self_updates = []
    
    if hasattr(brain, 'learning_engine') and brain.learning_engine:
        proposals = brain.learning_engine.get_pending_proposals()
    
    if hasattr(brain, 'self_update_manager') and brain.self_update_manager:
        self_updates = brain.self_update_manager.get_pending_updates()
    
    return {
        "learning_proposals": proposals,
        "self_update_proposals": self_updates,
        "total_pending": len(proposals) + len(self_updates)
    }

@app.post("/proposals/{proposal_id}/approve")
async def approve_proposal(proposal_id: str, extra: Dict = None):
    """מאשר הצעת למידה - המילה/עובדה הופכת לסופית"""
    if not brain or not hasattr(brain, 'learning_engine') or not brain.learning_engine:
        raise HTTPException(500, "Learning engine not available")
    
    extra = extra or {}
    result = brain.learning_engine.approve_proposal(proposal_id, extra_data=extra)
    
    if result["success"]:
        # Broadcast לקליינטים
        await manager.broadcast({
            "type": "proposal_approved",
            "proposal_id": proposal_id,
            "message": result["message"]
        })
        return result
    else:
        raise HTTPException(404, result["error"])

@app.post("/proposals/{proposal_id}/reject")
async def reject_proposal(proposal_id: str):
    """דוחה הצעת למידה"""
    if not brain or not hasattr(brain, 'learning_engine') or not brain.learning_engine:
        raise HTTPException(500, "Learning engine not available")
    
    result = brain.learning_engine.reject_proposal(proposal_id)
    
    if result["success"]:
        await manager.broadcast({
            "type": "proposal_rejected",
            "proposal_id": proposal_id,
            "message": result["message"]
        })
        return result
    else:
        raise HTTPException(404, result["error"])

class SelfUpdateApproveRequest(BaseModel):
    reason: str = ""

@app.post("/self-updates/{update_id}/approve")
async def approve_self_update(update_id: str):
    """מאשר עדכון עצמי של אדיאל - היא משתפרת אבל רק באישורך"""
    if not brain or not hasattr(brain, 'self_update_manager') or not brain.self_update_manager:
        raise HTTPException(500, "Self-update manager not available")
    
    result = brain.self_update_manager.approve_update(update_id)
    
    if result["success"]:
        await manager.broadcast({
            "type": "self_update_approved",
            "update_id": update_id,
            "message": result["message"],
            "update": result.get("update")
        })
        return result
    else:
        raise HTTPException(404, result["error"])

@app.post("/self-updates/{update_id}/reject")
async def reject_self_update(update_id: str, req: SelfUpdateApproveRequest = None):
    """דוחה עדכון עצמי"""
    if not brain or not hasattr(brain, 'self_update_manager') or not brain.self_update_manager:
        raise HTTPException(500, "Self-update manager not available")
    
    reason = req.reason if req else ""
    result = brain.self_update_manager.reject_update(update_id, reason=reason)
    
    if result["success"]:
        await manager.broadcast({
            "type": "self_update_rejected",
            "update_id": update_id,
            "message": result["message"]
        })
        return result
    else:
        raise HTTPException(404, result["error"])

@app.get("/profile")
async def get_profile():
    """פרופיל המשתמש שאדיאל זוכרת - הזיכרון החכם"""
    if not brain or not hasattr(brain, 'learning_engine') or not brain.learning_engine:
        raise HTTPException(500, "Learning engine not available")
    
    profile = brain.learning_engine.get_user_profile_summary()
    smart_context = brain.learning_engine.get_smart_context()
    
    return {
        "profile": profile,
        "smart_context": smart_context,
        "vocabulary": {
            "count": len(brain.learning_engine.vocab.vocab.get("words", {})),
            "words": list(brain.learning_engine.vocab.vocab.get("words", {}).keys())[-20:]
        },
        "memory": brain.memory.get_context_string()[:500] if hasattr(brain, 'memory') else ""
    }

# === Audio Devices - בחירת מיקרופון ===

@app.get("/audio/devices")
async def list_audio_devices():
    """רשימת כל המיקרופונים והרמקולים - לבחירה"""
    try:
        dev_manager = get_device_manager()
        if not dev_manager:
            raise HTTPException(500, "Device manager not available")
        
        devices = dev_manager.list_devices()
        input_devices = dev_manager.list_input_devices()
        output_devices = dev_manager.list_output_devices()
        
        return {
            "all": devices,
            "inputs": input_devices,
            "outputs": output_devices,
            "selected_input": dev_manager.get_selected_input(),
            "selected_output": dev_manager.get_selected_output(),
            "total": len(devices)
        }
    except Exception as e:
        print(f"[Audio Devices] Error: {e}")
        raise HTTPException(500, str(e))

@app.post("/audio/devices/input/{device_id}")
async def select_input_device(device_id: int):
    """בוחר מיקרופון - אדיאל תשתמש בו מהיום"""
    try:
        dev_manager = get_device_manager()
        if not dev_manager:
            raise HTTPException(500, "Device manager not available")
        
        result = dev_manager.select_input_device(device_id)
        
        # אם הצליח, עדכן את המנועים הקיימים
        global wake_detector, stt_engine
        if result["success"]:
            # צריך להפעיל מחדש את wake detector עם המיקרופון החדש
            if wake_detector:
                try:
                    wake_detector.stop()
                    def on_wake(text):
                        asyncio.run_coroutine_threadsafe(handle_wake_word(text), asyncio.get_event_loop())
                    
                    from audio.wake_word import WakeWordDetector
                    wake_detector = WakeWordDetector(on_wake=on_wake, device_id=device_id)
                    wake_detector.start()
                    result["restarted_wake"] = True
                except Exception as e:
                    result["restart_warning"] = f"צריך להפעיל מחדש את Backend כדי להשתמש במיקרופון החדש: {e}"
            
            if stt_engine:
                stt_engine.device_id = device_id
        
        if result["success"]:
            await manager_broadcast_safe({
                "type": "mic_changed",
                "device_id": device_id,
                "device_name": result.get("device", {}).get("name", ""),
                "message": result.get("message", "")
            })
        
        return result
    except Exception as e:
        print(f"[Select Input] Error: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(500, str(e))

@app.post("/audio/devices/output/{device_id}")
async def select_output_device(device_id: int):
    """בוחר רמקול"""
    try:
        dev_manager = get_device_manager()
        if not dev_manager:
            raise HTTPException(500, "Device manager not available")
        
        result = dev_manager.select_output_device(device_id)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/audio/devices/reset")
async def reset_audio_devices():
    """חוזר לברירת מחדל"""
    try:
        dev_manager = get_device_manager()
        if not dev_manager:
            raise HTTPException(500, "Device manager not available")
        
        result = dev_manager.reset_to_default()
        return result
    except Exception as e:
        raise HTTPException(500, str(e))

# === מילון עברי מלא ===
@app.get("/dictionary/lookup")
async def dictionary_lookup(word: str):
    try:
        if not HAS_HEBREW_DICT:
            raise HTTPException(500, "Dictionary not available")
        dict_manager = get_hebrew_dictionary()
        result = dict_manager.lookup(word)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/dictionary/search")
async def dictionary_search(query: str):
    try:
        if not HAS_HEBREW_DICT:
            raise HTTPException(500, "Dictionary not available")
        dict_manager = get_hebrew_dictionary()
        results = dict_manager.search_by_meaning(query)
        return {"query": query, "results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/dictionary/stats")
async def dictionary_stats():
    try:
        if not HAS_HEBREW_DICT:
            raise HTTPException(500, "Dictionary not available")
        return get_hebrew_dictionary().get_stats()
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/dictionary/random")
async def dictionary_random(count: int = 5):
    try:
        if not HAS_HEBREW_DICT:
            raise HTTPException(500, "Dictionary not available")
        words = get_hebrew_dictionary().get_random_words(count)
        return {"words": words, "count": len(words)}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/dictionary/add")
async def dictionary_add(request: Dict):
    try:
        if not HAS_HEBREW_DICT:
            raise HTTPException(500, "Dictionary not available")
        word = request.get("word", "").strip()
        meaning = request.get("meaning", "").strip()
        if not word or not meaning:
            raise HTTPException(400, "word and meaning required")
        get_hebrew_dictionary().add_word(word, meaning, request.get("type", "custom"), request.get("example", ""))
        return {"success": True, "word": word, "meaning": meaning}
    except Exception as e:
        raise HTTPException(500, str(e))

async def manager_broadcast_safe(msg: dict):
    """עוזר ל-broadcast בטוח"""
    try:
        await manager.broadcast(msg)
    except:
        pass

# === SUPER AGENT - ניהול משימות וגלישה, פרודוקטיביות, שליטה במחשב ===

class TaskRequest(BaseModel):
    text: str
    context: Optional[Dict] = None

@app.post("/tasks/execute")
async def execute_task(req: TaskRequest):
    """ביצוע משימה חכמה - מנתב לסוכן הנכון"""
    try:
        from tools.task_orchestrator import get_task_orchestrator
        orchestrator = get_task_orchestrator()
        result = await orchestrator.execute_task(req.text, context=req.context or {})
        
        # Broadcast progress
        await manager_broadcast_safe({
            "type": "task_result",
            "task_type": result.get("task_type"),
            "success": result.get("success"),
            "message": result.get("message")
        })
        
        return result
    except Exception as e:
        print(f"[Tasks] Execute failed: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(500, str(e))

@app.get("/tasks/history")
async def get_task_history():
    try:
        from tools.task_orchestrator import get_task_orchestrator
        orchestrator = get_task_orchestrator()
        return {"tasks": orchestrator.get_task_history()}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/tasks/shopping/search")
async def shopping_search(query: str, max_price: float = None):
    try:
        from tools.shopping_agent import get_shopping_agent
        agent = get_shopping_agent()
        products = await agent.search_product(query)
        cheapest = agent.find_cheapest(products)
        return {
            "query": query,
            "products": [{"name": p.name, "price": p.price, "store": p.store, "rating": p.rating, "url": p.url} for p in products],
            "cheapest": {"name": cheapest.name, "price": cheapest.price, "store": cheapest.store} if cheapest else None,
            "count": len(products)
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/tasks/travel/search")
async def travel_search(from_city: str, to_city: str, check_in: str, check_out: str, budget: float = None):
    try:
        from tools.travel_agent import get_travel_agent
        agent = get_travel_agent()
        result = await agent.find_best_deal(from_city, to_city, check_in, check_out, budget)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/tasks/research")
async def research_topic(topic: str, depth: int = 5):
    try:
        from tools.research_agent import get_research_agent
        agent = get_research_agent()
        result = await agent.research_topic(topic, depth=depth)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/tasks/email/inbox")
async def get_emails(limit: int = 20, urgent_only: bool = False):
    try:
        from tools.email_manager import get_email_manager
        mgr = get_email_manager()
        if urgent_only:
            emails = await mgr.get_urgent_emails()
        else:
            emails = await mgr.fetch_emails(limit=limit)
        return {
            "emails": [{"id": e.id, "from": e.from_addr, "subject": e.subject, "urgency": e.urgency, "category": e.category, "is_spam": e.is_spam} for e in emails],
            "count": len(emails)
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/tasks/calendar/slots")
async def get_calendar_slots(duration: int = 60, days: int = 5):
    try:
        from tools.calendar_manager import get_calendar_manager
        mgr = get_calendar_manager()
        slots = await mgr.find_free_slots(duration_minutes=duration, days_ahead=days)
        return {
            "slots": [{"start": s.start.isoformat(), "end": s.end.isoformat(), "score": s.score, "reason": s.reason} for s in slots],
            "count": len(slots)
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/tasks/files/scan")
async def scan_files(directory: str = "~/Downloads", pattern: str = "*"):
    try:
        from tools.cloud_file_manager import get_file_manager
        mgr = get_file_manager()
        files = mgr.scan_files(directory, pattern)
        return {"files": files[:50], "count": len(files), "directory": directory}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/tasks/files/organize")
async def organize_files(source_dir: str = "~/Downloads"):
    try:
        from tools.cloud_file_manager import get_file_manager
        mgr = get_file_manager()
        result = mgr.organize_files(source_dir)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/tasks/routines")
async def list_routines():
    try:
        from tools.routine_automation import get_routine_automation
        routine_mgr = get_routine_automation()
        return {"routines": routine_mgr.list_routines()}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/tasks/routines/{routine_id}/run")
async def run_routine(routine_id: str):
    try:
        from tools.routine_automation import get_routine_automation
        routine_mgr = get_routine_automation()
        result = await routine_mgr.run_routine(routine_id)
        await manager_broadcast_safe({
            "type": "routine_result",
            "routine_id": routine_id,
            "success": result.get("success"),
            "message": result.get("message")
        })
        return result
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/tasks/computer/open")
async def open_app_endpoint(app_name: str):
    try:
        from tools.computer_control import get_computer_control
        ctrl = get_computer_control()
        result = ctrl.open_app(app_name)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/tasks/computer/play")
async def play_media(service: str, query: str = "", action: str = "play"):
    try:
        from tools.computer_control import get_computer_control
        ctrl = get_computer_control()
        if service.lower() == "spotify":
            return ctrl.play_spotify(query=query, action=action)
        elif service.lower() == "youtube":
            return ctrl.play_youtube(query=query)
        else:
            return {"success": False, "error": f"Unknown service {service}"}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/speak")
async def speak_endpoint(req: SpeakRequest):
    """TTS ישיר עם base64"""
    if not tts_engine:
        raise HTTPException(500, "TTS not available")
    
    result = await tts_engine.synthesize(req.text, play=req.play)
    path = None
    b64 = None
    if isinstance(result, tuple):
        path, b64 = result
    elif isinstance(result, str):
        path = result
    
    await manager.broadcast({
        "type": "tts",
        "text": req.text,
        "audio_path": path,
        "audio_base64": b64
    })
    
    return {"text": req.text, "audio": path, "audio_base64": b64[:100]+"..." if b64 and len(b64)>100 else b64}

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
    """צילום מסך + context עם קריאת טקסט חכמה"""
    if not vision_engine:
        raise HTTPException(500, "Vision not available")
    
    # נסה קריאה מתקדמת קודם
    if HAS_ADV_READER:
        try:
            adv_reader = get_advanced_reader()
            result = adv_reader.read_screen()
            if result["success"]:
                b64 = vision_engine.get_base64_for_api(max_size=800) if vision_engine else None
                return {
                    "context": result["full_text"],
                    "advanced": result,
                    "understanding": adv_reader.understand_text(result["full_text"]),
                    "image_base64": b64[:100] + "..." if b64 and len(b64)>100 else b64,
                    "method": "advanced_reader"
                }
        except Exception as e:
            print(f"[Screen] Advanced reader failed, fallback: {e}")
    
    # Fallback ישן
    ctx = vision_engine.analyze_screen_context(include_ocr=True)
    b64 = vision_engine.get_base64_for_api(max_size=800)
    return {"context": ctx, "image_base64": b64[:100] + "..." if b64 and len(b64)>100 else b64, "method": "basic"}

@app.post("/vision/read-text")
async def read_text_endpoint(request: Dict):
    """קריאת טקסט מתקדמת - מקבל תמונה base64 או URL"""
    try:
        if HAS_ADV_READER:
            reader = get_advanced_reader()
            image_b64 = request.get("image_base64", "")
            question = request.get("question", "")
            
            if image_b64:
                import base64
                from PIL import Image
                import io
                
                # Decode base64
                if "," in image_b64:
                    image_b64 = image_b64.split(",")[1]
                img_data = base64.b64decode(image_b64)
                img = Image.open(io.BytesIO(img_data))
                
                blocks = reader.read_image(image=img)
                full_text = " ".join([b.text for b in blocks])
                understanding = reader.understand_text(full_text, question=question)
                
                return {
                    "success": True,
                    "full_text": full_text,
                    "blocks": len(blocks),
                    "understanding": understanding,
                    "message": f"קראתי {len(blocks)} בלוקים, {len(full_text)} תווים"
                }
        
        return {"success": False, "error": "Advanced reader not available"}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/audio/transcribe-fast")
async def transcribe_fast_endpoint(request: Dict):
    """תמלול דיבור מהיר - עם preprocessing"""
    try:
        # קבל audio base64
        audio_b64 = request.get("audio_base64", "")
        if not audio_b64:
            raise HTTPException(400, "audio_base64 required")
        
        import base64
        import numpy as np
        
        if "," in audio_b64:
            audio_b64 = audio_b64.split(",")[1]
        audio_bytes = base64.b64decode(audio_b64)
        audio = np.frombuffer(audio_bytes, dtype=np.float32)
        
        if HAS_FAST_STT:
            from audio.fast_stt import get_fast_stt
            stt = get_fast_stt()
            text = stt.transcribe_fast(audio)
            return {
                "success": True,
                "text": text,
                "method": "fast_stt",
                "message": f"תמללתי דיבור מהיר: '{text}'"
            }
        else:
            # Fallback לרגיל
            if stt_engine:
                text = stt_engine.transcribe_audio(audio)
                return {"success": True, "text": text, "method": "regular"}
            else:
                raise HTTPException(500, "STT not available")
                
    except Exception as e:
        print(f"[Transcribe Fast] Error: {e}")
        raise HTTPException(500, str(e))

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
        
        await manager.broadcast({
            "type": "listening",
            "state": True,
            "message": ack
        })

        try:
            # נסה AI Voice קודם
            ack_b64 = None
            if HAS_AI_VOICE:
                try:
                    from audio.ai_voice import get_ai_voice
                    ai_voice = get_ai_voice()
                    _, ack_b64 = await ai_voice.generate_voice_for_any_question(ack, play=True)
                except:
                    pass
            
            if not ack_b64 and tts_engine:
                result = await tts_engine.synthesize(ack, play=True)
                if isinstance(result, tuple):
                    _, ack_b64 = result
            
            await manager.broadcast({
                "type": "assistant_speaking",
                "text": ack,
                "audio_base64": ack_b64,
                "has_voice": ack_b64 is not None
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
    """עיבוד קלט משתמש - לב המוח עם למידה והצעות לשיפור"""
    global brain, vision_engine, system_tools, tts_engine

    if not brain:
        return {"error": "Brain not initialized"}

    screen_ctx = None
    screen_b64 = None
    if with_screen and vision_engine:
        try:
            screen_ctx = vision_engine.analyze_screen_context(include_ocr=True)
            screen_b64 = vision_engine.get_base64_for_api(max_size=600)
        except Exception as e:
            print(f"[Main] Screen context failed: {e}")
            screen_ctx = None

    # עבד עם המוח הפרטי (עכשיו עם למידה) - תיקון באג shadowing
    try:
        brain_result = await brain.process(user_text, screen_context=screen_ctx)
    except Exception as e:
        print(f"[Main] Brain process failed: {e}")
        import traceback; traceback.print_exc()
        brain_result = {
            "text": "אופס, הייתה לי תקלה בעיבוד. תנסה שוב, בוס?",
            "intent": "error",
            "hud_command": None,
            "system_action": None,
            "proposals": [],
            "self_updates": []
        }

    response_text = brain_result.get("text", "...")
    hud_cmd = brain_result.get("hud_command")
    sys_action = brain_result.get("system_action")
    proposals = brain_result.get("proposals", [])
    self_updates = brain_result.get("self_updates", [])

    print(f"[Main] Brain response: {response_text}")
    print(f"[Main] HUD cmd: {hud_cmd} | Sys action: {sys_action}")
    if proposals:
        print(f"[Main] 📚 {len(proposals)} הצעות למידה חדשות")
    if self_updates:
        print(f"[Main] 🤖 {len(self_updates)} הצעות שיפור עצמי")

    # בצע system action אם יש
    if sys_action and system_tools:
        try:
            sys_result = system_tools.execute_action(sys_action)
            print(f"[Main] System action result: {sys_result}")
            brain_result["system_result"] = sys_result
        except Exception as e:
            print(f"[Main] System action failed: {e}")

    # שלח ל-frontend - תשובה + הצעות - תיקון באג shadowing
    await manager.broadcast({
        "type": "brain_response",
        "user_text": user_text,
        "assistant_text": response_text,
        "intent": brain_result.get("intent"),
        "hud_command": hud_cmd,
        "system_action": sys_action,
        "screen_context": screen_ctx,
        "screen_image": screen_b64,
        "proposals": proposals,
        "self_updates": self_updates,
        "user_profile": brain_result.get("user_profile"),
        "learning_active": brain_result.get("learning_active", False),
        "timestamp": datetime.now().isoformat()
    })

    # אם יש הצעות למידה, שלח אותן בנפרד כ-cards
    if proposals:
        for prop in proposals:
            await manager.broadcast({
                "type": "learning_proposal",
                "proposal": prop
            })
    
    if self_updates:
        for upd in self_updates:
            await manager.broadcast({
                "type": "self_update_proposal",
                "proposal": upd
            })

    # אם יש HUD command, שלח בנפרד
    if hud_cmd:
        await manager.broadcast({
            "type": "hud_command",
            "command": hud_cmd
        })

    # TTS - AI Voice Generator - תמיד עם קול לכל שאלה!
    tts_audio_b64 = None
    if response_text:
        try:
            # נסה AI Voice Generator קודם - קול AI אמיתי לכל שאלה
            if HAS_AI_VOICE:
                try:
                    from audio.ai_voice import get_ai_voice
                    ai_voice = get_ai_voice()
                    ai_path, ai_b64 = await ai_voice.generate_voice_for_any_question(response_text, play=True)
                    if ai_b64:
                        tts_audio_b64 = ai_b64
                        print(f"[Main] 🔊 AI Voice generated for ANY question: {response_text[:30]}...")
                    elif ai_path:
                        # אם יש קובץ אבל לא base64, צור base64
                        import base64
                        from pathlib import Path
                        if Path(ai_path).exists():
                            data = Path(ai_path).read_bytes()
                            b64 = base64.b64encode(data).decode()
                            tts_audio_b64 = f"data:audio/mpeg;base64,{b64}" if str(ai_path).endswith('.mp3') else f"data:audio/wav;base64,{b64}"
                except Exception as e:
                    print(f"[Main] AI Voice failed, fallback to TTS: {e}")
            
            # Fallback ל-TTS הרגיל אם AI Voice לא הצליח
            if not tts_audio_b64 and tts_engine:
                tts_result = await tts_engine.synthesize(response_text, play=True)
                if isinstance(tts_result, tuple):
                    _, tts_audio_b64 = tts_result
                elif isinstance(tts_result, str) and tts_result.startswith("data:audio"):
                    tts_audio_b64 = tts_result
            
            await manager.broadcast({
                "type": "assistant_speaking",
                "text": response_text,
                "audio_base64": tts_audio_b64,
                "has_voice": tts_audio_b64 is not None,
                "voice_type": "ai_original" if tts_audio_b64 else "none"
            })
        except Exception as e:
            print(f"[Main] TTS/AI Voice failed: {e}")
            import traceback; traceback.print_exc()
            # גם אם נכשל, שלח טקסט בלי קול כדי שלא יקרוס
            await manager.broadcast({
                "type": "assistant_speaking",
                "text": response_text,
                "audio_base64": None,
                "has_voice": False
            })

    app_state.activate_conversation(duration=15)

    return {
        "user_text": user_text,
        "assistant_text": response_text,
        "intent": brain_result.get("intent"),
        "screen_context": screen_ctx,
        "proposals": proposals,
        "self_updates": self_updates,
        "user_profile": brain_result.get("user_profile")
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

            elif msg_type == "approve_proposal":
                prop_id = msg.get("id", "")
                if brain and hasattr(brain, 'learning_engine') and brain.learning_engine:
                    result = brain.learning_engine.approve_proposal(prop_id, extra_data=msg.get("extra", {}))
                    await manager.broadcast({
                        "type": "proposal_approved" if result["success"] else "error",
                        "proposal_id": prop_id,
                        "message": result.get("message", "") if result["success"] else result.get("error", "")
                    })

            elif msg_type == "reject_proposal":
                prop_id = msg.get("id", "")
                if brain and hasattr(brain, 'learning_engine') and brain.learning_engine:
                    result = brain.learning_engine.reject_proposal(prop_id)
                    await manager.broadcast({
                        "type": "proposal_rejected" if result["success"] else "error",
                        "proposal_id": prop_id,
                        "message": result.get("message", "")
                    })

            elif msg_type == "approve_self_update":
                upd_id = msg.get("id", "")
                if brain and hasattr(brain, 'self_update_manager') and brain.self_update_manager:
                    result = brain.self_update_manager.approve_update(upd_id)
                    await manager.broadcast({
                        "type": "self_update_approved" if result["success"] else "error",
                        "update_id": upd_id,
                        "message": result.get("message", ""),
                        "update": result.get("update")
                    })

            elif msg_type == "reject_self_update":
                upd_id = msg.get("id", "")
                if brain and hasattr(brain, 'self_update_manager') and brain.self_update_manager:
                    result = brain.self_update_manager.reject_update(upd_id, reason=msg.get("reason", ""))
                    await manager.broadcast({
                        "type": "self_update_rejected" if result["success"] else "error",
                        "update_id": upd_id,
                        "message": result.get("message", "")
                    })

            elif msg_type == "get_profile":
                if brain and hasattr(brain, 'learning_engine') and brain.learning_engine:
                    profile = brain.learning_engine.get_user_profile_summary()
                    smart = brain.learning_engine.get_smart_context()
                    await manager.send_personal({
                        "type": "profile_data",
                        "profile": profile,
                        "smart_context": smart
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
