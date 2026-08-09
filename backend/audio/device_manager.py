"""
Audio Device Manager - בחירת מיקרופון ורמקול
מאפשר לבחור איזה מיקרופון אדיאל תשתמש
"""
import json
from pathlib import Path
from typing import List, Dict, Optional

DATA_DIR = Path(__file__).parent.parent / "data"
DEVICE_CONFIG_FILE = DATA_DIR / "audio_devices.json"

try:
    import sounddevice as sd
    HAS_SD = True
except:
    HAS_SD = False
    print("[DeviceManager] sounddevice not available")

class AudioDeviceManager:
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.config = self._load_config()
        self.devices_cache = []

    def _load_config(self) -> Dict:
        if DEVICE_CONFIG_FILE.exists():
            try:
                return json.loads(DEVICE_CONFIG_FILE.read_text(encoding='utf-8'))
            except:
                pass
        return {
            "selected_input_id": None,
            "selected_input_name": None,
            "selected_output_id": None,
            "selected_output_name": None,
            "input_volume": 1.0,
            "auto_select": True
        }

    def _save_config(self):
        try:
            DEVICE_CONFIG_FILE.write_text(json.dumps(self.config, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as e:
            print(f"[DeviceManager] Save failed: {e}")

    def list_devices(self) -> List[Dict]:
        """מחזיר רשימת כל התקני הקול עם פרטים"""
        if not HAS_SD:
            return [
                {"id": -1, "name": "No audio devices (sounddevice not installed)", "type": "none", "channels": 0}
            ]
        
        try:
            devices = sd.query_devices()
            result = []
            
            for i, dev in enumerate(devices):
                dev_type = "unknown"
                if dev['max_input_channels'] > 0 and dev['max_output_channels'] > 0:
                    dev_type = "input+output"
                elif dev['max_input_channels'] > 0:
                    dev_type = "input"
                elif dev['max_output_channels'] > 0:
                    dev_type = "output"
                
                result.append({
                    "id": i,
                    "name": dev['name'],
                    "type": dev_type,
                    "input_channels": dev['max_input_channels'],
                    "output_channels": dev['max_output_channels'],
                    "samplerate": dev['default_samplerate'],
                    "is_default_input": i == sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else False,
                    "is_default_output": i == sd.default.device[1] if isinstance(sd.default.device, (list, tuple)) else False,
                    "selected": i == self.config.get("selected_input_id") if dev_type in ["input", "input+output"] else i == self.config.get("selected_output_id")
                })
            
            self.devices_cache = result
            return result
            
        except Exception as e:
            print(f"[DeviceManager] List devices failed: {e}")
            return [{"id": -1, "name": f"Error: {e}", "type": "error", "channels": 0}]

    def list_input_devices(self) -> List[Dict]:
        """רק מיקרופונים"""
        all_devs = self.list_devices()
        return [d for d in all_devs if "input" in d["type"]]

    def list_output_devices(self) -> List[Dict]:
        """רק רמקולים"""
        all_devs = self.list_devices()
        return [d for d in all_devs if "output" in d["type"]]

    def select_input_device(self, device_id: int) -> Dict:
        """בוחר מיקרופון"""
        devices = self.list_devices()
        
        # מצא את המכשיר
        selected = None
        for dev in devices:
            if dev["id"] == device_id:
                selected = dev
                break
        
        if not selected:
            return {"success": False, "error": f"Device {device_id} not found"}
        
        if "input" not in selected["type"]:
            return {"success": False, "error": f"Device {device_id} is not an input device"}
        
        self.config["selected_input_id"] = device_id
        self.config["selected_input_name"] = selected["name"]
        self.config["auto_select"] = False
        self._save_config()
        
        print(f"[DeviceManager] Selected input: {device_id} - {selected['name']}")
        
        # נסה לבדוק אם המכשיר עובד
        try:
            if HAS_SD:
                import numpy as np
                # הקלטה קצרה לבדיקה
                duration = 0.5
                samplerate = 16000
                recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, device=device_id, dtype='float32')
                sd.wait()
                max_vol = float(np.max(np.abs(recording)))
                return {
                    "success": True,
                    "device": selected,
                    "test_volume": max_vol,
                    "message": f"נבחר מיקרופון: {selected['name']} (עוצמה בבדיקה: {max_vol:.3f})"
                }
        except Exception as e:
            print(f"[DeviceManager] Test recording failed: {e}")
            return {
                "success": True,
                "device": selected,
                "test_volume": 0,
                "message": f"נבחר מיקרופון: {selected['name']} (בדיקה נכשלה: {e})",
                "warning": str(e)
            }
        
        return {"success": True, "device": selected}

    def select_output_device(self, device_id: int) -> Dict:
        """בוחר רמקול"""
        devices = self.list_devices()
        selected = None
        for dev in devices:
            if dev["id"] == device_id:
                selected = dev
                break
        
        if not selected:
            return {"success": False, "error": f"Device {device_id} not found"}
        
        self.config["selected_output_id"] = device_id
        self.config["selected_output_name"] = selected["name"]
        self._save_config()
        
        print(f"[DeviceManager] Selected output: {device_id} - {selected['name']}")
        return {"success": True, "device": selected, "message": f"נבחר רמקול: {selected['name']}"}

    def get_selected_input(self) -> Optional[int]:
        """מחזיר את ה-ID של המיקרופון הנבחר או None ל-default"""
        return self.config.get("selected_input_id")

    def get_selected_output(self) -> Optional[int]:
        return self.config.get("selected_output_id")

    def reset_to_default(self):
        """חוזר ל-default"""
        self.config["selected_input_id"] = None
        self.config["selected_input_name"] = None
        self.config["selected_output_id"] = None
        self.config["selected_output_name"] = None
        self.config["auto_select"] = True
        self._save_config()
        return {"success": True, "message": "חזר לברירת מחדל"}

# Singleton
_global_manager = None

def get_device_manager() -> AudioDeviceManager:
    global _global_manager
    if _global_manager is None:
        _global_manager = AudioDeviceManager()
    return _global_manager
