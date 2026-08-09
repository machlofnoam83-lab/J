"""
HUD Tools - שליטה בממשק
"""
from typing import Dict

class HUDController:
    def __init__(self, websocket_manager=None):
        self.ws_manager = websocket_manager
        self.current_mode = "center"  # center / side / hidden / orb

    async def send_hud_command(self, command: Dict):
        """שלח פקודת HUD ל-frontend"""
        if self.ws_manager:
            await self.ws_manager.broadcast({
                "type": "hud_command",
                "command": command
            })
        
        # עדכן מצב פנימי
        action = command.get("action")
        if action == "dock":
            self.current_mode = "side"
        elif action == "center":
            self.current_mode = "center"
        elif action == "hide":
            self.current_mode = "hidden"
        elif action == "show":
            self.current_mode = "center"

        print(f"[HUD] Command: {command} -> mode {self.current_mode}")

    def get_current_mode(self):
        return self.current_mode
