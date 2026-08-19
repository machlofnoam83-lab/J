from __future__ import annotations

import argparse
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .chat import DEFAULT_SYSTEM, generate_reply

_state: dict = {}
_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    from .model import load_for_inference

    model_path = os.environ.get("JAI_MODEL", "outputs/jai-3b-he")
    no_4bit = os.environ.get("JAI_NO_4BIT", "0") == "1"
    _state["model"], _state["tokenizer"] = load_for_inference(model_path, not no_4bit)
    yield
    _state.clear()


app = FastAPI(title="JAI 3B", version="0.1.0", lifespan=lifespan)


class Message(BaseModel):
    role: str
    content: str = Field(min_length=1, max_length=20_000)


class ChatRequest(BaseModel):
    messages: list[Message]
    max_new_tokens: int = Field(default=384, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0, le=2)


@app.get("/health")
def health() -> dict:
    return {"status": "ready" if "model" in _state else "loading"}


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    if not request.messages:
        raise HTTPException(400, "messages cannot be empty")
    messages = [message.model_dump() for message in request.messages]
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM})
    if any(message["role"] not in {"system", "user", "assistant"} for message in messages):
        raise HTTPException(400, "invalid message role")
    with _lock:
        answer = generate_reply(
            _state["model"], _state["tokenizer"], messages,
            request.max_new_tokens, request.temperature,
        )
    return {"answer": answer}


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """<!doctype html>
<html lang="he" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>JAI 3B</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;font-family:system-ui;background:#0b1020;color:#eef2ff}
main{max-width:850px;margin:auto;height:100vh;display:flex;flex-direction:column;padding:20px}.top{display:flex;align-items:center;gap:12px;border-bottom:1px solid #29324d;padding-bottom:14px}
.logo{background:linear-gradient(135deg,#7c3aed,#2563eb);padding:10px 14px;border-radius:14px;font-weight:800}.sub{color:#9ca3af;font-size:13px}
#chat{flex:1;overflow:auto;padding:24px 0}.msg{max-width:82%;padding:12px 15px;border-radius:16px;margin:10px 0;white-space:pre-wrap;line-height:1.6}.user{background:#2563eb;margin-right:auto}.assistant{background:#1b2338;border:1px solid #303a58;margin-left:auto}
form{display:flex;gap:10px;border-top:1px solid #29324d;padding-top:14px}textarea{flex:1;resize:none;border:1px solid #34405f;border-radius:14px;background:#141b2e;color:white;padding:13px;font:inherit}button{border:0;border-radius:14px;background:#7c3aed;color:white;padding:0 24px;font-weight:700;cursor:pointer}button:disabled{opacity:.5}
</style></head><body><main><div class="top"><div class="logo">JAI</div><div><b>העוזר האישי שלך</b><div class="sub">מודל 3B מקומי • השיחות נשארות אצלך</div></div></div><div id="chat"><div class="msg assistant">שלום! אני JAI. איך אפשר לעזור?</div></div>
<form id="form"><textarea id="input" rows="2" placeholder="כתוב הודעה..." required></textarea><button id="send">שליחה</button></form></main>
<script>const history=[],chat=document.querySelector('#chat'),input=document.querySelector('#input'),send=document.querySelector('#send');function add(text,role){const e=document.createElement('div');e.className='msg '+role;e.textContent=text;chat.append(e);chat.scrollTop=chat.scrollHeight}document.querySelector('#form').onsubmit=async e=>{e.preventDefault();const text=input.value.trim();if(!text)return;add(text,'user');history.push({role:'user',content:text});input.value='';send.disabled=true;const wait=document.createElement('div');wait.className='msg assistant';wait.textContent='חושב...';chat.append(wait);try{const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages:history})});const d=await r.json();if(!r.ok)throw Error(d.detail||'שגיאה');wait.remove();add(d.answer,'assistant');history.push({role:'assistant',content:d.answer})}catch(err){wait.textContent='שגיאה: '+err.message}finally{send.disabled=false;input.focus()}};input.onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();document.querySelector('#form').requestSubmit()}}</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve JAI as a local web app and JSON API")
    parser.add_argument("--model", default="outputs/jai-3b-he")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()
    os.environ["JAI_MODEL"] = args.model
    os.environ["JAI_NO_4BIT"] = "1" if args.no_4bit else "0"
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
