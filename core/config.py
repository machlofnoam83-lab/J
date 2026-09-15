"""JARVIS core configuration.

Single source of truth for every subsystem. Zero external services: nothing in
this file may contain an API key, a cloud endpoint, or a remote URL.

Values can be overridden by environment variables prefixed with ``JARVIS_``
(e.g. ``JARVIS_SERVER_PORT=9999``) or by a local ``jarvis.local.json``.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
MODELS = ROOT / "models"
CORPUS = ROOT / "corpus"
VOICEBANK = ROOT / "brain" / "voicebank"
DATA = ROOT / "data"
LOGS = ROOT / "logs"

for _d in (MODELS, CORPUS / "generated", VOICEBANK, DATA, LOGS):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class BrainConfig:
    """Neural core settings."""

    size: str = "nano"                 # nano | micro | core
    device: str = "cpu"                # cpu | cuda (auto-detected)
    backend: str = "auto"              # auto | torch | onnx | numpy
    max_tokens: int = 512
    temperature: float = 0.75
    top_p: float = 0.92
    top_k: int = 40
    repetition_penalty: float = 1.12
    seed: int = 1337
    checkpoint: str = "jarvis_nano.pt"


@dataclass
class ReasoningConfig:
    """Symbolic kernel / reasoning loop settings."""

    max_steps: int = 8                 # hard budget for think->act->observe loops
    verify_answers: bool = True        # run hallucination guard before speaking
    prefer_tools: bool = True          # route to deterministic tools when possible
    reflect_on_failure: bool = True
    step_timeout: float = 15.0


@dataclass
class VoiceConfig:
    """Voice engine settings."""

    language: str = "he"
    engine: str = "auto"               # auto | concat | formant
    sample_rate: int = 24000
    rate: float = 1.14                 # speech rate multiplier (crisp JARVIS cadence)
    pitch: float = 1.0                 # pitch multiplier
    volume: float = 1.0
    voicebank_dir: str = str(VOICEBANK)
    use_real_voicebank: bool = True
    wake_word: str = "ג'רוויס"
    wake_word_enabled: bool = True
    barge_in: bool = True              # allow interrupting speech
    vad_enabled: bool = True


@dataclass
class MemoryConfig:
    """Memory Palace settings."""

    db_path: str = str(DATA / "memory.sqlite3")
    embedding_dim: int = 256
    recall_top_k: int = 6
    decay_half_life_days: float = 21.0
    max_episodes: int = 50_000
    auto_summarize: bool = True
    # the visible conversation, persisted so a restart does not wipe the chat
    transcript_path: str = str(DATA / "transcript.jsonl")


@dataclass
class SecurityConfig:
    """Permission Firewall. Safety is a default, not an add-on."""

    level: str = "write"               # safe | write | critical
    require_confirmation: bool = True  # ask user in HUD before CRITICAL actions
    dry_run: bool = False              # if True: report what *would* happen
    confirm_timeout: float = 180.0     # seconds the HUD has to answer a CRITICAL prompt
    audit_log: str = str(LOGS / "audit.jsonl")
    kill_switch_key: str = "F12"
    shell_enabled: bool = True
    shell_allowlist: tuple = (
        "dir", "ls", "cd", "type", "cat", "echo", "where", "which",
        "ipconfig", "ifconfig", "systeminfo", "tasklist", "ps",
        "python", "python3", "node", "npm", "git",
    )
    shell_blocklist: tuple = (
        "format", "del /f /s /q c:\\", "rm -rf /", "shutdown", "reg delete",
        "diskpart", "mkfs", ":(){:|:&};:", "cipher /w", "bcdedit",
    )
    protected_paths: tuple = (
        "C:\\Windows\\System32", "/etc", "/boot", "/sys", "~/.ssh",
    )


@dataclass
class UIConfig:
    """HUD settings."""

    theme: str = "mark1"               # mark1 | mark7 | nightops | crimson
    fullscreen: bool = True
    frameless: bool = True
    transparent: bool = True
    always_on_top: bool = False
    boot_sequence: bool = True
    particles: int = 9000
    show_telemetry: bool = True
    show_neural_web: bool = True
    mini_orb: bool = False


@dataclass
class ServerConfig:
    """Local brain<->shell bridge."""

    host: str = "127.0.0.1"            # NEVER 0.0.0.0 for the production app
    port: int = 8756
    ws_path: str = "/ws"
    sandbox_preview_host: str = "0.0.0.0"   # used only by the dev preview harness


@dataclass
class RagConfig:
    """Retrieval over the user's own local files (offline, no embeddings API).

    ``roots`` is empty by default on purpose: JARVIS must never start crawling
    the disk uninvited. The user points it at folders explicitly, from the HUD or
    with ``tools/build_rag_index.py --root``.
    """

    enabled: bool = True
    db_path: str = str(DATA / "rag_index.sqlite3")
    roots: tuple = ()                     # absolute paths; empty = nothing indexed
    include: tuple = ()                   # glob allowlist (empty = every text file)
    exclude: tuple = ("*.min.js", "*.map", "*.lock", "package-lock.json",
                      "yarn.lock", "*.svg", "*.ipynb")
    max_file_bytes: int = 4 * 1024 * 1024
    max_files: int = 20_000
    chunk_target: int = 900
    chunk_max: int = 1600
    chunk_overlap: int = 160
    top_k: int = 8
    rerank_candidates: int = 120
    ngram_dim: int = 4096
    min_hit_score: float = 0.22
    allow_self_feedback: bool = False     # index JARVIS's own data/ and logs/?
    auto_index_on_boot: bool = False      # never crawl without being asked


@dataclass
class AgentConfig:
    orchestrator: str = "jarvis"
    agents: tuple = ("jarvis", "hephaestus", "mnemosyne", "argus", "hermes")
    coder_sandbox: str = str(DATA / "sandbox")
    coder_max_repairs: int = 3
    coder_timeout: float = 20.0
    parallel_agents: int = 2


@dataclass
class Config:
    brain: BrainConfig = field(default_factory=BrainConfig)
    reasoning: ReasonConfig = field(default_factory=ReasoningConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    rag: RagConfig = field(default_factory=RagConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    persona: Dict[str, Any] = field(default_factory=lambda: {
        "name": "JARVIS",
        "full_name": "Just A Rather Very Intelligent System",
        "owner_address": "אדוני",
        "language": "he",
        "tone": "מדויק, רגוע, יעיל, עם הומור יבש ועדין",
        "principles": (
            "לעולם לא מפברק עובדות — אם אין תשובה ודאית, אומר זאת בכנות. "
            "מחשב רק דרך מנוע המתמטיקה. מבצע פעולות רק דרך שכבת ההרשאות. "
            "מדבר בעברית ברורה ותמציתית."
        ),
    })

    # ------------------------------------------------------------------ I/O --
    @staticmethod
    def load(path: Path | None = None) -> "Config":
        cfg = Config()
        cfg._apply_env()
        p = path or (ROOT / "jarvis.local.json")
        if Path(p).exists():
            cfg._merge(json.loads(Path(p).read_text(encoding="utf-8")))
        return cfg

    def save(self, path: Path | None = None) -> Path:
        p = Path(path or (ROOT / "jarvis.local.json"))
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # tuples become lists through asdict; keep them lists for JSON safety
        return d

    def _merge(self, data: Dict[str, Any]) -> None:
        for section, values in data.items():
            obj = getattr(self, section, None)
            if obj is None or not isinstance(values, dict):
                continue
            for k, v in values.items():
                if hasattr(obj, k):
                    setattr(obj, k, type(getattr(obj, k))(v) if not isinstance(v, (dict, list)) else v)

    def _apply_env(self) -> None:
        env = {k[len("JARVIS_"):].lower(): v for k, v in os.environ.items() if k.startswith("JARVIS_")}
        flat = {
            "size": (self.brain, "size"), "device": (self.brain, "device"),
            "backend": (self.brain, "backend"), "theme": (self.ui, "theme"),
            "port": (self.server, "port"), "host": (self.server, "host"),
            "level": (self.security, "level"), "dry_run": (self.security, "dry_run"),
            "engine": (self.voice, "engine"), "language": (self.voice, "language"),
        }
        for key, (obj, attr) in flat.items():
            if key in env:
                cur = getattr(obj, attr)
                if isinstance(cur, bool):
                    setattr(obj, attr, env[key].lower() in ("1", "true", "yes"))
                elif isinstance(cur, int):
                    setattr(obj, attr, int(env[key]))
                else:
                    setattr(obj, attr, env[key])


CONFIG = Config.load()
