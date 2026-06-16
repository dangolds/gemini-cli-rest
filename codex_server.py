"""
REST API server that wraps OpenAI's Codex CLI (codex) in headless mode.

Unlike the agy bridge (server.py), which drives a live interactive TUI through a
tmux terminal-emulator mediator, Codex ships a first-class headless mode:

    codex exec [PROMPT]                  -> run one turn, print the answer
    codex exec resume <SESSION_ID> ...   -> continue a prior session statelessly

There is therefore NO warm process to keep alive. Process boot is ~0.25s
(negligible next to the model call), and OpenAI's server-side prompt cache spans
separate `exec` invocations, so keeping codex warm buys nothing. Each named
session (/chat/{name}) is just a small bit of state — its codex session UUID
plus a per-session working directory — and every turn shells out to a fresh
`codex exec` / `codex exec resume` process.

The REST contract is identical to the agy bridge (server.py) so a future
codex-bridge skill can mirror the gemini-bridge skill one-to-one.
"""

import asyncio
import json
import logging
import os
import shlex
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path as FsPath

from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
CODEX_CMD = os.getenv("CODEX_CMD", "codex")
# Space-separated extra args appended to every codex invocation, e.g.
# "--add-dir /repos" to grant read/write access to mounted repositories.
CODEX_EXTRA_ARGS = os.getenv("CODEX_EXTRA_ARGS", "")

# Each session gets its own working directory. The per-run id keeps paths unique
# across server restarts (same trick as the agy bridge): codex records sessions
# on disk keyed partly by cwd, so a fresh path guarantees a clean slate.
RUN_ID = uuid.uuid4().hex[:8]
SESSIONS_ROOT = FsPath(os.getenv("SESSIONS_ROOT", "/tmp/codex-rest-sessions")) / RUN_ID

# Local, persisted log destination (mounted to the host in docker-compose), so
# codex failures can be investigated without docker-exec'ing a live container.
LOG_DIR = FsPath(os.getenv("LOG_DIR", "/app/logs"))

# A single turn with xhigh reasoning over a real repo can run many minutes (a
# full code review hit the old 300s ceiling mid-tool-call and was killed, which
# the client saw as a 502). Bound it generously; the codex turn either finishes
# or this is the absolute backstop.
CODEX_EXEC_TIMEOUT = float(os.getenv("CODEX_EXEC_TIMEOUT", "900"))

_log_handlers: list[logging.Handler] = [logging.StreamHandler()]
try:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _log_handlers.append(
        RotatingFileHandler(
            LOG_DIR / "codex-rest.log", maxBytes=10_000_000, backupCount=5
        )
    )
except OSError:
    pass  # file logging is best-effort; never block startup on it
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=_log_handlers,
)
logger = logging.getLogger("codex-rest")


# ---------------------------------------------------------------------------
# codex --json event parsing
# ---------------------------------------------------------------------------
#
# With `--json`, codex exec streams newline-delimited JSON events to stdout.
# The two we care about (verified against codex-cli 0.139.0):
#   {"type": "thread.started", "thread_id": "<uuid>"}            -> session id
#   {"type": "item.completed", "item": {"type": "agent_message", -> answer text
#                                       "text": "..."}}
# The session id is needed once (first turn) to enable `exec resume`; the answer
# text is read primarily from the `-o` last-message file, with the agent_message
# event as a fallback.

def _iter_events(stdout: str):
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def _parse_thread_id(stdout: str) -> str | None:
    for ev in _iter_events(stdout):
        if ev.get("type") == "thread.started" and ev.get("thread_id"):
            return ev["thread_id"]
    return None


def _parse_agent_message(stdout: str) -> str:
    """Last agent_message text in the stream (matches `-o` semantics)."""
    text = ""
    for ev in _iter_events(stdout):
        if ev.get("type") == "item.completed":
            item = ev.get("item", {})
            if item.get("type") == "agent_message" and item.get("text"):
                text = item["text"]
    return text.strip()


# ---------------------------------------------------------------------------
# Codex Session (pure state — no live process)
# ---------------------------------------------------------------------------

@dataclass
class CodexSession:
    """A named conversation: a codex session UUID + a working directory.

    There is no process to manage. The first turn runs `codex exec` and records
    the session UUID from the thread.started event; subsequent turns run
    `codex exec resume <uuid>` so codex rehydrates context from its on-disk
    session log.
    """

    name: str
    _session_id: str | None = field(default=None, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _turn_count: int = field(default=0, init=False)
    # Bumped on clear/reset so the next turn lands in a fresh working dir and a
    # brand-new codex session — a genuine context wipe.
    _generation: int = field(default=0, init=False)

    # --- Identity ----------------------------------------------------------

    @property
    def cwd(self) -> FsPath:
        return SESSIONS_ROOT / self.name / f"c{self._generation}"

    @property
    def _output_file(self) -> FsPath:
        return self.cwd / ".codex-last.txt"

    # --- Command construction ---------------------------------------------

    def _base_args(self) -> list[str]:
        # --json: machine-readable events (we read thread_id from them)
        # --skip-git-repo-check: session dirs aren't git repos
        # --dangerously-bypass-approvals-and-sandbox: never block; the container
        #   is the sandbox (mirrors agy's "always-proceed"). Belt-and-suspenders
        #   alongside approval_policy=never in config.toml.
        args = [
            "--json",
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
        ]
        if CODEX_EXTRA_ARGS:
            args.extend(shlex.split(CODEX_EXTRA_ARGS))
        return args

    # --- Subprocess --------------------------------------------------------

    async def _run(self, argv: list[str], prompt: str) -> tuple[int, str, str]:
        """Run a codex invocation with the prompt on stdin; return (rc, out, err)."""
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(self.cwd),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(prompt.encode()), timeout=CODEX_EXEC_TIMEOUT
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            raise RuntimeError(f"codex exec timed out after {CODEX_EXEC_TIMEOUT:.0f}s")
        return proc.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")

    def _dump_failure(
        self, turn: int, argv: list[str], elapsed: float, *,
        reason: str, rc: int | None, stdout: str, stderr: str,
    ) -> None:
        """Snapshot a failed codex invocation to LOG_DIR for later investigation.

        Records the invocation (argv, cwd, session id), why it failed, and the
        stderr/stdout tails — the context that previously vanished into the 502
        body. Best-effort: never let diagnostics mask the real failure.
        """
        path = LOG_DIR / "timeouts" / f"codex-{self.name}-turn{turn}.log"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                f"session={self.name} turn={turn} reason={reason} rc={rc}",
                f"elapsed={elapsed:.1f}s session_id={self._session_id}",
                f"cwd={self.cwd}",
                f"argv={' '.join(argv)}",
                "\n=== stderr (last 2000 chars) ===",
                (stderr or "")[-2000:],
                "\n=== stdout (last 2000 chars) ===",
                (stdout or "")[-2000:],
            ]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            logger.warning("Session '%s': wrote codex failure diagnostic to %s", self.name, path)
        except OSError as exc:
            logger.warning(
                "Session '%s': could not write codex failure diagnostic: %s", self.name, exc
            )

    # --- Chat --------------------------------------------------------------

    async def send(self, prompt: str) -> str:
        """Send a prompt and return codex's final answer.

        The prompt is fed on stdin (the `-` argument), so newlines, quotes, and
        shell metacharacters arrive as literal text with no escaping concerns.
        """
        async with self._lock:
            self.cwd.mkdir(parents=True, exist_ok=True)
            turn = self._turn_count + 1  # committed only once the turn succeeds
            out_file = self._output_file
            try:
                out_file.unlink()  # avoid reading a stale answer
            except FileNotFoundError:
                pass

            resuming = self._session_id is not None
            logger.info(
                "Session '%s' turn %d — %s (%d chars)",
                self.name, turn,
                "resume" if resuming else "new session", len(prompt),
            )

            if resuming:
                argv = [CODEX_CMD, "exec", "resume", self._session_id,
                        *self._base_args(), "-o", str(out_file), "-"]
            else:
                argv = [CODEX_CMD, "exec",
                        *self._base_args(), "-o", str(out_file), "-"]

            run_start = time.monotonic()
            try:
                rc, stdout, stderr = await self._run(argv, prompt)
            except RuntimeError as exc:
                # timeout / spawn failure from _run: log the reason (it was only
                # ever surfaced in the 502 body before) and snapshot for later.
                run_elapsed = time.monotonic() - run_start
                logger.warning(
                    "Session '%s' turn %d — %s (after %.1fs)",
                    self.name, turn, exc, run_elapsed,
                )
                self._dump_failure(turn, argv, run_elapsed, reason=str(exc),
                                   rc=None, stdout="", stderr="")
                raise
            run_elapsed = time.monotonic() - run_start
            if rc != 0:
                detail = (stderr or stdout).strip()[-500:]
                logger.error(
                    "Session '%s' turn %d — codex exec failed in %.1fs (rc=%d): %s",
                    self.name, turn, run_elapsed, rc, detail,
                )
                self._dump_failure(turn, argv, run_elapsed, reason=f"rc={rc}",
                                   rc=rc, stdout=stdout, stderr=stderr)
                raise RuntimeError(f"codex exec failed (rc={rc}): {detail}")

            if self._session_id is None:
                self._session_id = _parse_thread_id(stdout)
                if self._session_id is None:
                    raise RuntimeError("Could not determine codex session id (no thread.started event)")
                logger.info("Session '%s' bound to codex thread %s", self.name, self._session_id)

            # The turn ran (rc==0, session bound) — commit the counter now so a
            # hard failure above never advances the client-visible turn number.
            self._turn_count = turn

            response = ""
            try:
                response = out_file.read_text(errors="replace").strip()
            except OSError:
                pass
            if not response:
                response = _parse_agent_message(stdout)

            logger.info(
                "Session '%s' turn %d — response (%.1fs, %d chars)",
                self.name, self._turn_count, run_elapsed, len(response),
            )
            return response

    # --- Lifecycle ---------------------------------------------------------

    async def clear(self) -> None:
        """Wipe context: new working dir + forget the codex session id.

        Unlike agy, codex does not summarize prior sessions into new ones, so
        simply starting a fresh session is a true context wipe.
        """
        async with self._lock:
            logger.info("Session '%s': clearing (fresh codex session)", self.name)
            self._generation += 1
            self._session_id = None
            self._turn_count = 0

    async def reset(self) -> None:
        """Identical to clear for codex (there is no process to reboot)."""
        await self.clear()

    async def stop(self) -> None:
        """Drop session state. No process to kill."""
        async with self._lock:
            self._session_id = None
            self._turn_count = 0

    # --- Status ------------------------------------------------------------

    async def is_alive(self) -> bool:
        # A codex session is "alive" once registered — it is just state.
        return True

    @property
    def turn_count(self) -> int:
        return self._turn_count


# ---------------------------------------------------------------------------
# Session Manager
# ---------------------------------------------------------------------------

@dataclass
class ChatManager:
    """Manages named chat sessions, each backed by its own CodexSession."""

    _sessions: dict[str, CodexSession] = field(default_factory=dict, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def get_or_create(self, name: str) -> CodexSession:
        async with self._lock:
            session = self._sessions.get(name)
            if session is None:
                logger.info("Creating new session: '%s'", name)
                session = CodexSession(name)
                self._sessions[name] = session
            return session

    async def get(self, name: str) -> CodexSession | None:
        async with self._lock:
            return self._sessions.get(name)

    async def remove(self, name: str) -> bool:
        async with self._lock:
            session = self._sessions.pop(name, None)
        if session:
            await session.stop()
            return True
        return False

    async def stop_all(self) -> int:
        async with self._lock:
            sessions = dict(self._sessions)
            self._sessions.clear()
        for session in sessions.values():
            await session.stop()
        return len(sessions)

    async def list_sessions(self) -> dict[str, dict]:
        async with self._lock:
            snapshot = dict(self._sessions)
        return {
            name: {"alive": await s.is_alive(), "turn_count": s.turn_count}
            for name, s in snapshot.items()
        }


manager = ChatManager()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Sessions are created lazily. On shutdown, drop all state."""
    SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)
    logger.info("Server starting (sessions will be created on demand)")
    yield
    logger.info("Shutting down — dropping all sessions...")
    count = await manager.stop_all()
    logger.info("Dropped %d session(s)", count)


app = FastAPI(
    title="Codex CLI REST Bridge",
    description=(
        "REST API that wraps OpenAI's Codex CLI (codex) in headless mode, "
        "providing named multi-session chat with conversation continuity. "
        "Mirrors the agy bridge's REST contract."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# --- Request/Response models ------------------------------------------------

_NAME = Path(..., pattern=r"^[a-zA-Z0-9_-]+$", max_length=64,
             description="Session name (alphanumeric, hyphens, underscores)")

class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500_000, description="The message to send")

class ChatResponse(BaseModel):
    response: str
    session: str
    turn: int
    elapsed_ms: int

class SessionStatus(BaseModel):
    name: str
    alive: bool
    turn_count: int

class HealthResponse(BaseModel):
    status: str
    active_sessions: int
    sessions: list[SessionStatus]

class MessageResponse(BaseModel):
    status: str
    message: str


# --- Endpoints --------------------------------------------------------------

@app.post("/chat/{name}", response_model=ChatResponse)
async def chat(req: ChatRequest, name: str = _NAME):
    """Send a message to a named session. Creates the session on first use."""
    session = await manager.get_or_create(name)

    t0 = time.monotonic()
    try:
        response = await session.send(req.prompt)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    elapsed = int((time.monotonic() - t0) * 1000)

    if not response:
        raise HTTPException(
            status_code=504,
            detail="No response received (timeout or empty output).",
        )

    return ChatResponse(
        response=response,
        session=name,
        turn=session.turn_count,
        elapsed_ms=elapsed,
    )


@app.post("/clear/{name}", response_model=MessageResponse)
async def clear(name: str = _NAME):
    """Clear conversation context for a session (starts a fresh codex session)."""
    session = await manager.get(name)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{name}' not found.")
    try:
        await session.clear()
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return MessageResponse(status="ok", message=f"Session '{name}' context cleared.")


@app.post("/reset/{name}", response_model=MessageResponse)
async def reset(name: str = _NAME):
    """Reset a specific session (fresh conversation)."""
    session = await manager.get(name)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{name}' not found.")
    try:
        await session.reset()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset session '{name}': {e}")
    return MessageResponse(status="ok", message=f"Session '{name}' reset.")


@app.delete("/chat/{name}", response_model=MessageResponse)
async def delete_session(name: str = _NAME):
    """Permanently remove a specific session."""
    removed = await manager.remove(name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Session '{name}' not found.")
    return MessageResponse(status="ok", message=f"Session '{name}' deleted.")


@app.post("/stop", response_model=MessageResponse)
async def stop_all():
    """Drop ALL sessions. Clean slate."""
    count = await manager.stop_all()
    return MessageResponse(status="ok", message=f"Stopped {count} session(s).")


@app.get("/health", response_model=HealthResponse)
async def health():
    """List all active sessions and their status."""
    sessions_info = await manager.list_sessions()
    session_list = [
        SessionStatus(name=name, alive=info["alive"], turn_count=info["turn_count"])
        for name, info in sessions_info.items()
    ]
    return HealthResponse(
        status="ok" if all(s.alive for s in session_list) or not session_list else "degraded",
        active_sessions=len(session_list),
        sessions=session_list,
    )
