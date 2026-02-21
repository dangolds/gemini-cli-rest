"""
REST API server that wraps Gemini CLI interactive mode.

Communicates with Gemini CLI via PTY (pseudo-terminal) using pexpect,
exposing named multi-session chat endpoints for programmatic access.
Each session (/chat/{name}) gets its own CLI process with isolated context.
"""

import asyncio
import logging
import os
import re
import signal
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import pexpect
from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
GEMINI_CMD = os.getenv("GEMINI_CMD", "gemini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "")  # e.g. "gemini-2.5-pro"
GEMINI_YOLO = os.getenv("GEMINI_YOLO", "true").lower() == "true"
GEMINI_SCREEN_READER = os.getenv("GEMINI_SCREEN_READER", "true").lower() == "true"
GEMINI_EXTRA_ARGS = os.getenv("GEMINI_EXTRA_ARGS", "")  # space-separated extras
RESPONSE_IDLE_TIMEOUT = float(os.getenv("RESPONSE_IDLE_TIMEOUT", "5"))  # seconds of silence → response done
RESPONSE_MAX_TIMEOUT = float(os.getenv("RESPONSE_MAX_TIMEOUT", "120"))  # hard cap per request
STARTUP_TIMEOUT = float(os.getenv("STARTUP_TIMEOUT", "60"))

# Marker that Gemini CLI prints when ready for input (screen-reader mode)
READY_MARKER = os.getenv("READY_MARKER", "Type your message")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gemini-rest")

# ---------------------------------------------------------------------------
# ANSI / control-character stripping
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"""
    \x1b       # ESC
    (?:
        \[       # CSI
        [0-?]*   # parameter bytes
        [ -/]*   # intermediate bytes
        [@-~]    # final byte
    |
        \]       # OSC
        .*?      # payload
        (?:\x07|\x1b\\)  # ST (BEL or ESC\)
    |
        [()][AB012]  # charset selection
    |
        [=><=]       # misc escapes
    |
        \[[0-9;]*[A-HJKSTfhlmnsu]  # catch-all CSI
    )
""", re.VERBOSE)

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes and stray control characters."""
    text = _ANSI_RE.sub("", text)
    text = _CONTROL_RE.sub("", text)
    return text


def clean_response(raw: str, prompt_sent: str) -> str:
    """
    Extract the model's response from raw PTY output.

    In screen-reader mode, the model's response lines are prefixed with "Model:".
    The TUI redraws these lines as the response streams in, so each successive
    "Model:" block is a longer version of the response. We take the last one.
    """
    text = strip_ansi(raw)

    # Extract "Model: ..." lines — each is a progressively longer snapshot
    # of the streaming response. The last/longest one is the complete answer.
    model_contents = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Model:"):
            content = stripped[len("Model:"):].strip()
            if content:
                model_contents.append(content)

    if model_contents:
        # Take the longest entry (the final complete response)
        return max(model_contents, key=len)

    # Fallback: remove known TUI artifacts and echoed prompt
    lines = text.splitlines()
    cleaned: list[str] = []
    prompt_stripped = prompt_sent.strip()
    # Patterns that are TUI chrome, not content
    skip_patterns = {
        "❯", ">", ">>>", "│", "╭", "╰", "─",
        "? for shortcuts", "YOLO mode", "YOLO ctrl+y",
    }
    for line in lines:
        stripped = line.strip()
        if stripped == prompt_stripped:
            continue
        if stripped.startswith("User:"):
            continue
        if stripped.startswith("/app "):
            continue
        if stripped.startswith("responding"):
            continue
        if any(stripped.startswith(p) for p in skip_patterns):
            continue
        if stripped.startswith("── Shortcuts"):
            continue
        if stripped in skip_patterns:
            continue
        if not stripped:
            continue
        cleaned.append(line)

    result = "\n".join(cleaned).strip()
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result


# ---------------------------------------------------------------------------
# Gemini CLI Process Manager
# ---------------------------------------------------------------------------

@dataclass
class GeminiProcess:
    """Manages a single Gemini CLI interactive subprocess."""

    _child: pexpect.spawn | None = field(default=None, init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _ready: bool = field(default=False, init=False)
    _turn_count: int = field(default=0, init=False)

    # --- Lifecycle -----------------------------------------------------------

    def _build_command(self) -> str:
        parts = [GEMINI_CMD]
        if GEMINI_SCREEN_READER:
            parts.append("--screen-reader")
        if GEMINI_YOLO:
            parts.append("--yolo")
        if GEMINI_MODEL:
            parts.extend(["--model", GEMINI_MODEL])
        if GEMINI_EXTRA_ARGS:
            parts.extend(GEMINI_EXTRA_ARGS.split())
        return " ".join(parts)

    async def start(self) -> None:
        """Spawn the Gemini CLI process and wait until it's ready."""
        async with self._lock:
            if self._child and self._child.isalive():
                logger.info("Process already running (pid=%s)", self._child.pid)
                return
            await self._spawn()

    async def _spawn(self) -> None:
        cmd = self._build_command()
        logger.info("Spawning: %s", cmd)

        env = os.environ.copy()
        env["TERM"] = "dumb"  # minimize TUI escape codes
        env["NO_COLOR"] = "1"

        self._child = pexpect.spawn(
            "/bin/bash",
            ["-c", cmd],
            encoding="utf-8",
            timeout=STARTUP_TIMEOUT,
            env=env,
            maxread=65536,
            echo=False,
        )
        self._child.setwinsize(50, 200)  # wide terminal avoids line-wrapping
        self._turn_count = 0
        self._ready = False

        # Wait for the CLI to become ready (input prompt / initial output)
        await self._wait_for_ready()
        self._ready = True
        logger.info("Gemini CLI ready (pid=%s)", self._child.pid)

    async def _wait_for_ready(self) -> None:
        """
        Consume initial startup output until we detect the CLI is ready.

        Waits for the READY_MARKER string (e.g. "Type your message") which
        indicates the CLI has finished auth and is accepting input.
        Falls back to idle-timeout if the marker isn't found.
        """
        loop = asyncio.get_event_loop()
        start = time.monotonic()
        accumulated = ""

        while time.monotonic() - start < STARTUP_TIMEOUT:
            try:
                chunk = await loop.run_in_executor(
                    None, lambda: self._child.read_nonblocking(4096, timeout=3)
                )
                if chunk:
                    accumulated += chunk
                    logger.debug("Startup output: %r", chunk[:200])
                    # Check for the ready marker in accumulated output
                    if READY_MARKER in strip_ansi(accumulated):
                        logger.info(
                            "Ready marker '%s' detected after %.1fs",
                            READY_MARKER, time.monotonic() - start,
                        )
                        return
            except pexpect.TIMEOUT:
                # If we've already seen substantial output, silence means ready
                if len(accumulated) > 100:
                    logger.info("Startup silence after output → ready (%.1fs)", time.monotonic() - start)
                    return
                # Otherwise keep waiting (CLI hasn't started outputting yet)
                continue
            except pexpect.EOF:
                raise RuntimeError(
                    "Gemini CLI exited during startup. "
                    "Check that 'gemini' is installed and authenticated."
                )

        # If we got output but no marker, assume ready anyway
        if accumulated:
            logger.warning("Ready marker not found after %.0fs, proceeding anyway", STARTUP_TIMEOUT)
            return

        raise RuntimeError("Gemini CLI startup timed out")

    async def stop(self) -> None:
        """Gracefully terminate the subprocess."""
        async with self._lock:
            self._kill()

    def _kill(self) -> None:
        if self._child and self._child.isalive():
            logger.info("Stopping Gemini CLI (pid=%s)", self._child.pid)
            try:
                self._child.sendline("/quit")
                self._child.expect(pexpect.EOF, timeout=5)
            except Exception:
                pass
            finally:
                if self._child.isalive():
                    self._child.kill(signal.SIGTERM)
                    time.sleep(0.5)
                    if self._child.isalive():
                        self._child.kill(signal.SIGKILL)
            self._child = None
            self._ready = False
            self._turn_count = 0

    async def reset(self) -> None:
        """Kill and re-spawn the process (new conversation)."""
        async with self._lock:
            self._kill()
            await self._spawn()
        return

    async def clear(self) -> None:
        """Send /clear to the CLI to clear screen and conversation history."""
        async with self._lock:
            if not self._child or not self._child.isalive():
                raise RuntimeError("Gemini CLI process is not running")

            logger.info("Clearing conversation context via /clear")
            self._child.send("/clear")
            await asyncio.sleep(0.1)
            self._child.send("\r")
            self._turn_count = 0

            # Wait for the CLI to be ready again
            await self._wait_for_ready()

    # --- Chat ----------------------------------------------------------------

    async def send(self, prompt: str) -> str:
        """
        Send a prompt and return Gemini's response.

        Detection strategy: we read output chunks and consider the response
        complete when we've seen RESPONSE_IDLE_TIMEOUT seconds of silence.
        """
        async with self._lock:
            if not self._child or not self._child.isalive():
                raise RuntimeError("Gemini CLI process is not running")

            self._turn_count += 1
            logger.info("Turn %d — sending prompt (%d chars)", self._turn_count, len(prompt))

            # Send the prompt text, then a separate \r after a brief pause.
            # Ink (React TUI) processes stdin byte-by-byte as key events.
            # Sending text+Enter as one burst causes Enter to be swallowed.
            self._child.send(prompt)
            await asyncio.sleep(0.1)
            self._child.send("\r")

            # Collect output
            response = await self._collect_response(prompt)
            logger.info("Turn %d — response collected (%d chars)", self._turn_count, len(response))
            return response

    async def _collect_response(self, prompt_sent: str) -> str:
        """Read output until idle-timeout indicates the response is complete."""
        loop = asyncio.get_event_loop()
        chunks: list[str] = []
        start = time.monotonic()

        while True:
            elapsed = time.monotonic() - start
            if elapsed > RESPONSE_MAX_TIMEOUT:
                logger.warning("Hit max timeout (%.0fs) — returning partial response", RESPONSE_MAX_TIMEOUT)
                break

            try:
                chunk = await loop.run_in_executor(
                    None,
                    lambda: self._child.read_nonblocking(65536, timeout=RESPONSE_IDLE_TIMEOUT),
                )
                if chunk:
                    chunks.append(chunk)
                    logger.debug("Chunk (%d bytes): %r", len(chunk), chunk[:120])
            except pexpect.TIMEOUT:
                # Idle timeout — response is likely complete
                if chunks:
                    logger.debug("Idle timeout after output → response complete")
                    break
                # No output yet — keep waiting (model might be thinking)
                if elapsed < RESPONSE_MAX_TIMEOUT:
                    continue
                break
            except pexpect.EOF:
                logger.error("Gemini CLI process terminated unexpectedly")
                self._ready = False
                raise RuntimeError("Gemini CLI process died during response")

        raw = "".join(chunks)
        return clean_response(raw, prompt_sent)

    # --- Status --------------------------------------------------------------

    @property
    def is_alive(self) -> bool:
        return bool(self._child and self._child.isalive())

    @property
    def turn_count(self) -> int:
        return self._turn_count


# ---------------------------------------------------------------------------
# Session Manager
# ---------------------------------------------------------------------------

@dataclass
class ChatManager:
    """Manages named chat sessions, each backed by its own GeminiProcess."""

    _sessions: dict[str, GeminiProcess] = field(default_factory=dict, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def get_or_create(self, name: str) -> GeminiProcess:
        """Return an existing session or create and start a new one."""
        async with self._lock:
            if name in self._sessions:
                session = self._sessions[name]
                if not session.is_alive:
                    logger.info("Session '%s' found dead, restarting", name)
                    await session.start()
                return session

            logger.info("Creating new session: '%s'", name)
            session = GeminiProcess()
            self._sessions[name] = session

        # Start outside the manager lock so other sessions aren't blocked
        # during the ~10s startup.
        await session.start()
        return session

    async def get(self, name: str) -> GeminiProcess | None:
        """Return a session by name, or None."""
        async with self._lock:
            return self._sessions.get(name)

    async def remove(self, name: str) -> bool:
        """Stop and remove a single session. Returns True if it existed."""
        async with self._lock:
            session = self._sessions.pop(name, None)
        if session:
            await session.stop()
            return True
        return False

    async def stop_all(self) -> int:
        """Stop and remove ALL sessions. Returns count stopped."""
        async with self._lock:
            sessions = dict(self._sessions)
            self._sessions.clear()
        for session in sessions.values():
            await session.stop()
        return len(sessions)

    async def list_sessions(self) -> dict[str, dict]:
        """Return status info for all sessions."""
        async with self._lock:
            snapshot = dict(self._sessions)
        return {
            name: {"alive": s.is_alive, "turn_count": s.turn_count}
            for name, s in snapshot.items()
        }


manager = ChatManager()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Sessions are created lazily. On shutdown, stop all."""
    logger.info("Server starting (sessions will be created on demand)")
    yield
    logger.info("Shutting down — stopping all sessions...")
    count = await manager.stop_all()
    logger.info("Stopped %d session(s)", count)


app = FastAPI(
    title="Gemini CLI REST Bridge",
    description=(
        "REST API that wraps Google's Gemini CLI interactive mode, "
        "providing named multi-session chat with conversation continuity."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# --- Request/Response models ------------------------------------------------

_NAME = Path(..., pattern=r"^[a-zA-Z0-9_-]+$", max_length=64,
             description="Session name (alphanumeric, hyphens, underscores)")

class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=50_000, description="The message to send")

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


# --- Endpoints ---------------------------------------------------------------

@app.post("/chat/{name}", response_model=ChatResponse)
async def chat(req: ChatRequest, name: str = _NAME):
    """Send a message to a named session. Creates the session on first use."""
    try:
        session = await manager.get_or_create(name)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Failed to start session '{name}': {e}")

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
    """Clear conversation context for a session without restarting the process."""
    session = await manager.get(name)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{name}' not found.")
    if not session.is_alive:
        raise HTTPException(status_code=503, detail=f"Session '{name}' is not running. Use /reset/{name}.")
    try:
        await session.clear()
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return MessageResponse(status="ok", message=f"Session '{name}' context cleared.")


@app.post("/reset/{name}", response_model=MessageResponse)
async def reset(name: str = _NAME):
    """Kill and restart a specific session (fresh conversation)."""
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
    """Kill and permanently remove a specific session."""
    removed = await manager.remove(name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Session '{name}' not found.")
    return MessageResponse(status="ok", message=f"Session '{name}' deleted.")


@app.post("/stop", response_model=MessageResponse)
async def stop_all():
    """Kill ALL sessions. Clean slate."""
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
