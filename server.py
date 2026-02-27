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
RESPONSE_DONE_MARKER = os.getenv("RESPONSE_DONE_MARKER", "YOLO ctrl+y")  # YOLO status bar rendered when response is complete
# After /clear, the TUI status bar may omit "YOLO ctrl+y" (e.g. no file
# context), but "Type your message" always appears when ready for input.
# We split primary markers by comma (high-confidence, only need MIN_WAIT)
# and keep a separate fallback that checks the chunk contains
# "Type your message" but NOT "responding" (which the TUI shows while the
# model is still streaming).
RESPONSE_DONE_MARKERS_PRIMARY: list[str] = [
    m.strip() for m in RESPONSE_DONE_MARKER.split(",")
]
# How long after the last new Model: content before we consider the response
# complete (fallback when primary marker never appears, e.g. after /clear).
RESPONSE_MODEL_STABLE_TIMEOUT = float(os.getenv("RESPONSE_MODEL_STABLE_TIMEOUT", "3"))
CLEAR_SETTLE_DELAY = float(os.getenv("CLEAR_SETTLE_DELAY", "0.2"))  # seconds to wait after /clear for TUI to re-attach stdin
RESPONSE_MIN_WAIT = float(os.getenv("RESPONSE_MIN_WAIT", "2"))  # seconds to ignore done_marker after sending (avoids false-positive from TUI status bar redraw)

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


_TUI_CHROME_PREFIXES = (
    "User:", "/app ", "responding", "── Shortcuts",
    "❯", ">", ">>>", "│", "╭", "╰", "─",
    "? for shortcuts", "YOLO mode", "YOLO ctrl+y",
    "Press Ctrl+O", "[Pasted Text",
)


def _is_tui_chrome(line: str) -> bool:
    """Return True if *line* is TUI chrome rather than model content.

    Blank lines are NOT considered chrome — they can appear inside
    multi-line model responses (e.g. between paragraphs or code blocks).
    """
    stripped = line.strip()
    if not stripped:
        return False
    return any(stripped.startswith(p) for p in _TUI_CHROME_PREFIXES)


def clean_response(raw: str, prompt_sent: str) -> str:
    """
    Extract the model's response from raw PTY output.

    In screen-reader mode, the model's response lines are prefixed with
    "Model:".  The TUI redraws these lines as the response streams in, so
    each successive "Model:" block is a progressively longer snapshot.

    Multi-line responses (e.g. code blocks) are rendered as a ``Model:``
    line followed by continuation lines that lack the prefix.  We collect
    each block (``Model:`` line + continuations) and return the longest one.
    """
    text = strip_ansi(raw)

    # --- Primary: block-based extraction from Model: prefixed output ---
    # A Model: line starts a new block.  Subsequent lines are appended
    # to the block *unless* they are TUI chrome, which is silently
    # skipped (NOT treated as a block terminator — the TUI interleaves
    # chrome like "/app ..." and "responding ..." between content lines
    # during redraws).  A block only ends when the next Model: line
    # starts a new one, or we reach end-of-input.
    model_re = re.compile(r'Model:\s+(.*)')
    blocks: list[list[str]] = []       # each element is a list of lines
    current_block: list[str] | None = None

    for line in text.splitlines():
        match = model_re.search(line)
        if match:
            # Start a new block
            content = match.group(1).strip()
            current_block = [content] if content else []
            blocks.append(current_block)
        elif current_block is not None:
            if _is_tui_chrome(line):
                continue              # skip chrome, keep block open
            stripped = line.strip()
            if stripped:
                current_block.append(stripped)
            # blank lines are silently skipped (common between redraws)

    if blocks:
        # Pick the block with the most total content
        def _block_len(b: list[str]) -> int:
            return sum(len(l) for l in b)
        best = max(blocks, key=_block_len)
        return "\n".join(best).strip()

    # --- Fallback: remove known TUI artifacts and echoed prompt ---
    lines = text.splitlines()
    cleaned: list[str] = []
    prompt_stripped = prompt_sent.strip()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == prompt_stripped:
            continue
        if _is_tui_chrome(line):
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
        # The TUI continues rendering the status bar after "Type your message".
        # Give it time to finish, then discard so it doesn't pollute the first turn.
        await asyncio.sleep(0.5)
        await self._drain_buffer()
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

    async def _drain_buffer(self) -> None:
        """Drain any pending output from the PTY buffer.

        This prevents _wait_for_ready from matching stale markers
        left over from previous interactions.
        """
        loop = asyncio.get_event_loop()
        drained = 0
        while True:
            try:
                chunk = await loop.run_in_executor(
                    None, lambda: self._child.read_nonblocking(65536, timeout=0.3)
                )
                if chunk:
                    drained += len(chunk)
            except (pexpect.TIMEOUT, pexpect.EOF):
                break
        if drained:
            logger.debug("Drained %d bytes of stale buffer", drained)

    async def clear(self) -> None:
        """Send /clear to the CLI to clear screen and conversation history."""
        async with self._lock:
            if not self._child or not self._child.isalive():
                raise RuntimeError("Gemini CLI process is not running")

            logger.info("Clearing conversation context via /clear")

            # Drain stale PTY output so _wait_for_ready won't match
            # an old "Type your message" marker from before the /clear.
            await self._drain_buffer()

            self._child.send("/clear")
            await asyncio.sleep(0.1)
            self._child.send("\r")
            self._turn_count = 0

            # Wait for the CLI to be ready again (fresh marker)
            await self._wait_for_ready()

            # The Ink TUI redraws "Type your message" almost instantly after
            # /clear, but its stdin handler isn't fully re-attached yet.
            # Without this delay, the next prompt sent gets swallowed.
            await asyncio.sleep(CLEAR_SETTLE_DELAY)
            await self._drain_buffer()

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

            # Drain stale output (startup status bar, previous turn leftovers)
            # so _collect_response doesn't false-positive on an old done marker.
            await self._drain_buffer()

            # Wrap prompt in bracketed paste mode so the Ink TUI treats it as
            # literal text, not as individual key events.  Without this,
            # characters like '!' (shell mode), backticks, '@', and '\n'
            # trigger TUI shortcuts instead of being sent as text.
            #
            # The Ink TUI collapses ANY multi-line bracketed paste into an
            # unsubmittable "[Pasted Text: N lines]" widget.  For large
            # prompts we write to a temp file and use the @file syntax so
            # the CLI reads the file and includes its content in the API
            # request to the model.
            PASTE_FILE_THRESHOLD = 50_000   # chars — above this, use @file

            temp_path: str | None = None
            try:
                if len(prompt) > PASTE_FILE_THRESHOLD:
                    temp_path = f"/app/.gemini_prompt_{id(self)}_{self._turn_count}.md"
                    with open(temp_path, "w") as f:
                        f.write(prompt)
                    logger.info("Large prompt — wrote %d chars to %s", len(prompt), temp_path)
                    self._child.send(f"@{temp_path}")
                    await asyncio.sleep(0.5)
                else:
                    self._child.send("\x1b[200~")
                    self._child.send(prompt)
                    self._child.send("\x1b[201~")
                    await asyncio.sleep(0.1)

                self._child.send("\r")

                # Collect output
                response = await self._collect_response(prompt)
            finally:
                if temp_path:
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass

            logger.info("Turn %d — response collected (%d chars)", self._turn_count, len(response))
            return response

    async def _collect_response(self, prompt_sent: str) -> str:
        """Read output until we're confident the response is complete.

        Detection strategy (two tiers):

        1. **Primary markers** (e.g. "YOLO ctrl+y"): high-confidence TUI
           status-bar indicators that the CLI is idle. Only requires
           RESPONSE_MIN_WAIT to have elapsed.

        2. **Model-content stabilization** (fallback): after /clear the primary
           marker may never appear.  Instead we watch for ``Model:`` prefixed
           lines in the output.  The ``clean_response`` function takes the
           *longest* ``Model:`` segment as the final answer, so once no chunk
           has contained a *longer* ``Model:`` segment for
           RESPONSE_MODEL_STABLE_TIMEOUT seconds, the answer has stabilized
           and we can stop reading.
        """
        loop = asyncio.get_event_loop()
        chunks: list[str] = []
        start = time.monotonic()
        exit_reason = "unknown"

        # Fallback tracking: length of longest Model: content seen so far,
        # and the timestamp at which it last grew.
        best_model_len = 0
        last_model_growth = 0.0  # monotonic time of last growth
        last_responding = 0.0    # monotonic time "responding" last seen in output
        model_re = re.compile(r'Model:\s+(.*)')
        paste_resubmitted = False  # only retry once

        while True:
            elapsed = time.monotonic() - start
            if elapsed > RESPONSE_MAX_TIMEOUT:
                exit_reason = "max_timeout"
                logger.warning("Hit max timeout (%.0fs) — returning partial response", RESPONSE_MAX_TIMEOUT)
                break

            # --- Fallback: model content stabilized? ---
            # Don't trigger if "responding" appeared recently — the model
            # may be executing a tool (file read, shell command) in YOLO
            # mode and Model: content pauses during tool execution.
            if best_model_len > 0 and elapsed >= RESPONSE_MIN_WAIT:
                since_growth = time.monotonic() - last_model_growth
                since_responding = time.monotonic() - last_responding
                if (since_growth >= RESPONSE_MODEL_STABLE_TIMEOUT
                        and since_responding >= RESPONSE_MODEL_STABLE_TIMEOUT):
                    exit_reason = "model_stable"
                    logger.info(
                        "Model content stabilized (%d chars, %.1fs since last growth)",
                        best_model_len, since_growth,
                    )
                    break

            try:
                chunk = await loop.run_in_executor(
                    None,
                    lambda: self._child.read_nonblocking(65536, timeout=RESPONSE_IDLE_TIMEOUT),
                )
                if chunk:
                    chunks.append(chunk)
                    stripped_chunk = strip_ansi(chunk)
                    logger.debug("Chunk (%d bytes): %r", len(chunk), chunk[:200])

                    # Track longest Model: content for stabilization fallback
                    for line in stripped_chunk.splitlines():
                        match = model_re.search(line)
                        if match:
                            content_len = len(match.group(1).strip())
                            if content_len > best_model_len:
                                best_model_len = content_len
                                last_model_growth = time.monotonic()
                                logger.debug("Model content grew to %d chars at %.1fs", best_model_len, elapsed)

                    chunk_lower = stripped_chunk.lower()

                    # Track "responding" indicator — present while the model
                    # is actively streaming.  The TUI status bar always
                    # contains "YOLO ctrl+y" (it's a persistent element),
                    # so "responding" in the same chunk means the model is
                    # NOT done yet — we must not treat the marker as a
                    # completion signal.
                    has_responding = "responding" in chunk_lower
                    if has_responding:
                        last_responding = time.monotonic()

                    # --- Paste-collapse recovery ---
                    # For very large inputs the Gemini CLI may collapse the
                    # bracketed paste into a "[Pasted Text: N lines]" widget
                    # without actually submitting it to the model.  If we
                    # detect this pattern early (before any model output has
                    # appeared), re-send Enter to trigger submission.
                    if (not paste_resubmitted
                            and elapsed < 3.0
                            and best_model_len == 0
                            and ("pasted text" in chunk_lower
                                 or "ctrl+o" in chunk_lower)
                            and not has_responding):
                        paste_resubmitted = True
                        logger.info(
                            "Paste-collapse detected at %.1fs — re-sending Enter to submit",
                            elapsed,
                        )
                        self._child.send("\r")
                        # Reset state so detection timers apply to the real response
                        start = time.monotonic()
                        chunks.clear()
                        best_model_len = 0
                        last_model_growth = 0.0
                        continue

                    # --- Tier 1: primary markers (high confidence) ---
                    # Guard: skip if "responding" is in the same chunk —
                    # "YOLO ctrl+y" is always in the status bar, even
                    # mid-stream.  It only signals completion when
                    # "responding" is absent.
                    # Also require that we've actually seen Model: content —
                    # the TUI redraws "YOLO ctrl+y" during the thinking
                    # phase before any model output; without this guard
                    # we false-positive on a status-bar redraw chunk that
                    # happens to lack "responding".
                    primary_hit = (
                        not has_responding
                        and best_model_len > 0
                        and any(
                            m.lower() in chunk_lower
                            for m in RESPONSE_DONE_MARKERS_PRIMARY
                        )
                    )
                    if primary_hit:
                        if elapsed >= RESPONSE_MIN_WAIT:
                            exit_reason = "done_marker"
                            logger.debug("Primary done marker matched at %.1fs", elapsed)
                            break
                        else:
                            logger.debug("Ignoring early primary marker at %.1fs (min_wait=%.1fs)", elapsed, RESPONSE_MIN_WAIT)

                    # --- Tier 2: secondary marker ("Type your message" without "responding") ---
                    # For plain-text responses (no tool use / file writes), the TUI
                    # never shows the primary "YOLO ctrl+y" marker. Instead it shows
                    # "Type your message" once the model finishes. While still
                    # streaming, the TUI also shows "responding" alongside "Type your
                    # message", so we require its absence as a completion signal.
                    # Same guard as Tier 1: must have seen Model: content first.
                    if elapsed >= RESPONSE_MIN_WAIT and best_model_len > 0:
                        has_ready = READY_MARKER.lower() in chunk_lower
                        if has_ready and not has_responding:
                            exit_reason = "done_marker_secondary"
                            logger.debug("Secondary done marker (ready w/o responding) at %.1fs", elapsed)
                            break

            except pexpect.TIMEOUT:
                # Idle timeout — response is likely complete
                if chunks:
                    exit_reason = "idle_timeout"
                    break
                # No output yet — keep waiting (model might be thinking)
                if elapsed < RESPONSE_MAX_TIMEOUT:
                    continue
                exit_reason = "idle_timeout_no_output"
                break
            except pexpect.EOF:
                logger.error("Gemini CLI process terminated unexpectedly")
                self._ready = False
                raise RuntimeError("Gemini CLI process died during response")

        raw = "".join(chunks)
        # Log how the response ended and a tail of the stripped output for debugging
        stripped_tail = strip_ansi(raw)[-300:] if raw else "(empty)"
        logger.info(
            "Response complete via %s (%.1fs, %d bytes raw). Tail: %r",
            exit_reason, time.monotonic() - start, len(raw), stripped_tail,
        )

        # Safety net: detect if the TUI accidentally entered shell mode.
        # If so, send Escape to exit and log a warning.
        stripped_raw = strip_ansi(raw).lower()
        if "shell mode" in stripped_raw or "type your shell command" in stripped_raw:
            logger.warning(
                "Shell mode detected in response — sending Escape to exit"
            )
            self._child.send("\x1b")        # Escape exits shell mode
            await asyncio.sleep(0.3)
            await self._drain_buffer()       # discard shell-mode TUI output

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
                if session.is_alive:
                    return session
                logger.info("Session '%s' found dead, restarting", name)
            else:
                logger.info("Creating new session: '%s'", name)
                session = GeminiProcess()
                self._sessions[name] = session

        # Start/restart outside the manager lock so other sessions aren't
        # blocked during the ~10s startup. The session's own _lock in
        # start() serializes concurrent starts for the same session.
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
