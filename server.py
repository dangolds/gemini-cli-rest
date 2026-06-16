"""
REST API server that wraps Antigravity CLI (agy) interactive mode.

Each named session (/chat/{name}) runs its own live `agy` process inside a
detached tmux session. tmux acts as a terminal emulator mediator: instead of
parsing the raw PTY escape-code stream, we read the *rendered* screen with
`tmux capture-pane` (used only for busy/idle detection) and extract the
model's response as structured data from agy's per-conversation transcript
file (brain/<conversation>/.system_generated/logs/transcript.jsonl).
"""

import asyncio
import json
import logging
import os
import shlex
import time
import uuid
from logging.handlers import RotatingFileHandler
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path as FsPath

from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
AGY_CMD = os.getenv("AGY_CMD", "agy")
# Auto-approve tool permission prompts. Prefer setting
# "toolPermission": "always-proceed" in agy's settings.json instead (the
# Docker entrypoint seeds it); this flag is the per-process equivalent.
AGY_SKIP_PERMISSIONS = os.getenv("AGY_SKIP_PERMISSIONS", "false").lower() == "true"
AGY_EXTRA_ARGS = os.getenv("AGY_EXTRA_ARGS", "")  # space-separated extras

TMUX_BIN = os.getenv("TMUX_BIN", "tmux")
# Dedicated tmux server socket so we never collide with a user's tmux.
TMUX_SOCKET = os.getenv("TMUX_SOCKET", "agy-rest")
TERM_WIDTH = int(os.getenv("TERM_WIDTH", "200"))
TERM_HEIGHT = int(os.getenv("TERM_HEIGHT", "50"))

# Where agy keeps its state (brain/<conversation-id>/... transcripts).
AGY_STATE_DIR = FsPath(
    os.getenv("AGY_STATE_DIR", os.path.expanduser("~/.gemini/antigravity-cli"))
)
# Each session gets its own working directory so agy's per-cwd project state
# never crosses between sessions. The per-run id keeps paths unique across
# server restarts: agy keys conversation memory by directory path, so reusing
# a path would hand a "fresh" session the previous run's conversation history.
RUN_ID = uuid.uuid4().hex[:8]
SESSIONS_ROOT = FsPath(os.getenv("SESSIONS_ROOT", "/tmp/agy-rest-sessions")) / RUN_ID

# Local, persisted log destination. Mounted to the host in docker-compose so
# turns can be investigated after the fact without docker-exec'ing a live
# container. Holds the rolling server log and per-incident timeout dumps.
LOG_DIR = FsPath(os.getenv("LOG_DIR", "/app/logs"))
TIMEOUT_LOG_DIR = LOG_DIR / "timeouts"
TIMEOUT_DUMP_STEPS = int(os.getenv("TIMEOUT_DUMP_STEPS", "40"))

STARTUP_TIMEOUT = float(os.getenv("STARTUP_TIMEOUT", "60"))
RESPONSE_POLL_INTERVAL = float(os.getenv("RESPONSE_POLL_INTERVAL", "0.5"))
RESPONSE_MIN_WAIT = float(os.getenv("RESPONSE_MIN_WAIT", "1"))
# A turn ends when agy STOPS MAKING PROGRESS, not just on a flat wall-clock —
# this lets a fast stall be cut off early instead of always burning the full
# budget. But the request is still HARD-CAPPED (RESPONSE_HARD_TIMEOUT) so it
# never outlives the client/proxy that's waiting on it. The cap stays ~140s by
# design; longer agentic turns are handled by the client re-polling, not by the
# server blocking for minutes.
RESPONSE_STALL_TIMEOUT = float(os.getenv("RESPONSE_STALL_TIMEOUT", "90"))
RESPONSE_HARD_TIMEOUT = float(
    # honor a legacy RESPONSE_MAX_TIMEOUT as the hard ceiling if someone set it
    os.getenv("RESPONSE_HARD_TIMEOUT", os.getenv("RESPONSE_MAX_TIMEOUT", "140"))
)
# After submitting, how long to wait for agy to *acknowledge* the turn (a new
# transcript step appears, or the screen goes busy) before concluding the
# submit Enter was dropped and re-pressing it. Kept deliberately generous: a
# genuinely-accepted submit acknowledges within ~1s, so waiting this long means
# a merely-slow-but-accepted submit is never mistaken for a drop and re-sent —
# which is what would otherwise produce a duplicate message.
SUBMIT_CONFIRM_WAIT = float(os.getenv("SUBMIT_CONFIRM_WAIT", "8"))
SUBMIT_MAX_RETRIES = int(os.getenv("SUBMIT_MAX_RETRIES", "2"))
# How long after /clear / startup we wait for agy to register the new
# conversation (brain dir appears).
CONVERSATION_DETECT_TIMEOUT = float(os.getenv("CONVERSATION_DETECT_TIMEOUT", "20"))

# Rendered-screen markers (read from tmux's emulated screen, never from the
# raw escape stream, so these are stable plain-text strings).
READY_MARKER = "? for shortcuts"          # idle status bar
BUSY_MARKERS = ("Generating...", "esc to cancel")

# tmux client commands are sub-second; anything longer means a stuck client.
TMUX_CMD_TIMEOUT = float(os.getenv("TMUX_CMD_TIMEOUT", "15"))

_log_handlers: list[logging.Handler] = [logging.StreamHandler()]
try:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _log_handlers.append(
        RotatingFileHandler(
            LOG_DIR / "agy-rest.log", maxBytes=10_000_000, backupCount=5
        )
    )
except OSError:
    pass  # file logging is best-effort; never block startup on it
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=_log_handlers,
)
logger = logging.getLogger("agy-rest")


# ---------------------------------------------------------------------------
# Subprocess / tmux helpers
# ---------------------------------------------------------------------------

async def _exec(*argv: str, stdin_data: bytes | None = None) -> tuple[int, str]:
    """Run a command, return (returncode, combined output)."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE if stdin_data is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(
            proc.communicate(stdin_data), timeout=TMUX_CMD_TIMEOUT
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        raise RuntimeError(f"Command timed out: {' '.join(argv[:4])}...")
    return proc.returncode or 0, out.decode(errors="replace")


async def _tmux(*args: str, stdin_data: bytes | None = None) -> tuple[int, str]:
    return await _exec(TMUX_BIN, "-L", TMUX_SOCKET, *args, stdin_data=stdin_data)


async def _ensure_tmux_server() -> None:
    """Start the dedicated tmux server with fully detached stdio.

    The first tmux client on a fresh socket forks the tmux server daemon.
    Under uvloop, a pipe inherited by that daemon is never seen as closed,
    which would hang the first session spawn — so the daemon is started here
    with DEVNULL stdio. "exit-empty off" keeps it alive with zero sessions.
    """
    proc = await asyncio.create_subprocess_exec(
        TMUX_BIN, "-L", TMUX_SOCKET,
        "start-server", ";", "set", "-g", "exit-empty", "off",
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()


# ---------------------------------------------------------------------------
# agy transcript access
# ---------------------------------------------------------------------------

def _brain_dirs() -> dict[str, float]:
    """Map conversation-id -> mtime for all known agy conversations."""
    brain = AGY_STATE_DIR / "brain"
    if not brain.is_dir():
        return {}
    result: dict[str, float] = {}
    for entry in brain.iterdir():
        if entry.is_dir():
            try:
                result[entry.name] = entry.stat().st_mtime
            except OSError:
                continue
    return result


def _transcript_path(conversation_id: str) -> FsPath:
    return (
        AGY_STATE_DIR / "brain" / conversation_id
        / ".system_generated" / "logs" / "transcript.jsonl"
    )


def _read_transcript(conversation_id: str) -> list[dict]:
    """Parse transcript.jsonl into a list of step dicts (empty if absent)."""
    path = _transcript_path(conversation_id)
    if not path.is_file():
        return []
    steps: list[dict] = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    steps.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # partially-written line; next poll gets it
    except OSError:
        return []
    return steps


def _max_step(steps: list[dict]) -> int:
    return max((s.get("step_index", -1) for s in steps), default=-1)


def _new_responses(steps: list[dict], baseline: int) -> list[str]:
    """Completed model messages with step_index greater than *baseline*."""
    return [
        s["content"]
        for s in steps
        if s.get("step_index", -1) > baseline
        and s.get("source") == "MODEL"
        and s.get("type") == "PLANNER_RESPONSE"
        and s.get("status") == "DONE"
        and s.get("content")
    ]


# ---------------------------------------------------------------------------
# agy Session (one live agy process inside a tmux session)
# ---------------------------------------------------------------------------

# Serializes spawn//clear windows so a newly created brain dir is always
# attributable to exactly one session.
_SPAWN_LOCK = asyncio.Lock()


@dataclass
class AgySession:
    """Manages a single live agy process hosted in a detached tmux session."""

    name: str
    _conversation_id: str | None = field(default=None, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _turn_count: int = field(default=0, init=False)
    # agy keeps cross-conversation memory per project (working directory):
    # new conversations receive summaries of previous ones and the agent can
    # read their transcripts. Every spawn therefore gets a fresh generation
    # directory, which is what makes clear/reset a true context wipe.
    _generation: int = field(default=0, init=False)

    # --- Identity ----------------------------------------------------------

    @property
    def tmux_session(self) -> str:
        return f"agy-{self.name}"

    @property
    def _target(self) -> str:
        # '=' forces exact-name matching; the trailing ':' makes the target
        # parse as a session for pane-taking commands (capture-pane etc.).
        return f"={self.tmux_session}:"

    @property
    def cwd(self) -> FsPath:
        return SESSIONS_ROOT / self.name / f"c{self._generation}"

    # --- Lifecycle ---------------------------------------------------------

    def _build_command(self) -> str:
        parts = [AGY_CMD]
        if AGY_SKIP_PERMISSIONS:
            parts.append("--dangerously-skip-permissions")
        if AGY_EXTRA_ARGS:
            parts.extend(AGY_EXTRA_ARGS.split())
        return shlex.join(parts)

    async def start(self) -> None:
        async with self._lock:
            if await self.is_alive():
                logger.info("Session '%s' already running", self.name)
                return
            await self._spawn()

    async def _spawn(self) -> None:
        self._generation += 1  # fresh project dir → no inherited conversation memory
        cmd = self._build_command()
        self.cwd.mkdir(parents=True, exist_ok=True)
        logger.info("Spawning '%s' in tmux session %s (cwd=%s)", cmd, self.tmux_session, self.cwd)

        await _tmux("kill-session", "-t", self._target)  # clear leftovers, ignore rc
        rc, out = await _tmux(
            "new-session", "-d",
            "-s", self.tmux_session,
            "-x", str(TERM_WIDTH), "-y", str(TERM_HEIGHT),
            "-c", str(self.cwd),
            cmd,
        )
        if rc != 0:
            raise RuntimeError(f"tmux new-session failed: {out.strip()}")

        try:
            await self._wait_ready()
        except Exception:
            await _tmux("kill-session", "-t", self._target)  # don't leave a zombie
            raise

        # agy creates its conversation lazily (on the first message or on
        # /clear), so the id is resolved during the first send().
        self._conversation_id = None
        self._turn_count = 0
        logger.info("Session '%s' ready", self.name)

    async def _wait_ready(self) -> None:
        """Poll the rendered screen until agy shows its idle input prompt."""
        start = time.monotonic()
        while time.monotonic() - start < STARTUP_TIMEOUT:
            if not await self.is_alive():
                raise RuntimeError(
                    "agy exited during startup. Check that it is installed and "
                    "authenticated (run 'agy' interactively once)."
                )
            screen = await self._capture()
            if READY_MARKER in screen and not self._is_busy(screen):
                logger.info(
                    "Session '%s' ready marker after %.1fs",
                    self.name, time.monotonic() - start,
                )
                return
            await asyncio.sleep(RESPONSE_POLL_INTERVAL)
        raise RuntimeError(f"agy startup timed out after {STARTUP_TIMEOUT:.0f}s")

    async def _detect_new_conversation(self, before: set[str]) -> str:
        """Wait for the new brain dir agy populates with this turn's transcript.

        agy also creates *empty* placeholder brain dirs that never receive a
        transcript. Selecting purely on "newest new dir" can latch onto one of
        those, which is silent but fatal: every transcript read then returns
        nothing, so the turn only ends at the max timeout reporting
        "0 message(s)" while the TUI showed the reply within seconds. Keying on
        the transcript file — the artifact we actually poll — sidesteps the
        trap; if several real conversations appear, the newest transcript wins.
        """
        start = time.monotonic()
        while time.monotonic() - start < CONVERSATION_DETECT_TIMEOUT:
            candidates: dict[str, float] = {}
            for cid in _brain_dirs():
                if cid in before:
                    continue
                try:
                    candidates[cid] = _transcript_path(cid).stat().st_mtime
                except OSError:
                    continue  # dir exists but no transcript yet — keep waiting
            if candidates:
                return max(candidates, key=candidates.get)  # newest transcript wins
            await asyncio.sleep(0.5)
        raise RuntimeError(
            "Could not determine agy conversation id (no new transcript appeared)"
        )

    async def stop(self) -> None:
        async with self._lock:
            await self._kill()

    async def _kill(self) -> None:
        rc, _ = await _tmux("kill-session", "-t", self._target)
        if rc == 0:
            logger.info("Killed tmux session %s", self.tmux_session)
        self._conversation_id = None
        self._turn_count = 0

    async def reset(self) -> None:
        """Kill and re-spawn the process (new conversation)."""
        async with self._lock:
            await self._kill()
            await self._spawn()

    async def clear(self) -> None:
        """Wipe conversation context by respawning into a fresh project dir.

        agy's own /clear is NOT enough: it starts a new conversation, but the
        new one receives summaries of the project's previous conversations
        and the agent can (and does) read their transcripts to recover
        "cleared" context. A respawn in a new generation directory is the
        only true wipe.
        """
        async with self._lock:
            if not await self.is_alive():
                raise RuntimeError("agy process is not running")
            logger.info("Session '%s': clearing (respawn, fresh project dir)", self.name)
            await self._kill()
            await self._spawn()

    # --- Screen access -----------------------------------------------------

    async def _capture(self) -> str:
        rc, out = await _tmux("capture-pane", "-p", "-t", self._target)
        if rc != 0:
            raise RuntimeError("agy process died (tmux capture-pane failed)")
        return out

    @staticmethod
    def _is_busy(screen: str) -> bool:
        return any(m in screen for m in BUSY_MARKERS)

    # --- Chat ----------------------------------------------------------------

    async def send(self, prompt: str) -> str:
        """Send a prompt and return agy's response.

        Input goes in as a tmux bracketed paste, so newlines and TUI shortcut
        characters (!, @, /, backticks...) arrive as literal text. The
        response is read from the conversation transcript: completion means
        a new DONE model step exists AND the rendered screen is idle.
        """
        async with self._lock:
            if not await self.is_alive():
                raise RuntimeError("agy process is not running")

            self._turn_count += 1
            logger.info(
                "Session '%s' turn %d — sending prompt (%d chars)",
                self.name, self._turn_count, len(prompt),
            )

            if self._conversation_id is None:
                # First message: agy creates the conversation when it receives
                # it. Submit under the spawn lock so the new brain dir is
                # attributable to this session, then resolve the id.
                async with _SPAWN_LOCK:
                    before = set(_brain_dirs())
                    await self._submit(prompt)
                    self._conversation_id = await self._detect_new_conversation(before)
                logger.info(
                    "Session '%s': resolved conversation id %s",
                    self.name, self._conversation_id,
                )
                baseline = -1
            else:
                baseline = _max_step(_read_transcript(self._conversation_id))
                await self._submit_confirmed(prompt, baseline)

            response = await self._collect_response(baseline)
            logger.info(
                "Session '%s' turn %d — response collected (%d chars)",
                self.name, self._turn_count, len(response),
            )
            return response

    async def _submit(self, prompt: str) -> None:
        """Paste the prompt (bracketed), then submit with Enter."""
        buf = f"agyrest-{self.name}"
        rc, out = await _tmux(
            "load-buffer", "-b", buf, "-", stdin_data=prompt.encode()
        )
        if rc != 0:
            raise RuntimeError(f"tmux load-buffer failed: {out.strip()}")
        rc, out = await _tmux(
            "paste-buffer", "-p", "-d", "-b", buf, "-t", self._target
        )
        if rc != 0:
            raise RuntimeError(f"tmux paste-buffer failed: {out.strip()}")
        await asyncio.sleep(0.15)
        await _tmux("send-keys", "-t", self._target, "Enter")

    async def _submit_confirmed(self, prompt: str, baseline: int) -> None:
        """Submit *prompt* and make sure agy actually ingested it.

        agy occasionally swallows the submit Enter when its TUI is still
        settling after rendering the previous (often long) response: the pasted
        text sits in the input box unsubmitted while the screen still looks
        idle, so the turn stalls until the max timeout and the bridge reports
        "0 message(s)" even though agy was free the whole time. We confirm
        ingestion before trusting the submit, and only re-press Enter when agy
        is plainly idle and has ingested nothing. We never re-paste and we wait
        a full SUBMIT_CONFIRM_WAIT before each retry, so a slow-but-accepted
        submit can never be turned into a duplicate message.
        """
        await self._submit(prompt)
        for attempt in range(1, SUBMIT_MAX_RETRIES + 1):
            if await self._await_ingest(baseline):
                return
            logger.warning(
                "Session '%s': submit unacknowledged after %.0fs (agy idle, "
                "nothing ingested) — re-pressing Enter [retry %d/%d]",
                self.name, SUBMIT_CONFIRM_WAIT, attempt, SUBMIT_MAX_RETRIES,
            )
            await _tmux("send-keys", "-t", self._target, "Enter")
        # Final grace wait. If agy still took nothing, fall through and let
        # _collect_response run its course (it times out exactly as today).
        await self._await_ingest(baseline)

    async def _await_ingest(self, baseline: int) -> bool:
        """Wait up to SUBMIT_CONFIRM_WAIT for agy to ingest the submitted turn.

        Ingested == a new transcript step (step_index > baseline) appears — agy
        logged the turn — OR the screen shows a busy marker — agy is
        generating. Either way the submit took, so returning here can never
        cause a duplicate. Returns False only if the whole window elapses with
        agy idle and nothing new ingested: the signature of a dropped Enter,
        where re-pressing it is safe because nothing was ever submitted.
        """
        deadline = time.monotonic() + SUBMIT_CONFIRM_WAIT
        while time.monotonic() < deadline:
            await asyncio.sleep(RESPONSE_POLL_INTERVAL)
            if _max_step(_read_transcript(self._conversation_id)) > baseline:
                return True
            if self._is_busy(await self._capture()):
                return True
        return False

    def _transcript_mtime(self) -> float:
        try:
            return _transcript_path(self._conversation_id).stat().st_mtime
        except OSError:
            return 0.0

    async def _collect_response(self, baseline: int) -> str:
        """Poll until the turn is complete, then return the new model text.

        A turn is complete when a new DONE model response exists in the
        transcript AND the rendered screen is idle (covers multi-step turns
        with tool calls), for two consecutive polls (debounce).

        We do NOT cut a turn off on a flat wall-clock: agy can legitimately
        work for minutes (reading files, running commands, streaming a long
        answer). Instead we track progress — a busy screen, or the transcript
        growing/being rewritten — and only give up once agy has made no
        progress for RESPONSE_STALL_TIMEOUT, or RESPONSE_HARD_TIMEOUT absolute.
        """
        start = time.monotonic()
        responses: list[str] = []
        confirm = 0
        exit_reason = "unknown"
        last_progress = start
        last_max_step = baseline
        last_mtime = self._transcript_mtime()

        while True:
            now = time.monotonic()
            elapsed = now - start
            if elapsed > RESPONSE_HARD_TIMEOUT:
                exit_reason = "hard_timeout"
                logger.warning(
                    "Session '%s': hit hard timeout (%.0fs) — returning %d partial "
                    "response(s)", self.name, RESPONSE_HARD_TIMEOUT, len(responses),
                )
                break
            if now - last_progress > RESPONSE_STALL_TIMEOUT:
                exit_reason = "stalled"
                logger.warning(
                    "Session '%s': no progress for %.0fs (idle, transcript not "
                    "growing) — returning %d response(s)",
                    self.name, RESPONSE_STALL_TIMEOUT, len(responses),
                )
                break

            await asyncio.sleep(RESPONSE_POLL_INTERVAL)

            steps = _read_transcript(self._conversation_id)
            responses = _new_responses(steps, baseline)
            cur_max_step = _max_step(steps)
            cur_mtime = self._transcript_mtime()
            screen = await self._capture()  # raises if process died
            busy = self._is_busy(screen)
            idle = READY_MARKER in screen and not busy

            # agy is demonstrably alive and working if it's generating on
            # screen or the transcript advanced/was rewritten since last poll.
            if busy or cur_max_step > last_max_step or cur_mtime > last_mtime:
                last_progress = now
                last_max_step = cur_max_step
                last_mtime = cur_mtime

            if responses and idle and elapsed >= RESPONSE_MIN_WAIT:
                confirm += 1
                if confirm >= 2:
                    exit_reason = "transcript_done"
                    break
            else:
                confirm = 0

        total = time.monotonic() - start
        logger.info(
            "Session '%s': response complete via %s (%.1fs, %d message(s))",
            self.name, exit_reason, total, len(responses),
        )
        if exit_reason != "transcript_done":
            # The failure class we keep having to autopsy by hand — capture a
            # local snapshot (screen + transcript tail) for later investigation.
            await self._dump_timeout_diagnostic(exit_reason, baseline, total, responses)
        return "\n\n".join(responses).strip()

    async def _dump_timeout_diagnostic(
        self, reason: str, baseline: int, elapsed: float, responses: list[str]
    ) -> None:
        """Write a self-contained snapshot of a non-clean turn end to LOG_DIR.

        Records what an after-the-fact investigation needs: the resolved
        conversation id and transcript path, the tail of the transcript (step
        types/statuses), and the rendered screen at the moment we gave up.
        Best-effort — diagnostics must never mask the original turn result.
        """
        try:
            screen = await self._capture()
        except Exception as exc:  # pragma: no cover - capture already failed once
            screen = f"(screen capture failed: {exc})"
        steps = _read_transcript(self._conversation_id)
        tail = steps[-TIMEOUT_DUMP_STEPS:]
        path = TIMEOUT_LOG_DIR / f"{self.name}-turn{self._turn_count}-{self._conversation_id}.log"
        try:
            TIMEOUT_LOG_DIR.mkdir(parents=True, exist_ok=True)
            lines = [
                f"session={self.name} turn={self._turn_count} reason={reason}",
                f"conversation_id={self._conversation_id}",
                f"transcript={_transcript_path(self._conversation_id)}",
                f"baseline={baseline} elapsed={elapsed:.1f}s "
                f"collected_messages={len(responses)} total_steps={len(steps)}",
                f"\n=== last {len(tail)} transcript steps ===",
            ]
            for d in tail:
                c = d.get("content")
                clen = len(c) if isinstance(c, str) else c
                lines.append(
                    f"step {d.get('step_index')} | {d.get('source')} "
                    f"{d.get('type')} {d.get('status')} | "
                    f"{d.get('created_at')} | content_len={clen}"
                )
            lines.append("\n=== rendered screen when we gave up ===")
            lines.append(screen)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            logger.warning("Session '%s': wrote timeout diagnostic to %s", self.name, path)
        except OSError as exc:
            logger.warning(
                "Session '%s': could not write timeout diagnostic: %s", self.name, exc
            )

    # --- Status --------------------------------------------------------------

    async def is_alive(self) -> bool:
        rc, _ = await _tmux("has-session", "-t", self._target)
        return rc == 0

    @property
    def turn_count(self) -> int:
        return self._turn_count


# ---------------------------------------------------------------------------
# Session Manager
# ---------------------------------------------------------------------------

@dataclass
class ChatManager:
    """Manages named chat sessions, each backed by its own AgySession."""

    _sessions: dict[str, AgySession] = field(default_factory=dict, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def get_or_create(self, name: str) -> AgySession:
        """Return an existing session or create and start a new one."""
        async with self._lock:
            if name in self._sessions:
                session = self._sessions[name]
            else:
                logger.info("Creating new session: '%s'", name)
                session = AgySession(name)
                self._sessions[name] = session

        # Start (or restart a dead session) outside the manager lock so other
        # sessions aren't blocked during the ~25s startup. start() is a no-op
        # when the process is already alive.
        await session.start()
        return session

    async def get(self, name: str) -> AgySession | None:
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
        result: dict[str, dict] = {}
        for name, s in snapshot.items():
            result[name] = {"alive": await s.is_alive(), "turn_count": s.turn_count}
        return result


manager = ChatManager()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Sessions are created lazily. On shutdown, stop all."""
    await _ensure_tmux_server()
    logger.info("Server starting (sessions will be created on demand)")
    yield
    logger.info("Shutting down — stopping all sessions...")
    count = await manager.stop_all()
    logger.info("Stopped %d session(s)", count)


app = FastAPI(
    title="Antigravity CLI REST Bridge",
    description=(
        "REST API that wraps Google's Antigravity CLI (agy) interactive mode, "
        "providing named multi-session chat with conversation continuity."
    ),
    version="3.0.0",
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
    if not await session.is_alive():
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
