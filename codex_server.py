"""
REST API server that wraps OpenAI's Codex CLI (codex) interactive mode.

Each named session (/chat/{name}) runs its own live `codex` process inside a
detached tmux session — the same architecture as the agy bridge (server.py).
tmux acts as a terminal-emulator mediator: instead of parsing the raw PTY
escape-code stream, we read the *rendered* screen with `tmux capture-pane`
(used only for startup-ready detection, interstitial handling, and liveness)
and extract the model's response as structured data from codex's per-session
rollout file (~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl).

Why a warm process instead of `codex exec` per turn: a live interactive session
keeps conversation context in-process across turns and mirrors the agy bridge
one-to-one, so a single skill drives both. Context wipe (/clear) respawns into a
fresh working directory, exactly like agy.

Codex's rollout makes response reading cleaner than agy's: it records explicit
turn boundaries. A turn is `event_msg/task_started{turn_id}` ... a sequence of
`response_item`s ... `event_msg/task_complete{turn_id, last_agent_message}`. The
arrival of a NEW task_complete is a definitive done-signal (no screen-scrape
debounce), and `last_agent_message` is the final answer text.

The REST contract is identical to the agy bridge (server.py).
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

from fastapi import FastAPI, HTTPException, Path, Query, Request
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
CODEX_CMD = os.getenv("CODEX_CMD", "codex")
# Space-separated extra args appended to every codex invocation, e.g.
# "--add-dir /repos" to grant read/write access to mounted repositories.
CODEX_EXTRA_ARGS = os.getenv("CODEX_EXTRA_ARGS", "")

TMUX_BIN = os.getenv("TMUX_BIN", "tmux")
# Dedicated tmux server socket, distinct from the agy bridge's ("agy-rest"), so
# the two bridges in one container never collide and neither touches a user tmux.
TMUX_SOCKET = os.getenv("CODEX_TMUX_SOCKET", "codex-rest")
TERM_WIDTH = int(os.getenv("TERM_WIDTH", "200"))
TERM_HEIGHT = int(os.getenv("TERM_HEIGHT", "50"))

# Where codex keeps its state. Sessions (rollout JSONL) live under
# CODEX_HOME/sessions/YYYY/MM/DD/. CODEX_HOME mirrors codex's own env var.
CODEX_HOME = FsPath(os.getenv("CODEX_HOME", os.path.expanduser("~/.codex")))
CODEX_SESSIONS_DIR = FsPath(os.getenv("CODEX_SESSIONS_DIR", str(CODEX_HOME / "sessions")))

# Each session gets its own working directory so codex's per-cwd state never
# crosses between sessions. The per-run id keeps paths unique across server
# restarts: a fresh path guarantees a clean slate for a "new" session.
RUN_ID = uuid.uuid4().hex[:8]
SESSIONS_ROOT = FsPath(os.getenv("SESSIONS_ROOT", "/tmp/codex-rest-sessions")) / RUN_ID

# Local, persisted log destination (mounted to the host in docker-compose), so
# turns can be investigated after the fact without docker-exec'ing a container.
LOG_DIR = FsPath(os.getenv("LOG_DIR", "/app/logs"))
TIMEOUT_LOG_DIR = LOG_DIR / "timeouts"
TIMEOUT_DUMP_STEPS = int(os.getenv("TIMEOUT_DUMP_STEPS", "40"))

STARTUP_TIMEOUT = float(os.getenv("CODEX_STARTUP_TIMEOUT", "60"))
RESPONSE_POLL_INTERVAL = float(os.getenv("CODEX_RESPONSE_POLL_INTERVAL", "0.5"))
RESPONSE_MIN_WAIT = float(os.getenv("CODEX_RESPONSE_MIN_WAIT", "1"))
# A turn ends when codex finishes (a new task_complete) — but we also bound it.
# A turn that makes NO progress for RESPONSE_STALL_TIMEOUT is cut off early; any
# turn is HARD-CAPPED at RESPONSE_HARD_TIMEOUT so it never blocks the client
# longer than the per-request cap (3 min for both bridges).
RESPONSE_STALL_TIMEOUT = float(os.getenv("CODEX_RESPONSE_STALL_TIMEOUT", "90"))
RESPONSE_HARD_TIMEOUT = float(
    # honor a legacy CODEX_EXEC_TIMEOUT as the hard ceiling if someone set it
    os.getenv("CODEX_RESPONSE_HARD_TIMEOUT", os.getenv("CODEX_EXEC_TIMEOUT", "180"))
)
# Any turn slower than this gets a diagnostic dump even if it SUCCEEDED, so a
# slow-but-fine turn's rollout tail (timestamped events — where the time went)
# is captured, not just outright timeouts.
RESPONSE_SLOW_DUMP_SECS = float(os.getenv("CODEX_SLOW_DUMP_SECS", "90"))
# After submitting, how long to wait for codex to *acknowledge* the turn (a new
# task_started appears, or the screen goes busy) before concluding the submit
# Enter was dropped and re-pressing it. A genuinely-accepted submit acknowledges
# within ~1s, so waiting this long means a merely-slow-but-accepted submit is
# never mistaken for a drop and re-sent — which would duplicate the message.
SUBMIT_CONFIRM_WAIT = float(os.getenv("CODEX_SUBMIT_CONFIRM_WAIT", "8"))
SUBMIT_MAX_RETRIES = int(os.getenv("CODEX_SUBMIT_MAX_RETRIES", "2"))
# How long after startup we wait for codex to register the new session (its
# rollout file, tagged with our cwd, appears).
SESSION_DETECT_TIMEOUT = float(os.getenv("CODEX_SESSION_DETECT_TIMEOUT", "20"))

# /last read-back: cap how long a single /last?wait=N call may block, so it
# never holds the client longer than a /chat would (same hard ceiling).
LAST_MAX_WAIT = float(os.getenv("CODEX_LAST_MAX_WAIT", str(RESPONSE_HARD_TIMEOUT)))

# Rendered-screen markers (read from tmux's emulated screen, never from the raw
# escape stream, so these are stable plain-text strings). Correctness rides on
# the rollout (task_started/task_complete); the screen is only used for startup
# readiness, dismissing startup interstitials, and liveness.
# Idle-prompt markers — ANY present means the TUI reached its input prompt.
# Kept version-tolerant: 0.139 showed a "Context N% used" meter in the idle
# status line, but 0.141 dropped it; the welcome banner ("OpenAI Codex" /
# "/model to change") is stable across versions and absent on the trust/update
# interstitial screens, so it cleanly signals "past startup, at the prompt".
READY_MARKERS = ("/model to change", "OpenAI Codex", "% used")
# Best-effort busy hints (secondary to the rollout's in-flight detection).
BUSY_MARKERS = ("esc to interrupt", "ctrl+c to interrupt", "working", "thinking")

# tmux client commands are sub-second; anything longer means a stuck client.
TMUX_CMD_TIMEOUT = float(os.getenv("TMUX_CMD_TIMEOUT", "15"))

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


# --- Audit-trail helpers (used by /last logging) ----------------------------

def _client(request: "Request") -> str:
    """Best-effort caller identity for the log: the originating IP, honoring a
    reverse proxy's X-Forwarded-For when present, else the socket peer."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "?"


def _ua(request: "Request") -> str:
    """Caller's User-Agent (which client/tool issued the recovery), truncated."""
    return (request.headers.get("user-agent") or "?")[:120]


def _preview(text: str, limit: int = 160) -> str:
    """One-line, whitespace-collapsed snippet of an answer for the trail — so
    the log records WHAT was handed back without dumping the whole response."""
    flat = " ".join((text or "").split())
    return flat[:limit] + ("…" if len(flat) > limit else "")


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
# codex rollout access
# ---------------------------------------------------------------------------
#
# An interactive codex session appends every event to a single rollout file for
# the life of the process. The shapes we depend on (verified against codex-cli
# 0.139.0):
#   {"type":"session_meta","payload":{"id":"<uuid>","cwd":"...","originator":"codex-tui",...}}
#   {"type":"event_msg","payload":{"type":"task_started","turn_id":"..."}}
#   {"type":"response_item","payload":{"type":"message","role":"assistant",
#                                      "content":[{"type":"output_text","text":"..."}]}}
#   {"type":"event_msg","payload":{"type":"task_complete","turn_id":"...",
#                                  "last_agent_message":"<final answer or null>"}}


def _rollout_files() -> list[FsPath]:
    """All rollout JSONL files codex has written, cheapest-first by nothing."""
    if not CODEX_SESSIONS_DIR.is_dir():
        return []
    try:
        return list(CODEX_SESSIONS_DIR.rglob("rollout-*.jsonl"))
    except OSError:
        return []


def _payload(ev: dict) -> dict:
    p = ev.get("payload")
    return p if isinstance(p, dict) else {}


def _rollout_meta(path: FsPath) -> dict | None:
    """Read just the session_meta (first line) of a rollout file, if present."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ev = json.loads(line)
                if ev.get("type") == "session_meta":
                    return _payload(ev)
                return None  # first event isn't session_meta — unexpected; bail
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _read_rollout(path: FsPath | None) -> list[dict]:
    """Parse a rollout JSONL into a list of event dicts (empty if absent)."""
    if path is None or not path.is_file():
        return []
    events: list[dict] = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # partially-written line; next poll gets it
    except OSError:
        return []
    return events


def _count_task_completes(events: list[dict]) -> int:
    return sum(
        1 for ev in events
        if ev.get("type") == "event_msg" and _payload(ev).get("type") == "task_complete"
    )


def _count_task_starts(events: list[dict]) -> int:
    return sum(
        1 for ev in events
        if ev.get("type") == "event_msg" and _payload(ev).get("type") == "task_started"
    )


def _answer_for_new_turn(events: list[dict], baseline_completes: int) -> str:
    """Final model text for the turn(s) completed since *baseline_completes*.

    Prefer the latest task_complete's `last_agent_message` (codex's canonical
    final answer). Fall back to concatenating assistant `output_text` produced
    after the baseline turn boundary, so a null last_agent_message still yields
    the visible reply.
    """
    # Find the position just after the baseline_completes-th task_complete; the
    # new turn's events all live beyond it. baseline 0 -> scan from the start.
    completes = 0
    start = 0
    for i, ev in enumerate(events):
        if ev.get("type") == "event_msg" and _payload(ev).get("type") == "task_complete":
            completes += 1
            if completes == baseline_completes:
                start = i + 1
                break
    window = events[start:]

    answer = ""
    for ev in window:
        if ev.get("type") == "event_msg" and _payload(ev).get("type") == "task_complete":
            msg = _payload(ev).get("last_agent_message")
            if msg:
                answer = msg
    if answer:
        return answer.strip()

    texts: list[str] = []
    for ev in window:
        p = _payload(ev)
        if ev.get("type") == "response_item" and p.get("type") == "message" \
                and p.get("role") == "assistant":
            for part in p.get("content", []) or []:
                if isinstance(part, dict) and part.get("type") == "output_text" \
                        and part.get("text"):
                    texts.append(part["text"])
    return "\n".join(texts).strip()


# ---------------------------------------------------------------------------
# Codex Session (one live codex process inside a tmux session)
# ---------------------------------------------------------------------------

# Serializes spawn/clear windows so a newly created rollout file is always
# attributable to exactly one session.
_SPAWN_LOCK = asyncio.Lock()


@dataclass
class CodexSession:
    """Manages a single live codex process hosted in a detached tmux session."""

    name: str
    _rollout_path: FsPath | None = field(default=None, init=False)
    _session_id: str | None = field(default=None, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _turn_count: int = field(default=0, init=False)
    # Completed-turn count captured when the most recent turn was submitted.
    # /last uses it to tell THIS turn's answer apart from a previous one: a new
    # task_complete beyond this baseline means your submitted turn has finished.
    _last_baseline_completes: int = field(default=0, init=False)
    # Every spawn gets a fresh generation directory (and a brand-new codex
    # session), which is what makes clear/reset a true context wipe.
    _generation: int = field(default=0, init=False)

    # --- Identity ----------------------------------------------------------

    @property
    def tmux_session(self) -> str:
        return f"codex-{self.name}"

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
        # --dangerously-bypass-approvals-and-sandbox: never block on an approval
        #   or the "do you trust this directory?" prompt (the container is the
        #   sandbox; mirrors agy's "always-proceed"). Belt-and-suspenders with
        #   approval_policy=never + sandbox_mode=danger-full-access in config.toml.
        parts = [CODEX_CMD, "--dangerously-bypass-approvals-and-sandbox"]
        if CODEX_EXTRA_ARGS:
            parts.extend(shlex.split(CODEX_EXTRA_ARGS))
        return shlex.join(parts)

    async def start(self) -> None:
        async with self._lock:
            if await self.is_alive():
                logger.info("Session '%s' already running", self.name)
                return
            await self._spawn()

    async def _spawn(self) -> None:
        self._generation += 1  # fresh dir + fresh codex session → clean slate
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

        # codex creates its rollout lazily (on the first turn), so the session
        # id / rollout path are resolved during the first send().
        self._rollout_path = None
        self._session_id = None
        self._turn_count = 0
        self._last_baseline_completes = 0
        logger.info("Session '%s' ready", self.name)

    async def _wait_ready(self) -> None:
        """Poll the rendered screen until codex shows its idle input prompt.

        The codex TUI may gate startup behind interstitials that `codex exec`
        never shows — a trust-directory prompt, an "Update available" prompt, a
        model NUX. The bypass flag preempts the trust prompt, but we still
        dismiss any interstitial defensively so a warm spawn never hangs on one.
        """
        start = time.monotonic()
        while time.monotonic() - start < STARTUP_TIMEOUT:
            if not await self.is_alive():
                raise RuntimeError(
                    "codex exited during startup. Check that it is installed and "
                    "authenticated (run 'codex login' once)."
                )
            screen = await self._capture()
            if any(m in screen for m in READY_MARKERS):
                logger.info(
                    "Session '%s' ready marker after %.1fs",
                    self.name, time.monotonic() - start,
                )
                return
            await self._maybe_dismiss_interstitial(screen)
            await asyncio.sleep(RESPONSE_POLL_INTERVAL)
        raise RuntimeError(f"codex startup timed out after {STARTUP_TIMEOUT:.0f}s")

    async def _maybe_dismiss_interstitial(self, screen: str) -> bool:
        """Clear a known startup interstitial; return True if we acted.

        Conservative by design: the only prompt whose DEFAULT button is
        dangerous is the update prompt ("Update now" runs `npm install`), so we
        never blind-press Enter on it — we send Escape and let the next poll
        re-check. Trust / NUX / tip prompts are safe to confirm with Enter.
        """
        low = screen.lower()
        if "update available" in low or "update now" in low:
            logger.warning(
                "Session '%s': codex update prompt at startup — sending Esc "
                "(NOT Enter; its default runs npm install)", self.name,
            )
            await _tmux("send-keys", "-t", self._target, "Escape")
            return True
        if "do you trust" in low or "trust the contents" in low:
            await _tmux("send-keys", "-t", self._target, "Enter")  # default: Yes
            return True
        if "press enter to continue" in low:  # model NUX, tips, etc.
            await _tmux("send-keys", "-t", self._target, "Enter")
            return True
        return False

    async def _detect_new_session(self, before: set[str]) -> FsPath:
        """Wait for the new rollout file codex tags with this session's cwd.

        codex writes the rollout lazily on the first turn; its session_meta
        records the cwd, so we key on that — the per-run cwd is unique to this
        session, so at most one new rollout can match.
        """
        start = time.monotonic()
        want = str(self.cwd)
        while time.monotonic() - start < SESSION_DETECT_TIMEOUT:
            candidates: dict[FsPath, float] = {}
            for path in _rollout_files():
                if str(path) in before:
                    continue
                meta = _rollout_meta(path)
                if not meta or meta.get("cwd") != want:
                    continue  # not ours, or session_meta not written yet
                try:
                    candidates[path] = path.stat().st_mtime
                except OSError:
                    continue
            if candidates:
                return max(candidates, key=candidates.get)  # newest wins
            await asyncio.sleep(0.5)
        raise RuntimeError(
            "Could not determine codex session (no new rollout for this cwd appeared)"
        )

    async def stop(self) -> None:
        async with self._lock:
            await self._kill()

    async def _kill(self) -> None:
        rc, _ = await _tmux("kill-session", "-t", self._target)
        if rc == 0:
            logger.info("Killed tmux session %s", self.tmux_session)
        self._rollout_path = None
        self._session_id = None
        self._turn_count = 0
        self._last_baseline_completes = 0

    async def reset(self) -> None:
        """Kill and re-spawn the process (new conversation)."""
        async with self._lock:
            await self._kill()
            await self._spawn()

    async def clear(self) -> None:
        """Wipe conversation context by respawning into a fresh working dir.

        A fresh generation directory + a brand-new codex process (hence a new
        rollout/session) is an unambiguous context wipe.
        """
        async with self._lock:
            if not await self.is_alive():
                raise RuntimeError("codex process is not running")
            logger.info("Session '%s': clearing (respawn, fresh working dir)", self.name)
            await self._kill()
            await self._spawn()

    # --- Screen access -----------------------------------------------------

    async def _capture(self) -> str:
        rc, out = await _tmux("capture-pane", "-p", "-t", self._target)
        if rc != 0:
            raise RuntimeError("codex process died (tmux capture-pane failed)")
        return out

    @staticmethod
    def _is_busy(screen: str) -> bool:
        low = screen.lower()
        return any(m in low for m in BUSY_MARKERS)

    # --- Chat ----------------------------------------------------------------

    async def send(self, prompt: str) -> str:
        """Send a prompt and return codex's response.

        Input goes in as a tmux bracketed paste, so newlines and TUI shortcut
        characters (!, @, /, backticks...) arrive as literal text. The response
        is read from the rollout: completion means a new task_complete event
        appeared for this turn.
        """
        async with self._lock:
            if not await self.is_alive():
                raise RuntimeError("codex process is not running")

            self._turn_count += 1
            logger.info(
                "Session '%s' turn %d — sending prompt (%d chars)",
                self.name, self._turn_count, len(prompt),
            )

            if self._rollout_path is None:
                # First message: codex creates the rollout when it receives the
                # turn. Submit under the spawn lock so the new file is
                # attributable to this session, then resolve it.
                async with _SPAWN_LOCK:
                    before = {str(p) for p in _rollout_files()}
                    await self._submit(prompt)
                    self._rollout_path = await self._detect_new_session(before)
                meta = _rollout_meta(self._rollout_path) or {}
                self._session_id = meta.get("id")
                logger.info(
                    "Session '%s': bound to codex session %s (%s)",
                    self.name, self._session_id, self._rollout_path.name,
                )
                baseline_completes = 0
                baseline_starts = _count_task_starts(_read_rollout(self._rollout_path))
                # Record where this turn began so a later /last can recover its
                # answer and never mistake a previous turn's for it.
                self._last_baseline_completes = baseline_completes
            else:
                events = _read_rollout(self._rollout_path)
                baseline_completes = _count_task_completes(events)
                baseline_starts = _count_task_starts(events)
                # Publish the baseline BEFORE submitting, so a /last arriving
                # while this turn runs reads the right turn boundary.
                self._last_baseline_completes = baseline_completes
                await self._submit_confirmed(prompt, baseline_starts)

            response = await self._collect_response(baseline_completes, baseline_starts)
            logger.info(
                "Session '%s' turn %d — response collected (%d chars)",
                self.name, self._turn_count, len(response),
            )
            return response

    async def _submit(self, prompt: str) -> None:
        """Paste the prompt (bracketed), then submit with Enter."""
        buf = f"codexrest-{self.name}"
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

    async def _submit_confirmed(self, prompt: str, baseline_starts: int) -> None:
        """Submit *prompt* and make sure codex actually ingested it.

        A TUI can swallow the submit Enter when it is still settling after a long
        previous response: the pasted text sits unsubmitted while the screen
        looks idle, so the turn stalls. We confirm ingestion before trusting the
        submit, and only re-press Enter when codex is plainly idle and has
        ingested nothing. We never re-paste and we wait a full SUBMIT_CONFIRM_WAIT
        before each retry, so a slow-but-accepted submit can't become a duplicate.
        """
        await self._submit(prompt)
        for attempt in range(1, SUBMIT_MAX_RETRIES + 1):
            if await self._await_ingest(baseline_starts):
                return
            logger.warning(
                "Session '%s': submit unacknowledged after %.0fs (codex idle, "
                "nothing ingested) — re-pressing Enter [retry %d/%d]",
                self.name, SUBMIT_CONFIRM_WAIT, attempt, SUBMIT_MAX_RETRIES,
            )
            await _tmux("send-keys", "-t", self._target, "Enter")
        # Final grace wait. If codex still took nothing, fall through and let
        # _collect_response run its course (it times out exactly as today).
        await self._await_ingest(baseline_starts)

    async def _await_ingest(self, baseline_starts: int) -> bool:
        """Wait up to SUBMIT_CONFIRM_WAIT for codex to ingest the submitted turn.

        Ingested == a new task_started appears (codex began the turn) OR the
        screen shows a busy marker. Either way the submit took, so returning
        here can never cause a duplicate. Returns False only if the whole window
        elapses with codex idle and no new task_started: the signature of a
        dropped Enter, where re-pressing it is safe because nothing was submitted.
        """
        deadline = time.monotonic() + SUBMIT_CONFIRM_WAIT
        while time.monotonic() < deadline:
            await asyncio.sleep(RESPONSE_POLL_INTERVAL)
            if _count_task_starts(_read_rollout(self._rollout_path)) > baseline_starts:
                return True
            if self._is_busy(await self._capture()):
                return True
        return False

    def _rollout_mtime(self) -> float:
        try:
            return self._rollout_path.stat().st_mtime  # type: ignore[union-attr]
        except (OSError, AttributeError):
            return 0.0

    async def _collect_response(self, baseline_completes: int, baseline_starts: int) -> str:
        """Poll until the turn completes, then return the new model text.

        A turn is complete when a NEW task_complete event appears in the rollout
        (codex's definitive end-of-turn signal).

        We do NOT cut a turn off on a flat wall-clock: codex can legitimately
        work for minutes (reading files, running commands, long reasoning). A
        turn that has started but not completed counts as progress (it is
        in-flight), so a long pure-thinking turn — which may write nothing else
        for ~90s — is never mistaken for a stall. We give up only after
        RESPONSE_STALL_TIMEOUT of genuine no-progress, or RESPONSE_HARD_TIMEOUT
        absolute.
        """
        start = time.monotonic()
        response = ""
        exit_reason = "unknown"
        last_progress = start
        last_mtime = self._rollout_mtime()
        last_starts = baseline_starts

        while True:
            now = time.monotonic()
            elapsed = now - start
            if elapsed > RESPONSE_HARD_TIMEOUT:
                exit_reason = "hard_timeout"
                logger.warning(
                    "Session '%s': hit hard timeout (%.0fs) — returning partial "
                    "(%d chars)", self.name, RESPONSE_HARD_TIMEOUT, len(response),
                )
                break
            if now - last_progress > RESPONSE_STALL_TIMEOUT:
                exit_reason = "stalled"
                logger.warning(
                    "Session '%s': no progress for %.0fs (idle, rollout not "
                    "growing) — returning %d chars", self.name,
                    RESPONSE_STALL_TIMEOUT, len(response),
                )
                break

            await asyncio.sleep(RESPONSE_POLL_INTERVAL)

            events = _read_rollout(self._rollout_path)
            cur_completes = _count_task_completes(events)
            cur_starts = _count_task_starts(events)
            cur_mtime = self._rollout_mtime()
            await self._capture()  # raises if the codex process died

            # codex is demonstrably alive and working if the rollout grew, a new
            # turn started, or a started turn has not yet completed (in-flight).
            in_flight = cur_starts > baseline_starts and cur_completes <= baseline_completes
            if cur_mtime > last_mtime or cur_starts > last_starts or in_flight:
                last_progress = now
                last_mtime = cur_mtime
                last_starts = cur_starts

            if cur_completes > baseline_completes and elapsed >= RESPONSE_MIN_WAIT:
                response = _answer_for_new_turn(events, baseline_completes)
                exit_reason = "rollout_done"
                break

        total = time.monotonic() - start
        logger.info(
            "Session '%s': response complete via %s (%.1fs, %d chars)",
            self.name, exit_reason, total, len(response),
        )
        if exit_reason != "rollout_done" or total > RESPONSE_SLOW_DUMP_SECS:
            # Snapshot screen + rollout tail for later investigation — on any
            # timeout, and on a slow-but-successful turn so we can see WHERE the
            # time went (the rollout events are timestamped).
            dump_reason = exit_reason if exit_reason != "rollout_done" else "slow_success"
            await self._dump_diagnostic(dump_reason, baseline_completes, total, response)
        return response

    async def _dump_diagnostic(
        self, reason: str, baseline: int, elapsed: float, response: str
    ) -> None:
        """Write a self-contained snapshot of a non-clean/slow turn to LOG_DIR.

        Records what an after-the-fact investigation needs: the resolved session
        id and rollout path, the tail of the rollout (event types/turn ids), and
        the rendered screen at the moment we gave up. Best-effort — diagnostics
        must never mask the original turn result.
        """
        try:
            screen = await self._capture()
        except Exception as exc:  # pragma: no cover - capture already failed once
            screen = f"(screen capture failed: {exc})"
        events = _read_rollout(self._rollout_path)
        tail = events[-TIMEOUT_DUMP_STEPS:]
        rollout_name = self._rollout_path.name if self._rollout_path else "unresolved"
        path = TIMEOUT_LOG_DIR / f"codex-{self.name}-turn{self._turn_count}-{rollout_name}.log"
        try:
            TIMEOUT_LOG_DIR.mkdir(parents=True, exist_ok=True)
            lines = [
                f"session={self.name} turn={self._turn_count} reason={reason}",
                f"session_id={self._session_id}",
                f"rollout={self._rollout_path}",
                f"baseline_completes={baseline} elapsed={elapsed:.1f}s "
                f"response_len={len(response)} total_events={len(events)}",
                f"\n=== last {len(tail)} rollout events ===",
            ]
            for ev in tail:
                p = _payload(ev)
                lines.append(
                    f"{ev.get('type')}/{p.get('type')} | turn={p.get('turn_id')} | "
                    f"ts={ev.get('timestamp')}"
                )
            lines.append("\n=== rendered screen when we gave up ===")
            lines.append(screen)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            logger.warning("Session '%s': wrote diagnostic to %s", self.name, path)
        except OSError as exc:
            logger.warning(
                "Session '%s': could not write diagnostic: %s", self.name, exc
            )

    # --- Read-back -----------------------------------------------------------

    async def last(self, wait: float) -> tuple[bool, str, int]:
        """Re-read the most recent turn's final answer — only once it is done.

        Read-only and LOCK-FREE: it inspects the rollout file (never the tmux
        pane), so it works WHILE a turn is still in flight. The point is to
        recover an answer a /chat call never received — a 3-min hard cap, or a
        connectivity blip after codex had already finished — WITHOUT re-asking.
        The answer is durable in the rollout regardless of whether the HTTP
        response arrived.

        Returns (done, answer, turn). done=True only after a NEW task_complete
        appears beyond the baseline captured when this turn was submitted; until
        then done=False with no text, so a still-running or never-ingested turn
        never yields a partial answer or a stale previous one. With wait>0 it
        polls up to `wait` seconds for an in-flight turn to finish.
        """
        deadline = time.monotonic() + max(0.0, wait)
        while True:
            path = self._rollout_path
            baseline = self._last_baseline_completes
            if path is not None:
                events = _read_rollout(path)
                if _count_task_completes(events) > baseline:
                    return True, _answer_for_new_turn(events, baseline), self._turn_count
            if time.monotonic() >= deadline:
                return False, "", self._turn_count
            await asyncio.sleep(RESPONSE_POLL_INTERVAL)

    # --- Status --------------------------------------------------------------

    async def is_alive(self) -> bool:
        rc, _ = await _tmux("has-session", "-t", self._target)
        return rc == 0

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def ref(self) -> str:
        """Durable artifact this session reads answers from — the codex session
        id (rollout). Logged on /last so a recovered answer can be traced back
        to the exact rollout it came from ('-' before the first turn)."""
        return self._session_id or "-"


# ---------------------------------------------------------------------------
# Session Manager
# ---------------------------------------------------------------------------

@dataclass
class ChatManager:
    """Manages named chat sessions, each backed by its own CodexSession."""

    _sessions: dict[str, CodexSession] = field(default_factory=dict, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def get_or_create(self, name: str) -> CodexSession:
        """Return an existing session or create and start a new one."""
        async with self._lock:
            if name in self._sessions:
                session = self._sessions[name]
            else:
                logger.info("Creating new session: '%s'", name)
                session = CodexSession(name)
                self._sessions[name] = session

        # Start (or restart a dead session) outside the manager lock so other
        # sessions aren't blocked during startup. start() is a no-op when the
        # process is already alive.
        await session.start()
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
    title="Codex CLI REST Bridge",
    description=(
        "REST API that wraps OpenAI's Codex CLI (codex) interactive mode, "
        "providing named multi-session chat with conversation continuity. "
        "Mirrors the agy bridge's tmux architecture and REST contract."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def access_log(request: Request, call_next):
    """Log every request to the persisted server log for post-mortem debugging.

    Writes one line when a request ARRIVES and one when it FINISHES, sharing a
    short id. This is what makes the three failure modes investigable from the
    file log alone:
      * stuck  — an arrival line ('-->') with no matching completion ('<--')
                 is a request still hung in its handler (e.g. a wedged turn).
      * break  — an unhandled error logs a full traceback ('!!!') and returns
                 500 instead of vanishing; HTTPExceptions show as their status.
      * slow   — every completion carries its wall-clock duration in ms.
    Health checks log at DEBUG so routine polling never floods the file.
    """
    rid = uuid.uuid4().hex[:8]
    level = logging.DEBUG if request.url.path == "/health" else logging.INFO
    t0 = time.monotonic()
    logger.log(level, "--> %s %s [%s]", request.method, request.url.path, rid)
    try:
        response = await call_next(request)
    except Exception:
        elapsed = int((time.monotonic() - t0) * 1000)
        logger.exception(
            "!!! %s %s [%s] unhandled error after %dms",
            request.method, request.url.path, rid, elapsed,
        )
        raise
    elapsed = int((time.monotonic() - t0) * 1000)
    logger.log(
        level, "<-- %s %s [%s] %d in %dms",
        request.method, request.url.path, rid, response.status_code, elapsed,
    )
    return response


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

class LastResponse(BaseModel):
    done: bool          # True only if the most recent turn has FINISHED
    response: str | None  # the answer when done; null while still running
    turn: int
    session: str
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


@app.get("/last/{name}", response_model=LastResponse)
async def get_last(
    request: Request,
    name: str = _NAME,
    wait: float = Query(
        0.0, ge=0.0,
        description="Seconds to wait for an in-flight turn to finish "
                    f"(capped at {LAST_MAX_WAIT:.0f}s). 0 = instant snapshot.",
    ),
):
    """Re-read a session's latest COMPLETED answer, without re-asking.

    For recovering a turn whose /chat response was lost (the 3-min cap, a
    connectivity blip after codex already answered): the answer is durable in
    the rollout even when the HTTP response never arrived. Returns
    {done:true, response} only once the most recent turn has finished;
    {done:false, response:null} while it is still running or was never
    ingested — never a partial answer or a stale previous one.

    /last is the recovery path, so it leaves a full audit trail in the log:
    WHO/WHY (caller host + user-agent + the wait they asked for) on entry, and
    WHAT (turn, durable rollout ref, length + a one-line preview of the answer)
    on exit — enough to reconstruct, after the fact, which client recovered
    which answer from which rollout and when.
    """
    session = await manager.get(name)
    if not session:
        logger.warning(
            "/last MISS session '%s' (unknown) <- %s ua=%r",
            name, _client(request), _ua(request),
        )
        raise HTTPException(status_code=404, detail=f"Session '{name}' not found.")

    logger.info(
        "/last REQUEST session '%s' ref=%s turn=%d wait=%.1fs <- %s ua=%r",
        name, session.ref, session.turn_count, wait, _client(request), _ua(request),
    )
    t0 = time.monotonic()
    done, response, turn = await session.last(min(wait, LAST_MAX_WAIT))
    elapsed = int((time.monotonic() - t0) * 1000)
    if done:
        logger.info(
            "/last HIT session '%s' ref=%s turn=%d recovered %d chars in %dms: %r",
            name, session.ref, turn, len(response), elapsed, _preview(response),
        )
    else:
        logger.info(
            "/last PENDING session '%s' ref=%s turn=%d — no completed answer yet "
            "(still running or never ingested) after %dms",
            name, session.ref, turn, elapsed,
        )
    return LastResponse(
        done=done,
        response=response if done else None,
        turn=turn,
        session=name,
        elapsed_ms=elapsed,
    )


@app.post("/clear/{name}", response_model=MessageResponse)
async def clear(name: str = _NAME):
    """Clear conversation context for a session (respawn in a fresh dir)."""
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
