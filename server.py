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
from datetime import datetime
from pathlib import Path as FsPath

from fastapi import FastAPI, HTTPException, Path, Query, Request
from pydantic import BaseModel, Field

import worktree

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
# budget. But the request is HARD-CAPPED at 3 minutes no matter what
# (RESPONSE_HARD_TIMEOUT), so it never blocks the client for longer.
RESPONSE_STALL_TIMEOUT = float(os.getenv("RESPONSE_STALL_TIMEOUT", "90"))
RESPONSE_HARD_TIMEOUT = float(
    # honor a legacy RESPONSE_MAX_TIMEOUT as the hard ceiling if someone set it
    os.getenv("RESPONSE_HARD_TIMEOUT", os.getenv("RESPONSE_MAX_TIMEOUT", "180"))
)
# Any turn slower than this gets a diagnostic dump even if it SUCCEEDED, so a
# slow-but-fine turn's transcript (timestamped steps — where the time went) is
# captured, not just outright timeouts.
RESPONSE_SLOW_DUMP_SECS = float(os.getenv("RESPONSE_SLOW_DUMP_SECS", "90"))
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

# /last read-back: cap how long a single /last?wait=N call may block, so it
# never holds the client longer than a /chat would (same hard ceiling).
LAST_MAX_WAIT = float(os.getenv("LAST_MAX_WAIT", str(RESPONSE_HARD_TIMEOUT)))

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


def _parse_iso_ts(ts: str | None) -> float | None:
    """An ISO-8601 timestamp (with or without fractional secs / a 'Z') -> epoch
    seconds, or None if absent/unparseable. Lets /last turn the transcript's own
    timestamps into the durations it logs (model time vs poll-wait)."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


# Session names used by liveness probes, not real callers: a /last MISS for one
# is routine noise, so it logs at DEBUG instead of warning the operator.
_PROBE_NAMES = frozenset({"__livecheck__"})


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


def _new_response_bounds(steps: list[dict], baseline: int) -> tuple[str | None, str | None]:
    """(turn_start, answer_complete) ISO timestamps for steps past *baseline*.

    Read straight from the durable transcript so /last can report how long the
    model actually worked and how long the finished answer then sat before a
    poll claimed it — i.e. tell genuine model time apart from our own flow
    latency, from the log alone. start = the earliest new step; complete = the
    last DONE planner response (the moment the recovered answer existed)."""
    new = [s for s in steps if s.get("step_index", -1) > baseline]
    if not new:
        return None, None
    start = min((s["created_at"] for s in new if s.get("created_at")), default=None)
    complete = None
    for s in new:
        if (s.get("source") == "MODEL" and s.get("type") == "PLANNER_RESPONSE"
                and s.get("status") == "DONE" and s.get("content") and s.get("created_at")):
            complete = s["created_at"]
    return start, complete


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
    # Highest transcript step that existed when the most recent turn was
    # submitted. /last uses it to tell THIS turn's answer apart from a previous
    # one: a new completed PLANNER_RESPONSE beyond this step is your turn's.
    _last_baseline_step: int = field(default=-1, init=False)
    # agy keeps cross-conversation memory per project (working directory):
    # new conversations receive summaries of previous ones and the agent can
    # read their transcripts. Every spawn therefore gets a fresh generation
    # directory, which is what makes clear/reset a true context wipe.
    _generation: int = field(default=0, init=False)
    # /last observability only (never affects correctness): how many /last polls
    # the current turn has taken, and the timeout dump (if any) still awaiting a
    # completion footer once /last recovers the answer the turn went on to make.
    _last_attempt: int = field(default=0, init=False)
    _last_dump_path: str | None = field(default=None, init=False)
    _last_dump_turn: int = field(default=-1, init=False)
    _last_dump_finished: bool = field(default=False, init=False)

    # --- Identity ----------------------------------------------------------

    @property
    def tmux_session(self) -> str:
        # safe_name(): the key now carries '@' and '/' (from bases like
        # origin/dev), which are illegal in a tmux session name — so every
        # tmux-name derivation goes through it. The raw self.name stays the
        # manager key / identity; only the tmux/path encoding is sanitized.
        return f"agy-{worktree.safe_name(self.name)}"

    @property
    def _target(self) -> str:
        # '=' forces exact-name matching; the trailing ':' makes the target
        # parse as a session for pane-taking commands (capture-pane etc.).
        return f"={self.tmux_session}:"

    @property
    def cwd(self) -> FsPath:
        # safe_name() collapses the '@'/'/'-bearing key into a single safe path
        # component; a raw key would otherwise fragment into nested dirs.
        return SESSIONS_ROOT / worktree.safe_name(self.name) / f"c{self._generation}"

    # --- Lifecycle ---------------------------------------------------------

    def _build_command(self) -> str:
        parts = [AGY_CMD]
        if AGY_SKIP_PERMISSIONS:
            parts.append("--dangerously-skip-permissions")
        if AGY_EXTRA_ARGS:
            parts.extend(AGY_EXTRA_ARGS.split())
        # Grant the read-only agent access to THIS session's worktree. The
        # static env grant points at the main clone, not the per-session /tmp
        # worktree, so each generation's checkout must be trusted dynamically.
        parts.extend(["--add-dir", str(self.cwd)])
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
        # Each session reads inside its own DETACHED worktree of WORKTREE_REPO,
        # pinned to the caller-named base. resolve_base fetches every spawn
        # (freshness is intentional) and raises if the branch is missing —
        # /chat catches that and replies conversationally so the agent can
        # self-correct. The branchless case is gated at /chat; this guard is
        # only defensive in case _spawn is ever reached without a base.
        _, base = worktree.split_base(self.name)
        if base is None:
            raise RuntimeError("session has no base branch")
        ref = await worktree.resolve_base(base)
        await worktree.add(self.cwd, ref)
        logger.info("Spawning '%s' in tmux session %s (cwd=%s)", cmd, self.tmux_session, self.cwd)

        await _tmux("kill-session", "-t", self._target)  # clear leftovers, ignore rc
        rc, out = await _tmux(
            "new-session", "-d",
            "-s", self.tmux_session,
            "-x", str(TERM_WIDTH), "-y", str(TERM_HEIGHT),
            "-c", str(self.cwd),
            cmd,
        )
        # The worktree already exists; any spawn failure from here must tear it
        # down, or this generation's checkout leaks until the next prune_stale().
        try:
            if rc != 0:
                raise RuntimeError(f"tmux new-session failed: {out.strip()}")
            await self._wait_ready()
        except Exception:
            await _tmux("kill-session", "-t", self._target)  # don't leave a zombie
            await worktree.remove(self.cwd)                   # tear down half-spawned worktree
            raise

        # agy creates its conversation lazily (on the first message or on
        # /clear), so the id is resolved during the first send().
        self._conversation_id = None
        self._turn_count = 0
        self._last_baseline_step = -1
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
        self._last_baseline_step = -1

    async def reset(self) -> None:
        """Kill and re-spawn the process (new conversation)."""
        async with self._lock:
            await self._kill()
            # Drop the current generation's worktree before _spawn re-cuts a
            # fresh one (idempotent; the next generation gets a new path).
            await worktree.remove(self.cwd)
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
            # Tear down this generation's worktree before _spawn re-cuts a fresh
            # one, so the wipe covers the read surface too, not just the process.
            await worktree.remove(self.cwd)
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
            self._last_attempt = 0  # new turn -> fresh recovery-poll count
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
                # Record where this turn began so a later /last can recover its
                # answer and never mistake a previous turn's for it.
                self._last_baseline_step = baseline
            else:
                baseline = _max_step(_read_transcript(self._conversation_id))
                # Publish the baseline BEFORE submitting, so a /last arriving
                # while this turn runs reads the right turn boundary.
                self._last_baseline_step = baseline
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
                    "Session '%s' ref=%s: hit hard timeout (%.0fs) — returning %d "
                    "partial response(s); turn keeps running, recover via /last",
                    self.name, self._conversation_id or "-", RESPONSE_HARD_TIMEOUT,
                    len(responses),
                )
                break
            if now - last_progress > RESPONSE_STALL_TIMEOUT:
                exit_reason = "stalled"
                logger.warning(
                    "Session '%s' ref=%s: no progress for %.0fs (idle, transcript "
                    "not growing) — returning %d response(s)",
                    self.name, self._conversation_id or "-", RESPONSE_STALL_TIMEOUT,
                    len(responses),
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
        if exit_reason != "transcript_done" or total > RESPONSE_SLOW_DUMP_SECS:
            # Snapshot screen + transcript tail for later investigation — on any
            # timeout, and on a slow-but-successful turn so we can see WHERE the
            # time went (the transcript steps are timestamped).
            dump_reason = exit_reason if exit_reason != "transcript_done" else "slow_success"
            await self._dump_timeout_diagnostic(dump_reason, baseline, total, responses)
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
        # Drop EPHEMERAL_MESSAGE housekeeping (one per step, identical filler) so
        # the tail is the model/tool steps that actually show where time went.
        signal = [s for s in steps if s.get("type") != "EPHEMERAL_MESSAGE"]
        tail = signal[-TIMEOUT_DUMP_STEPS:]
        # safe_name, not raw self.name: a base like @origin/dev contains '/',
        # which would nest the dump under a missing parent dir and get the write
        # silently swallowed by the except OSError below.
        path = TIMEOUT_LOG_DIR / f"{worktree.safe_name(self.name)}-turn{self._turn_count}-{self._conversation_id}.log"
        try:
            TIMEOUT_LOG_DIR.mkdir(parents=True, exist_ok=True)
            lines = [
                f"session={self.name} turn={self._turn_count} reason={reason}",
                f"conversation_id={self._conversation_id}",
                f"transcript={_transcript_path(self._conversation_id)}",
                f"baseline={baseline} elapsed={elapsed:.1f}s "
                f"collected_messages={len(responses)} total_steps={len(steps)}",
                f"\n=== last {len(tail)} of {len(signal)} steps "
                f"(EPHEMERAL_MESSAGE filtered; {len(steps)} total) ===",
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
            if reason in ("hard_timeout", "stalled"):
                # Mark this turn's dump so /last can append a completion footer
                # once it recovers the answer the turn produced after give-up.
                self._last_dump_path = str(path)
                self._last_dump_turn = self._turn_count
                self._last_dump_finished = False
        except OSError as exc:
            logger.warning(
                "Session '%s': could not write timeout diagnostic: %s", self.name, exc
            )

    # --- Read-back -----------------------------------------------------------

    async def last(self, wait: float) -> tuple[bool, str, int]:
        """Re-read the most recent turn's answer — only once it is done.

        Read-only and LOCK-FREE: it reads the transcript for the answer and only
        GLANCES at the rendered screen to confirm agy has stopped working — so
        it can recover an answer a /chat call never received (a 3-min hard cap,
        or a connectivity blip) while the turn keeps running in the warm process.

        Unlike codex, agy's transcript carries NO in-flight marker — every step
        is written already-DONE — so "finished" cannot be read from the file
        alone; it is decided by the idle screen, exactly as _collect_response
        does. done=True therefore needs BOTH a new completed answer past the
        submit baseline AND an idle (or gone) process. Otherwise done=False with
        no text, so a still-working turn never yields a partial, and a previous
        turn's answer is never returned in place of one that has not landed.

        Returns (done, answer, turn). With wait>0 it polls up to `wait` seconds.
        """
        deadline = time.monotonic() + max(0.0, wait)
        while True:
            cid = self._conversation_id
            baseline = self._last_baseline_step
            if cid is not None:
                responses = _new_responses(_read_transcript(cid), baseline)
                if responses and await self._idle_or_gone():
                    return True, "\n\n".join(responses).strip(), self._turn_count
            if time.monotonic() >= deadline:
                return False, "", self._turn_count
            await asyncio.sleep(RESPONSE_POLL_INTERVAL)

    async def _idle_or_gone(self) -> bool:
        """True if agy is at its idle prompt, or the process is gone.

        A content-bearing DONE planner response is itself complete, so a dead
        process still has a final answer; only a *live* process that is still
        generating means "not done yet".
        """
        if not await self.is_alive():
            return True
        try:
            screen = await self._capture()
        except RuntimeError:
            return True  # died between the check and the capture
        return READY_MARKER in screen and not self._is_busy(screen)

    def next_attempt(self) -> int:
        """Increment and return this turn's /last poll count (observability)."""
        self._last_attempt += 1
        return self._last_attempt

    def recovery_timing(self) -> tuple[str | None, str | None]:
        """(turn_start, answer_complete) ISO timestamps for the most recent turn,
        from the durable transcript — so /last can log model_turn (real work) vs
        the gap the answer waited for a poll. (None, None) before the first turn."""
        if self._conversation_id is None:
            return None, None
        return _new_response_bounds(
            _read_transcript(self._conversation_id), self._last_baseline_step
        )

    def record_recovery(self, completed_at: str | None, model_secs: float | None) -> None:
        """Append a completion footer to this turn's timeout dump, once /last has
        recovered the answer the turn produced after we gave up — turning the
        dump (frozen at the give-up snapshot) into a complete turn record. Only
        the turn that timed out is touched, and only once. Best-effort."""
        if (not self._last_dump_path or self._last_dump_turn != self._turn_count
                or self._last_dump_finished):
            return
        lines = [
            "\n=== turn COMPLETED after give-up (recovered via /last) ===",
            f"answer_completed_at={completed_at}",
        ]
        if model_secs is not None:
            lines.append(f"model_turn_secs={model_secs:.1f}")
        try:
            with open(self._last_dump_path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            self._last_dump_finished = True
        except OSError:
            pass

    # --- Status --------------------------------------------------------------

    async def is_alive(self) -> bool:
        rc, _ = await _tmux("has-session", "-t", self._target)
        return rc == 0

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def ref(self) -> str:
        """Durable artifact this session reads answers from — the agy
        conversation id. Logged on /last so a recovered answer can be traced
        back to the exact transcript it came from ('-' before the first turn)."""
        return self._conversation_id or "-"


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
            # Deleting a session removes its read surface too (idempotent).
            await worktree.remove(session.cwd)
            return True
        return False

    async def stop_all(self) -> int:
        async with self._lock:
            sessions = dict(self._sessions)
            self._sessions.clear()
        for session in sessions.values():
            await session.stop()
            await worktree.remove(session.cwd)  # drop each session's worktree
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
    # Startup hygiene: the ephemeral SESSIONS_ROOT under /tmp is wiped on
    # container restart while the repo's worktree registry survives in the named
    # volume, so drop registry entries whose checkouts vanished.
    await worktree.prune_stale()
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
    request.state.rid = rid  # so endpoint-level audit lines share the HTTP id
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

# The key is "<name>@<base>": the base is encoded in the path so callers can ask
# about many branches concurrently. The optional "@<base>" suffix permits the
# '.', '/', and '@' a ref name carries (e.g. origin/dev, a SHA). max_length is
# bumped to fit a long base on top of the name.
_NAME = Path(..., pattern=r"^[A-Za-z0-9_-]+(@[A-Za-z0-9._/-]+)?$", max_length=128,
             description="Session name, optionally '<name>@<base-branch>'")

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


# --- Endpoints ---------------------------------------------------------------

# Words that mark a RuntimeError as a caller branch/repo mistake (raised by
# worktree.resolve_base / the no-base guard) rather than a genuine tmux/agy
# startup failure. A caller mistake is handed back as a conversational 200 so
# the agent can self-correct (name a real branch); a startup failure keeps 5xx.
_BRANCH_ERROR_HINTS = ("not found in", "not a git clone", "no base branch")


def _is_branch_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(hint in msg for hint in _BRANCH_ERROR_HINTS)


@app.post("/chat/{name:path}", response_model=ChatResponse)
async def chat(req: ChatRequest, name: str = _NAME):
    """Send a message to a named session. Creates the session on first use."""
    # No base in the key -> nothing is spawned; reply conversationally telling
    # the caller to re-send to /chat/<name>@<branch>. The agent reads this and
    # retries with a branch, so a missing base never costs a CLI spawn.
    session_name, base = worktree.split_base(name)
    if base is None:
        return ChatResponse(
            response=worktree.NEEDS_BRANCH_MSG.format(name=name),
            session=name, turn=0, elapsed_ms=0,
        )

    try:
        session = await manager.get_or_create(name)
    except RuntimeError as e:
        # A bad/missing branch surfaces here (resolve_base raised inside
        # _spawn). Hand it back as a 200 so the calling agent can self-correct
        # instead of seeing an opaque 5xx; genuine startup failures stay 503.
        if _is_branch_error(e):
            return ChatResponse(response=str(e), session=name, turn=0, elapsed_ms=0)
        raise HTTPException(status_code=503, detail=f"Failed to start session '{name}': {e}")

    t0 = time.monotonic()
    try:
        response = await session.send(req.prompt)
    except RuntimeError as e:
        # The first send can also trigger a (re)spawn whose resolve_base fails;
        # treat the same branch errors conversationally rather than as a 502.
        if _is_branch_error(e):
            elapsed = int((time.monotonic() - t0) * 1000)
            return ChatResponse(response=str(e), session=name, turn=0, elapsed_ms=elapsed)
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


@app.get("/last/{name:path}", response_model=LastResponse)
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
    connectivity blip after agy already answered): the answer is durable in the
    transcript even when the HTTP response never arrived. Returns
    {done:true, response} only once the most recent turn has finished;
    {done:false, response:null} while it is still running or was never
    ingested — never a partial answer or a stale previous one.

    /last is the recovery path, so it leaves a full audit trail in the log:
    WHO/WHY (caller host + user-agent + the wait they asked for) on entry, and
    WHAT (turn, durable conversation ref, length + a one-line preview of the
    answer) on exit — enough to reconstruct, after the fact, which client
    recovered which answer from which transcript and when.
    """
    rid = getattr(request.state, "rid", "-")
    session = await manager.get(name)
    if not session:
        # A liveness probe asking for its sentinel session is routine, not a real
        # lost-answer lookup — keep it out of the operator's WARNING stream.
        level = logging.DEBUG if name in _PROBE_NAMES else logging.WARNING
        logger.log(
            level, "[%s] /last MISS session '%s' (unknown) <- %s ua=%r",
            rid, name, _client(request), _ua(request),
        )
        raise HTTPException(status_code=404, detail=f"Session '{name}' not found.")

    attempt = session.next_attempt()
    logger.info(
        "[%s] /last REQUEST session '%s' ref=%s turn=%d attempt=%d wait=%.1fs <- %s ua=%r",
        rid, name, session.ref, session.turn_count, attempt, wait,
        _client(request), _ua(request),
    )
    t0 = time.monotonic()
    done, response, turn = await session.last(min(wait, LAST_MAX_WAIT))
    elapsed = int((time.monotonic() - t0) * 1000)
    if done:
        # Pull the turn's real timing from the transcript: model_turn = how long
        # the agent actually worked; waited_for_poll = how long the finished
        # answer sat before a poll claimed it (~poll cadence + idle settle, i.e.
        # our flow's share, not the model's). Answers "why so slow?" from the log.
        start_ts, done_ts = session.recovery_timing()
        start_epoch, done_epoch = _parse_iso_ts(start_ts), _parse_iso_ts(done_ts)
        model_secs = (done_epoch - start_epoch) if (start_epoch and done_epoch) else None
        waited_secs = max(0.0, time.time() - done_epoch) if done_epoch else None
        session.record_recovery(done_ts, model_secs)
        timing = ""
        if done_ts:
            bits = [f"completed_at={done_ts}"]
            if model_secs is not None:
                bits.append(f"model_turn={model_secs:.0f}s")
            if waited_secs is not None:
                bits.append(f"waited_for_poll={waited_secs:.0f}s")
            timing = " (" + ", ".join(bits) + ")"
        logger.info(
            "[%s] /last HIT session '%s' ref=%s turn=%d attempt=%d recovered %d "
            "chars in %dms%s: %r",
            rid, name, session.ref, turn, attempt, len(response), elapsed, timing,
            _preview(response),
        )
    else:
        logger.info(
            "[%s] /last PENDING session '%s' ref=%s turn=%d attempt=%d — no completed "
            "answer yet (still running or never ingested) after %dms",
            rid, name, session.ref, turn, attempt, elapsed,
        )
    return LastResponse(
        done=done,
        response=response if done else None,
        turn=turn,
        session=name,
        elapsed_ms=elapsed,
    )


@app.post("/clear/{name:path}", response_model=MessageResponse)
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


@app.post("/reset/{name:path}", response_model=MessageResponse)
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


@app.delete("/chat/{name:path}", response_model=MessageResponse)
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
