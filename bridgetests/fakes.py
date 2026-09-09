"""Fakes for hermetic stories: a scripted CLI, a controllable git, a fake clock.

All three sit at seams the servers already expose (the same ones test_bell.py,
test_notify.py, test_server.py and test_worktree.py mock):

  FakeCLI    replaces `<server>._tmux`. It plays tmux AND the CLI behind the
             pane: new-session/has-session/kill-session/capture-pane/
             display-message/load-buffer/paste-buffer/send-keys. A paste + Enter
             starts a turn; the turn "ingests" at once (agy: brain dir +
             transcript step, codex: rollout with session_meta + task_started)
             and completes after the scripted fake delay (agy: DONE
             PLANNER_RESPONSE + bell line, codex: task_complete + notify event).
             Script it with cli.answer("text", after=seconds); unscripted turns
             answer cli.default.
  FakeGit    replaces `worktree._git`. refs is {ref: sha}; `worktree add`
             creates the directory and writes HEAD = sha, `worktree remove`
             deletes it; advance(ref) moves a ref. add_error makes the next add
             fail.
  FakeClock  replaces `<module>.time` and `<module>.asyncio` with proxies whose
             monotonic()/time() read the fake clock and whose sleep() advances
             it, runs the CLI's due turns, then yields to the loop ONCE
             (real asyncio.sleep(0)). wait_for() is covered too: its timeout
             fires on the fake clock (when the awaited task is idle the clock
             jumps to the deadline). Limits: only code INSIDE the patched
             modules sees fake time - Lock waits, real subprocesses and the
             TestClient thread keep real time; a story that needs real time
             to pass (there is none in this suite) must not use it.

`install(monkeypatch, tmp_path, agent)` wires all three into server.py or
codex_server.py with fresh locks, a fresh manager and every on-disk path under
tmp_path, and returns a `FakeBridge`. `FakeBridge.client()` gives a FastAPI
TestClient that runs the real lifespan (prune, notify hook) against the fakes.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import worktree
from bridgetests.live import tmux_display_name

# --- fake clock -----------------------------------------------------------------


class FakeClock:
    """Deterministic time for the patched modules; see the module docstring."""

    def __init__(self, start: float = 1000.0):
        self.start = start
        self.now = start
        self._epoch0 = time.time()
        self._real_sleep = asyncio.sleep
        self.on_advance: list = []  # callbacks(now) run after every advance
        self.sleeps = 0

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        return self._epoch0 + (self.now - self.start)

    @property
    def elapsed(self) -> float:
        return self.now - self.start

    def advance(self, secs: float) -> None:
        self.now += max(0.0, float(secs))
        for cb in list(self.on_advance):
            cb(self.now)

    async def sleep(self, secs: float = 0.0, result: Any = None) -> Any:
        self.advance(secs)
        self.sleeps += 1
        await self._real_sleep(0)
        return result

    def install(self, monkeypatch, *modules) -> None:
        for mod in modules:
            monkeypatch.setattr(mod, "time", _TimeProxy(self))
            monkeypatch.setattr(mod, "asyncio", _AsyncioProxy(self))


class _TimeProxy:
    def __init__(self, clock: FakeClock):
        self._clock = clock

    def monotonic(self) -> float:
        return self._clock.monotonic()

    def time(self) -> float:
        return self._clock.time()

    def __getattr__(self, name: str):
        return getattr(time, name)


class _AsyncioProxy:
    def __init__(self, clock: FakeClock):
        self._clock = clock

    async def sleep(self, delay: float, result: Any = None) -> Any:
        return await self._clock.sleep(delay, result)

    # How many idle yields (task not done, clock not moved) before the clock
    # jumps to the deadline: the awaited task is waiting for something that
    # only fake time can bring (a scripted turn) or nothing at all.
    IDLE_YIELDS = 3

    async def wait_for(self, aw, timeout: float | None):
        """asyncio.wait_for on the fake clock: the timeout is fake seconds."""
        if timeout is None:
            return await aw
        clock = self._clock
        task = asyncio.ensure_future(aw)
        deadline = clock.now + float(timeout)
        idle = 0
        try:
            while True:
                if task.done():
                    return task.result()
                if clock.now >= deadline:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    raise asyncio.TimeoutError()
                before = clock.now
                await clock._real_sleep(0)
                if task.done():
                    continue
                idle = idle + 1 if clock.now == before else 0
                if idle >= self.IDLE_YIELDS:
                    clock.advance(deadline - clock.now)
                    await clock._real_sleep(0)
                    idle = 0
        except asyncio.CancelledError:
            # the caller was cancelled: cancel and await the child, as asyncio does
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            raise

    def __getattr__(self, name: str):
        return getattr(asyncio, name)


# --- fake git ---------------------------------------------------------------------


def _sha(seed: str) -> str:
    import hashlib
    return hashlib.sha1(seed.encode()).hexdigest()


class FakeGit:
    """worktree._git stand-in with controllable origin/<base> refs."""

    def __init__(self, refs: dict[str, str] | None = None):
        # None -> the default origin/main; an explicit {} stays empty so a
        # missing-ref story can exercise the failure.
        self.refs: dict[str, str] = (
            {"origin/main": _sha("origin/main@1")} if refs is None else dict(refs))
        self.repo_ok = True
        self.fetch_rc = 0
        self.add_error: str | None = None  # next `worktree add` fails with this text
        self.calls: list[tuple[str, ...]] = []
        self.added: list[tuple[Path, str, str | None]] = []  # (cwd, ref, sha)
        self.removed: list[Path] = []
        self.fetches = 0
        self.prunes = 0

    def set_ref(self, ref: str, sha: str) -> None:
        self.refs[ref] = sha

    def advance(self, ref: str, sha: str | None = None) -> str:
        """Move *ref* to a new commit (as a fetch would); returns the new sha."""
        self.refs[ref] = sha or _sha(f"{ref}@{uuid.uuid4().hex}")
        return self.refs[ref]

    @staticmethod
    def commit_of(cwd: Path) -> str | None:
        try:
            return (Path(cwd) / "HEAD").read_text().strip()
        except OSError:
            return None

    async def _git(self, *args: str, check: bool = True) -> tuple[int, str]:
        self.calls.append(args)
        rc, text = 0, ""
        if args[:2] == ("rev-parse", "--git-dir"):
            rc = 0 if self.repo_ok else 128
        elif args[:1] == ("fetch",):
            self.fetches += 1
            rc, text = self.fetch_rc, ("" if self.fetch_rc == 0 else "fatal: could not fetch")
        elif args[0] == "rev-parse" and "--verify" in args:
            ref = args[-1][: -len("^{commit}")] if args[-1].endswith("^{commit}") else args[-1]
            if ref in self.refs:
                rc, text = 0, self.refs[ref] + "\n"
            else:
                rc = 1
        elif args[:2] == ("worktree", "add"):
            cwd, ref = Path(args[-2]), args[-1]
            if self.add_error:
                rc, text = 128, self.add_error
                self.add_error = None
            else:
                sha = self.refs.get(ref)
                cwd.mkdir(parents=True, exist_ok=True)
                (cwd / "HEAD").write_text((sha or "") + "\n")
                self.added.append((cwd, ref, sha))
        elif args[:2] == ("worktree", "remove"):
            cwd = Path(args[-1])
            shutil.rmtree(cwd, ignore_errors=True)
            self.removed.append(cwd)
        elif args[:2] == ("worktree", "prune"):
            self.prunes += 1
        if check and rc != 0:
            raise RuntimeError(f"git {' '.join(args)} failed (rc={rc}): {text.strip()}")
        return rc, text

    def install(self, monkeypatch) -> None:
        monkeypatch.setattr(worktree, "_git", self._git)
        monkeypatch.setattr(worktree, "_last_fetch", 0.0)
        monkeypatch.setattr(worktree, "_fetch_lock", None)


# --- fake CLI ---------------------------------------------------------------------


@dataclass
class FakeTurn:
    tmux_session: str
    prompt: str
    answer: str
    delay: float
    submitted_at: float
    due: float
    completed_at: float | None = None


@dataclass
class _Proc:
    """One fake CLI process in one fake tmux session."""
    tmux_session: str
    cwd: Path
    pid: int
    alive: bool = True
    busy: bool = False
    pasted: str | None = None
    pending: FakeTurn | None = None
    # agy: conversation id; codex: rollout path + session id
    conv_id: str | None = None
    rollout: Path | None = None
    session_id: str | None = None
    steps: int = 0        # agy: next transcript step_index
    turns: int = 0        # codex: task_started count


class FakeCLI:
    """Scripted stand-in for tmux + the CLI behind it (agy or codex)."""

    AGY_READY = "╭─ Antigravity ─╮\n\n? for shortcuts\n"
    AGY_BUSY = "╭─ Antigravity ─╮\n\nGenerating...  esc to cancel\n"
    CODEX_READY = "OpenAI Codex\n\n› \n  {model} · {cwd}\n"
    CODEX_BUSY = "OpenAI Codex\n\n  working  esc to interrupt\n"

    def __init__(self, agent: str, module, clock: FakeClock, state_dir: Path):
        assert agent in ("agy", "codex")
        self.agent = agent
        self.mod = module
        self.clock = clock
        self.state_dir = Path(state_dir)
        self.script: deque[tuple[str, float]] = deque()
        self.default: tuple[str, float] = ("ok", 1.0)
        self.procs: dict[str, _Proc] = {}
        self.buffers: dict[str, str] = {}
        self.calls: list[tuple[str, ...]] = []
        self.turns: list[FakeTurn] = []
        self._next_pid = 4242
        clock.on_advance.append(self.tick)

    # --- scripting ----------------------------------------------------------------

    def answer(self, text: str, after: float = 1.0) -> None:
        """The next submitted turn (any session) answers *text* after *after*
        fake seconds."""
        self.script.append((text, float(after)))

    def proc(self, tmux_session: str) -> _Proc | None:
        return self.procs.get(tmux_session)

    def _iso(self) -> str:
        """Timestamp for transcript/rollout events, read from the FAKE clock so
        a scripted 7-second turn shows a 7-second gap between its events."""
        return datetime.fromtimestamp(self.clock.time(), timezone.utc).isoformat()

    def live_sessions(self) -> list[str]:
        return [n for n, p in self.procs.items() if p.alive]

    # --- tmux --------------------------------------------------------------------

    @staticmethod
    def _raw_target(args: tuple[str, ...]) -> str | None:
        return args[args.index("-t") + 1] if "-t" in args else None

    @classmethod
    def _target(cls, args: tuple[str, ...]) -> str | None:
        t = cls._raw_target(args)
        return None if t is None else t.lstrip("=").rstrip(":")

    @staticmethod
    def _bad_target(raw: str) -> str | None:
        """tmux 3.5a (probed in the container): a session is created under
        the normalized name ('.'/':' -> '_'), but a `-t` target is NOT
        normalized. The servers use the session form `={derived}:`; with a
        '.' or ':' in the session part it fails with
        "can't find session: <session part>", rc 1 (has-session, send-keys,
        capture-pane alike). A bare dotted target fails as a pane lookup:
        "can't find pane: <after the last dot>". Returns the message, or None."""
        session_form = raw.startswith("=") or raw.endswith(":")
        part = raw.lstrip("=").rstrip(":")
        if session_form:
            if "." in part or ":" in part:
                return f"can't find session: {part}"
            return None
        if "." in raw:
            return f"can't find pane: {raw.rsplit('.', 1)[1]}"
        if ":" in raw:
            return f"can't find window: {raw.split(':', 1)[1]}"
        return None

    async def _tmux(self, *args: str, stdin_data: bytes | None = None) -> tuple[int, str]:
        self.calls.append(args)
        cmd = args[0]
        target = self._target(args)
        if target is not None and cmd != "new-session":
            bad = self._bad_target(self._raw_target(args))
            if bad is not None:
                return 1, bad
        proc = self.procs.get(target) if target else None
        if cmd == "new-session":
            name = tmux_display_name(args[args.index("-s") + 1])  # as tmux would list it
            cwd = Path(args[args.index("-c") + 1])
            self._next_pid += 1
            self.procs[name] = _Proc(tmux_session=name, cwd=cwd, pid=self._next_pid)
            return 0, ""
        if cmd == "has-session":
            return (0, "") if proc and proc.alive else (1, "no session")
        if cmd == "kill-session":
            if proc and proc.alive:
                proc.alive = False
                proc.pending = None
                return 0, ""
            return 1, "no session"
        if cmd == "display-message":
            return (0, f"{proc.pid}\n") if proc and proc.alive else (1, "no session")
        if cmd == "capture-pane":
            return (0, self._screen(proc)) if proc and proc.alive else (1, "no session")
        if cmd == "load-buffer":
            self.buffers[args[args.index("-b") + 1]] = (stdin_data or b"").decode()
            return 0, ""
        if cmd == "paste-buffer":
            if not (proc and proc.alive):
                return 1, "no session"
            proc.pasted = self.buffers.pop(args[args.index("-b") + 1], "")
            return 0, ""
        if cmd == "send-keys":
            if not (proc and proc.alive):
                return 1, "no session"
            if args[-1] == "Enter" and proc.pasted is not None and not proc.busy:
                self._submit(proc, proc.pasted)
                proc.pasted = None
            return 0, ""
        return 0, ""

    def _screen(self, proc: _Proc) -> str:
        if proc.busy:
            return self.AGY_BUSY if self.agent == "agy" else self.CODEX_BUSY
        if self.agent == "agy":
            return self.AGY_READY
        model = getattr(self.mod, "CODEX_MODEL", "") or "fake-model"
        screen = self.CODEX_READY.format(model=model, cwd=proc.cwd)
        if proc.pasted is not None:
            screen = screen.replace("› \n", f"› [Pasted Content {len(proc.pasted)} chars]\n")
        return screen

    # --- the CLI ------------------------------------------------------------------

    def _submit(self, proc: _Proc, prompt: str) -> None:
        answer, delay = self.script.popleft() if self.script else self.default
        turn = FakeTurn(proc.tmux_session, prompt, answer, delay,
                        self.clock.now, self.clock.now + delay)
        proc.busy = True
        proc.pending = turn
        self.turns.append(turn)
        self._ingest(proc, prompt)
        if delay <= 0:
            self._complete(proc)

    def tick(self, now: float) -> None:
        for proc in self.procs.values():
            if proc.alive and proc.pending and proc.pending.due <= now:
                self._complete(proc)

    # agy: brain dir + transcript + bell
    def _agy_transcript(self, proc: _Proc) -> Path:
        return self.mod._transcript_path(proc.conv_id)

    def _agy_step(self, proc: _Proc, **kw) -> None:
        step = {"step_index": proc.steps, "created_at": self._iso(), "status": "DONE"} | kw
        proc.steps += 1
        with open(self._agy_transcript(proc), "a", encoding="utf-8") as f:
            f.write(json.dumps(step) + "\n")

    # codex: rollout + notify
    def _codex_event(self, proc: _Proc, ev: dict) -> None:
        with open(proc.rollout, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev) + "\n")

    def _ingest(self, proc: _Proc, prompt: str) -> None:
        if self.agent == "agy":
            if proc.conv_id is None:
                proc.conv_id = uuid.uuid4().hex
                self._agy_transcript(proc).parent.mkdir(parents=True, exist_ok=True)
                self._agy_transcript(proc).touch()
            self._agy_step(proc, source="USER", type="USER_MESSAGE", content=prompt)
        else:
            if proc.rollout is None:
                proc.session_id = str(uuid.uuid4())
                sessions = self.mod.CODEX_SESSIONS_DIR
                sessions.mkdir(parents=True, exist_ok=True)
                proc.rollout = sessions / f"rollout-{proc.session_id}.jsonl"
                self._codex_event(proc, {"type": "session_meta",
                                         "payload": {"id": proc.session_id, "cwd": str(proc.cwd)}})
            proc.turns += 1
            turn_id = f"t{proc.turns}"
            self._codex_event(proc, {"type": "event_msg", "timestamp": self._iso(),
                                     "payload": {"type": "task_started", "turn_id": turn_id}})
            model = getattr(self.mod, "CODEX_MODEL", "") or "fake-model"
            self._codex_event(proc, {"type": "turn_context", "timestamp": self._iso(),
                                     "payload": {"model": model, "effort": "xhigh"}})
            self._codex_event(proc, {"type": "response_item", "timestamp": self._iso(),
                                     "payload": {"type": "message", "role": "user",
                                                 "content": [{"type": "input_text", "text": prompt}]}})

    def _complete(self, proc: _Proc) -> None:
        turn = proc.pending
        if turn is None:
            return
        proc.pending = None
        proc.busy = False
        turn.completed_at = self.clock.now
        if self.agent == "agy":
            self._agy_step(proc, source="MODEL", type="PLANNER_RESPONSE", content=turn.answer)
            bell = self.mod.BELL_DIR / proc.tmux_session
            bell.parent.mkdir(parents=True, exist_ok=True)
            with open(bell, "ab") as f:
                f.write(f"{self.clock.time():.9f}\n".encode())
        else:
            turn_id = f"t{proc.turns}"
            self._codex_event(proc, {"type": "response_item", "timestamp": self._iso(),
                                     "payload": {"type": "message", "role": "assistant",
                                                 "content": [{"type": "output_text", "text": turn.answer}]}})
            self._codex_event(proc, {"type": "event_msg", "timestamp": self._iso(),
                                     "payload": {"type": "task_complete", "turn_id": turn_id,
                                                 "last_agent_message": turn.answer}})
            log = self.mod.NOTIFY_LOG
            log.parent.mkdir(parents=True, exist_ok=True)
            with open(log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"type": "agent-turn-complete", "thread-id": proc.session_id,
                                    "turn-id": turn_id, "cwd": str(proc.cwd)}) + "\n")


# --- wiring -----------------------------------------------------------------------


@dataclass
class FakeBridge:
    """Everything a hermetic story needs for one server, wired to the fakes."""
    agent: str
    module: Any
    cli: FakeCLI
    git: FakeGit
    clock: FakeClock
    state_dir: Path
    _client: Any = field(default=None, repr=False)

    @property
    def expected_via(self) -> str:
        """The push completion token per port (TestPRD group H)."""
        return "bell" if self.agent == "agy" else "notify"

    @property
    def manager(self):
        return self.module.manager

    def tmux_session_name(self, key: str) -> str:
        return f"{self.agent}-{worktree.tmux_safe_name(key)}"

    def worktree_dir(self, key: str, generation: int = 1) -> Path:
        return self.module.SESSIONS_ROOT / worktree.safe_name(key) / f"c{generation}"

    def client(self):
        """A FastAPI TestClient; enter it (`with fb.client() as c:`) to run the
        lifespan against the fakes."""
        from fastapi.testclient import TestClient
        return TestClient(self.module.app)


def install(monkeypatch, tmp_path: Path, agent: str, *, clock: FakeClock | None = None,
            git: FakeGit | None = None) -> FakeBridge:
    """Wire the fakes into server.py (agy) or codex_server.py (codex)."""
    if agent == "agy":
        import server as mod
    elif agent == "codex":
        import codex_server as mod
    else:
        raise ValueError(agent)
    state = Path(tmp_path) / agent
    state.mkdir(parents=True, exist_ok=True)
    clock = clock or FakeClock()
    git = git or FakeGit()
    git.install(monkeypatch)
    clock.install(monkeypatch, mod, worktree)
    cli = FakeCLI(agent, mod, clock, state)

    monkeypatch.setattr(mod, "SESSIONS_ROOT", state / "sessions-root")
    monkeypatch.setattr(mod, "TIMEOUT_LOG_DIR", state / "timeouts")
    monkeypatch.setattr(mod, "_tmux", cli._tmux)
    monkeypatch.setattr(mod, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(mod, "_ensure_tmux_server", AsyncMock())
    monkeypatch.setattr(mod, "_SPAWN_LOCK", asyncio.Lock())
    monkeypatch.setattr(mod, "manager", mod.ChatManager())
    if agent == "agy":
        monkeypatch.setattr(mod, "AGY_STATE_DIR", state / "agy-state")
        monkeypatch.setattr(mod, "BELL_DIR", state / "bells")
        monkeypatch.setattr(mod, "_STARTUP_LOCK", asyncio.Lock())
        (state / "agy-state" / "brain").mkdir(parents=True, exist_ok=True)
    else:
        monkeypatch.setattr(mod, "CODEX_SESSIONS_DIR", state / "codex-sessions")
        monkeypatch.setattr(mod, "NOTIFY_DIR", state / "notify")
        monkeypatch.setattr(mod, "NOTIFY_LOG", state / "notify" / "events.jsonl")
        monkeypatch.setattr(mod, "NOTIFY_HOOK", state / "notify" / "notify-hook.py")
    return FakeBridge(agent=agent, module=mod, cli=cli, git=git, clock=clock, state_dir=state)
