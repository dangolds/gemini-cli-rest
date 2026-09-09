"""Live-bridge client and session ownership for the behavior suite.

One `Bridge` per port (agy on 8000, codex on 8001). Every HTTP helper takes the
FULL key exactly as `bridgetests.names` built it: no default-basing happens
here, the helper is the only place names are built.

Ownership (TestPRD section 5) is enforced HERE, not by convention: every
helper that names a session (chat, chat_raw, last, clear, reset, delete)
raises ValueError before any HTTP unless the key carries this run's stamp
(`names.is_ours`). Only health/sessions are open. A `SessionRegistry`
remembers the keys a story or a class opened and deletes them at teardown,
also on failure; it refuses to adopt a key unless /health answered 200 and
neither lists the key nor a pane named for it exists (a pinned
BRIDGE_RUN_STAMP shared by two concurrent runs, or a leftover).

Teardown never trusts the HTTP status alone: both servers pop the manager
entry BEFORE stopping the session, so a delete that failed or timed out can
leave a live pane whose next DELETE answers 404. After the delete (any
status, or past DELETE_DEADLINE - a running turn can hold the session lock for
the whole 180 s cap) teardown lists the container's tmux panes; if the ONE
pane whose tmux session name is derived from the key (the same
`<agent>-<worktree.tmux_safe_name(key)>` derivation the servers use) still exists,
it kills only that pid, verifies it is gone, deletes again and lists again.
Nothing unverified is ever killed; keys that could not be resolved are kept
in `UNRESOLVED` for the session-finish report. A worktree generation the
delete left on disk (the servers only prune at startup) is swept by teardown
(`git worktree remove --force`, then prune), and the outcome says so
("deleted+swept").

Every Bridge owns one `httpx.Client`; `close_all()` (called at session finish
by stories/conftest.py) closes them, which also tears down the daemon workers
of abandoned (deadline-expired) requests.

Nothing in this module calls POST /stop. The retained suites' live stop tests
are the only live stop callers, and they need BRIDGE_LIVE_STOP=1.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import threading
import time
import weakref
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

import httpx

import worktree
from bridgetests import names

# --- knobs ---------------------------------------------------------------------
HOST = os.environ.get("BRIDGE_HOST", "127.0.0.1")
CONTAINER = os.environ.get("BRIDGE_CONTAINER", "gemini-cli-rest-bridges-1")
PORTS: dict[str, int] = {"agy": 8000, "codex": 8001}
SESSIONS_ROOTS: dict[str, str] = {
    "agy": "/tmp/agy-rest-sessions",
    "codex": "/tmp/codex-rest-sessions",
}
TMUX_SOCKETS: dict[str, str] = {"agy": "agy-rest", "codex": "codex-rest"}
REPO_IN_CONTAINER = "/app/slitled-platform"
LOG_DIR_IN_CONTAINER = "/app/logs"

# A /chat is capped at 180 s inside the bridge (RESPONSE_HARD_TIMEOUT); the
# client waits a little longer so the 504 itself arrives instead of a client
# timeout. A first turn also spawns (up to 60 s) - pass a larger timeout when a
# story provokes that worst case.
CHAT_TIMEOUT = 185.0
# Delete waits for a running turn (the cap) before it acts, so the teardown
# deadline sits above the cap; past it teardown kills the verified pane.
DELETE_DEADLINE = 200.0
# The second delete, after the pane kill, has nothing left to wait for.
DELETE_RETRY_TIMEOUT = 60.0
# After kill -9 the pane is re-listed until it is gone, at most this long.
KILL_VERIFY_WAIT = 5.0
# Live /stop unlock for the retained suites (never used by the new stories).
LIVE_STOP_FLAG = "BRIDGE_LIVE_STOP"

# Keys whose teardown did not end in deleted/absent/killed+deleted, as
# (agent, key, outcome); stories/conftest.py prints it at session finish.
UNRESOLVED: list[tuple[str, str, str]] = []


def live_stop_allowed() -> bool:
    return os.environ.get(LIVE_STOP_FLAG, "").strip() == "1"


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _say(msg: str) -> None:
    print(f"  [{_ts()}] {msg}", flush=True)


def tmux_display_name(name: str) -> str:
    """The name tmux (3.5a, session_check_name) actually gives a session
    created as *name*: ':' and '.' become '_' (both are target separators),
    nothing else changes. worktree.safe_name keeps dots, so a key with a dotted
    base ("x@release/1.2") is LISTED under a different name than the server
    derived; every comparison against `list-panes` output goes through here."""
    return name.replace(".", "_").replace(":", "_")


# The servers' route grammar (server.py / codex_server.py `_NAME`), split in two
# so the base can also be checked segment by segment.
_NAME_PART = re.compile(r"^[A-Za-z0-9_-]+$")
_BASE_PART = re.compile(r"^[A-Za-z0-9._/-]+$")
_MAX_KEY_LEN = 128


def _own(session: str) -> str:
    """The ownership gate every session-naming helper passes through: the key
    must carry this run's stamp AND fit the servers' route grammar, with no
    empty, "." or ".." base segment (a stamped key could otherwise smuggle
    "@main/../../stop" into the URL). The retained suites' chat() helpers call
    it directly (test_server.py, test_codex_server.py)."""
    if not names.is_ours(session):
        raise ValueError(
            f"refusing to address {session!r}: it does not carry run stamp {names.STAMP} "
            f"(build it with bridgetests.names.key/bare; raw() names are hermetic-only)")
    name, at, base = session.partition("@")
    # The repository prefix ("slitled/" after PRD-multi-repo) is the servers'
    # business; the grammar applies to what follows it.
    name = name.removeprefix(names.REPO_PREFIX)
    if len(session) > _MAX_KEY_LEN or not _NAME_PART.fullmatch(name):
        raise ValueError(f"refusing to address {session!r}: name part is not route-safe")
    if at:
        if not _BASE_PART.fullmatch(base):
            raise ValueError(f"refusing to address {session!r}: base is not route-safe")
        if any(seg in ("", ".", "..") for seg in base.split("/")):
            raise ValueError(f"refusing to address {session!r}: base has an empty, '.' or '..' segment")
    return session


@dataclass
class Reply:
    """An HTTP outcome the caller decides about: nothing here raises.

    status is None when no response arrived (timeout, connection error); then
    `error` says why. `body` is the parsed JSON object, or None.
    """
    status: int | None
    body: dict | None
    text: str
    elapsed: float
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == 200

    def __getitem__(self, item: str) -> Any:
        if self.body is None:
            raise KeyError(f"no JSON body (status={self.status}, error={self.error})")
        return self.body[item]

    def __contains__(self, item: object) -> bool:
        return self.body is not None and item in self.body

    def get(self, item: str, default: Any = None) -> Any:
        return default if self.body is None else self.body.get(item, default)


@dataclass
class Bridge:
    """One live bridge: HTTP helpers plus a docker exec into its container."""

    agent: str
    port: int
    host: str = HOST
    container: str = CONTAINER
    # Abandoned (deadline-expired) requests still in flight, per session key:
    # their POST may create or mutate the session later, so teardown waits
    # for them before it deletes (see _request / teardown_session).
    _pending: dict[str, list[threading.Event]] = field(default_factory=dict, repr=False)
    # Keys whose request ended with status None for an ordinary httpx timeout
    # or connection loss (not the deadline path): the server may still have
    # executed it, so teardown re-checks and marks its outcome with "?".
    _uncertain: set[str] = field(default_factory=set, repr=False)
    # One HTTP client per bridge (lazily built); close()/close_all() end it,
    # and with it the daemon workers of abandoned requests.
    _client: httpx.Client | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _BRIDGES[id(self)] = self

    @classmethod
    def for_agent(cls, agent: str) -> "Bridge":
        return cls(agent=agent, port=PORTS[agent])

    @classmethod
    def for_url(cls, agent: str, base_url: str) -> "Bridge":
        """A bridge at an explicit URL (the retained suites' BASE), ignoring
        BRIDGE_HOST; the port defaults to the agent's when the URL has none."""
        u = urlsplit(base_url)
        if not u.hostname:
            raise ValueError(f"no host in {base_url!r}")
        return cls(agent=agent, port=u.port or PORTS[agent], host=u.hostname)

    # --- identity ---------------------------------------------------------------

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def sessions_root(self) -> str:
        """Where this bridge keeps worktrees inside the container
        (<root>/<run-id>/<safe-name>/c<generation>)."""
        return SESSIONS_ROOTS[self.agent]

    @property
    def tmux_socket(self) -> str:
        return TMUX_SOCKETS[self.agent]

    def tmux_session_name(self, key: str) -> str:
        """The tmux session name the server derives for *key* (server.py /
        codex_server.py: f"{agent}-{worktree.tmux_safe_name(name)}"). Compare
        it to listed names only through tmux_display_name()."""
        return f"{self.agent}-{worktree.tmux_safe_name(key)}"

    def __repr__(self) -> str:  # short ids in pytest output
        return f"Bridge({self.agent}:{self.port})"

    # --- container access --------------------------------------------------------

    def docker_exec(self, *cmd: str, timeout: float = 30.0) -> tuple[int, str]:
        """Run *cmd* inside the container; (rc, combined output). rc 127 on a
        docker failure (no docker, no container)."""
        try:
            p = subprocess.run(
                ["docker", "exec", self.container, *cmd],
                capture_output=True, text=True, timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return 127, f"docker exec failed: {exc}"
        return p.returncode, (p.stdout or "") + (p.stderr or "")

    def list_panes(self) -> list[tuple[str, int]]:
        """[(tmux session name, pane pid)] on this bridge's tmux socket
        ([] also when docker itself is unavailable; see _list_panes_checked)."""
        return self._list_panes_checked() or []

    PANE_RC_MARKER = "__tmux_rc="
    # What tmux says when there is simply nothing to list (tmux 3.5a in the
    # container: "error connecting to /tmp/tmux-0/<socket> (No such file or
    # directory)" with no server, "no current target" with a server but no
    # sessions; older tmux: "no server running on ...", "no sessions").
    NO_PANES_TEXT = re.compile(
        r"error connecting to \S+ \(No such file or directory\)|no server running"
        r"|no sessions|no current target")

    def _list_panes_checked(self) -> list[tuple[str, int]] | None:
        """Like list_panes, but None when the container could not be asked
        (no docker, no container, daemon down, exec timeout) - "unknown" is
        not "none". docker itself often exits 1, like tmux does with no
        server, so the tmux rc travels inside the output as a marker: a
        missing marker means the shell in the container never ran."""
        rc, out = self.docker_exec(
            "sh", "-c",
            f"tmux -L {self.tmux_socket} list-panes -a -F '#{{session_name}} #{{pane_pid}}'; "
            f"echo '{self.PANE_RC_MARKER}'$?",
        )
        m = re.search(rf"{re.escape(self.PANE_RC_MARKER)}(\d+)", out)
        if m is None:
            _say(f"cannot list panes of {self.agent} (docker rc={rc}): {out.strip()[:200]}")
            return None
        panes: list[tuple[str, int]] = []
        if int(m.group(1)) != 0:
            if self.NO_PANES_TEXT.search(out):
                return panes  # confirmed: no tmux server / no sessions
            _say(f"tmux list-panes on {self.agent} failed for another reason: {out.strip()[:200]}")
            return None
        for line in out.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit():
                panes.append((parts[0], int(parts[1])))
        return panes

    def pane_pids(self, key: str) -> list[int] | None:
        """Pids of the panes named exactly as the server derives for *key*;
        None when the container could not be asked."""
        panes = self._list_panes_checked()
        if panes is None:
            return None
        want = tmux_display_name(self.tmux_session_name(key))
        return [pid for (sess, pid) in panes if sess == want]

    def worktree_dirs(self, key: str) -> list[str]:
        """Container paths of *key*'s worktree generations that exist on disk."""
        safe = worktree.safe_name(key)
        rc, out = self.docker_exec(
            "sh", "-c", f"ls -d {self.sessions_root}/*/{safe}/c* 2>/dev/null",
        )
        return [ln.strip() for ln in out.splitlines() if ln.strip()] if rc == 0 else []

    def image_id(self) -> str:
        try:
            p = subprocess.run(
                ["docker", "inspect", "-f", "{{.Image}}", self.container],
                capture_output=True, text=True, timeout=15,
            )
            return (p.stdout.strip() or "unknown") if p.returncode == 0 else "unknown"
        except (OSError, subprocess.TimeoutExpired):
            return "unknown"

    # --- HTTP -------------------------------------------------------------------

    # Slack on top of the per-phase httpx timeout before the absolute deadline
    # of _request fires (a healthy request ends within `timeout`; the slack only
    # lets the httpx exception itself arrive first).
    DEADLINE_SLACK = 5.0

    def _request(self, method: str, path: str, *, timeout: float, session: str | None = None,
                 **kw) -> Reply:
        # httpx.Timeout(t) is PER PHASE (connect, read, write, pool), not an
        # absolute deadline: a slow trickle can keep a request alive past t.
        # So the call runs in a daemon worker thread and is waited for at most
        # timeout + DEADLINE_SLACK; on expiry the Reply says so and the thread
        # is ABANDONED (it dies with the process, or when httpx gives up on its
        # own) - nothing waits for it, and its late result is dropped.
        # A plain daemon Thread, not a ThreadPoolExecutor: executor workers
        # are joined at interpreter exit even after shutdown(wait=False), so an
        # abandoned trickling request would hang pytest's exit. close() ends
        # the shared client, and the abandoned worker with it.
        t0 = time.monotonic()
        url = f"{self.base_url}{path}"
        client = self.client()
        box: list = []  # [("ok", response)] or [("err", exception)]
        done = threading.Event()

        def run() -> None:
            try:
                box.append(("ok", client.request(method, url, timeout=httpx.Timeout(timeout), **kw)))
            except BaseException as exc:  # noqa: BLE001 - relayed to the caller
                box.append(("err", exc))
            finally:
                done.set()
                if session is not None:  # an abandoned worker unregisters itself
                    self._forget_pending(session, done)

        threading.Thread(target=run, name="bridge-http", daemon=True).start()
        if not done.wait(timeout + self.DEADLINE_SLACK):
            if session is not None and not done.is_set():
                self._pending.setdefault(session, []).append(done)
            return Reply(None, None, "", time.monotonic() - t0,
                         f"deadline of {timeout + self.DEADLINE_SLACK:.0f}s exceeded ({method} {path})")
        kind, val = box[0]
        if kind == "err":
            if isinstance(val, httpx.HTTPError) and session is not None:
                self._uncertain.add(session)  # the server may still have run it
            if isinstance(val, httpx.TimeoutException):
                return Reply(None, None, "", time.monotonic() - t0, f"timeout after {timeout:.0f}s: {val!r}")
            if isinstance(val, httpx.HTTPError):
                return Reply(None, None, "", time.monotonic() - t0, f"{type(val).__name__}: {val}")
            raise val
        r = val
        body: dict | None
        try:
            parsed = r.json()
            body = parsed if isinstance(parsed, dict) else None
        except ValueError:
            body = None
        return Reply(r.status_code, body, r.text, time.monotonic() - t0)

    def client(self) -> httpx.Client:
        """The bridge's HTTP client (built on first use, rebuilt after close())."""
        if self._client is None:
            self._client = httpx.Client()
        return self._client

    def close(self) -> None:
        """Close the HTTP client (safe to repeat); an abandoned worker still
        inside it fails and ends instead of trickling on."""
        c, self._client = self._client, None
        if c is not None:
            c.close()

    def _forget_pending(self, session: str, ev: threading.Event) -> None:
        lst = self._pending.get(session)
        if lst and ev in lst:
            lst.remove(ev)
        if lst is not None and not lst:
            self._pending.pop(session, None)

    def pending_requests(self, session: str) -> list[threading.Event]:
        """Abandoned requests to *session* whose worker has not finished."""
        return [ev for ev in self._pending.get(session, []) if not ev.is_set()]

    def wait_pending(self, session: str, timeout: float) -> bool:
        """Wait up to *timeout* for every abandoned request to *session* to
        finish; True when none is left in flight."""
        deadline = time.monotonic() + timeout
        for ev in list(self._pending.get(session, [])):
            if not ev.wait(max(0.0, deadline - time.monotonic())):
                return False
        return not self.pending_requests(session)

    def health(self, timeout: float = 10.0) -> Reply:
        return self._request("GET", "/health", timeout=timeout)

    def is_up(self) -> bool:
        return self.health(timeout=3.0).status == 200

    def chat(self, session: str, prompt: str, timeout: float = CHAT_TIMEOUT) -> Reply:
        """POST /chat/<session>. Never raises: a 504, a 5xx or a client timeout
        come back as a Reply for the story to judge. Raises ValueError (before
        any HTTP) for a session this run does not own."""
        _own(session)
        _say(f">>> {self.agent} {session}: {prompt[:120]!r}")
        rep = self._request("POST", f"/chat/{session}", timeout=timeout, session=session,
                            json={"prompt": prompt})
        if rep.body is not None and "turn" in rep.body:
            _say(f"<<< {self.agent} {session} (turn {rep.body['turn']}, {rep.elapsed:.1f}s, "
                 f"via={rep.body.get('via')}): {str(rep.body.get('response'))[:160]!r}")
        else:
            _say(f"<<< {self.agent} {session}: status={rep.status} {rep.error or rep.text[:160]!r} "
                 f"({rep.elapsed:.1f}s)")
        return rep

    def chat_raw(self, session: str, payload: Any, timeout: float = 30.0, *,
                 content: bytes | None = None) -> Reply:
        """POST /chat/<session> with an arbitrary JSON payload (or raw bytes),
        for malformed-request stories. The session itself must still be ours."""
        _own(session)
        if content is not None:
            return self._request("POST", f"/chat/{session}", timeout=timeout, session=session,
                                 content=content, headers={"content-type": "application/json"})
        return self._request("POST", f"/chat/{session}", timeout=timeout, session=session, json=payload)

    def last(self, session: str, wait: float | None = None, timeout: float | None = None) -> Reply:
        _own(session)
        params = {} if wait is None else {"wait": wait}
        if timeout is None:
            timeout = (wait or 0) + 30.0
        rep = self._request("GET", f"/last/{session}", timeout=timeout, session=session, params=params)
        if rep.body is not None and "done" in rep.body:
            _say(f"/last {self.agent} {session}: done={rep.body['done']} status={rep.body.get('status')} "
                 f"turn={rep.body.get('turn')} resp={str(rep.body.get('response'))[:120]!r}")
        else:
            _say(f"/last {self.agent} {session}: status={rep.status} {rep.error or rep.text[:120]!r}")
        return rep

    def clear(self, session: str, timeout: float = DELETE_DEADLINE) -> Reply:
        _own(session)
        rep = self._request("POST", f"/clear/{session}", timeout=timeout, session=session)
        _say(f"/clear {self.agent} {session}: status={rep.status} ({rep.elapsed:.1f}s)")
        return rep

    def reset(self, session: str, timeout: float = DELETE_DEADLINE) -> Reply:
        _own(session)
        rep = self._request("POST", f"/reset/{session}", timeout=timeout, session=session)
        _say(f"/reset {self.agent} {session}: status={rep.status} ({rep.elapsed:.1f}s)")
        return rep

    def delete(self, session: str, timeout: float = DELETE_DEADLINE) -> Reply:
        _own(session)
        rep = self._request("DELETE", f"/chat/{session}", timeout=timeout, session=session)
        _say(f"DELETE {self.agent} {session}: status={rep.status} ({rep.elapsed:.1f}s)")
        return rep

    def sessions(self) -> dict[str, dict]:
        """{name: {alive, turn_count}} from /health ({} when it is down)."""
        h = self.health()
        if h.body is None:
            return {}
        return {s["name"]: s for s in h.body.get("sessions", [])}

    # --- ownership -------------------------------------------------------------

    def assert_not_live(self, key: str) -> None:
        """Verified "nothing of *key* exists here": /health answered 200 and does
        not list it, and the container could be asked for panes and has none
        named for it. ValueError otherwise (health unavailable, listed key,
        panes unknown, orphan pane) - a pinned stamp shared by two runs, or a
        leftover, is never adopted."""
        h = self.health()
        if h.status != 200 or h.body is None:
            raise ValueError(
                f"cannot verify ownership of {key!r}: health unavailable on "
                f"{self.agent}:{self.port} (status={h.status}, {h.error})")
        if any(s.get("name") == key for s in h.body.get("sessions", [])):
            raise ValueError(
                f"{key!r} is already live on {self.agent}:{self.port}: "
                f"key already live on the bridge: pinned stamp collision or leftover")
        pids = self.pane_pids(key)
        if pids is None:
            raise ValueError(f"cannot verify ownership of {key!r}: the container cannot be asked for panes")
        if pids:
            raise ValueError(
                f"{key!r} has an orphan pane on {self.agent} (pids {pids}): "
                f"pinned stamp collision or leftover")

    # --- ownership teardown -----------------------------------------------------

    def kill_verified_pane(self, key: str) -> int | None:
        """Kill the pane pid of *key*'s tmux session, only if the key carries
        this run's stamp AND exactly one pane with the derived session name
        exists. Returns the pid only once the pane is verified gone (re-listed
        for up to KILL_VERIFY_WAIT); None when nothing qualified, the kill
        failed, or the pane survived."""
        if not names.is_ours(key):
            _say(f"refusing to kill: {key!r} does not carry run stamp {names.STAMP}")
            return None
        want = tmux_display_name(self.tmux_session_name(key))
        match = self.pane_pids(key)
        if match is None:
            _say(f"cannot verify a pane for {want!r}; nothing killed")
            return None
        if len(match) != 1:
            _say(f"no single pane named {want!r} (found {match}); nothing killed")
            return None
        pid = match[0]
        # `kill` is a shell builtin in the container (no /bin/kill): go through sh.
        rc, out = self.docker_exec("sh", "-c", f"kill -9 {pid}")
        if rc != 0:
            _say(f"kill -9 {pid} of {want} failed (rc={rc}) {out.strip()[:200]}")
            return None
        deadline = time.monotonic() + KILL_VERIFY_WAIT
        while True:
            left = self.pane_pids(key)
            if left == []:
                _say(f"killed pane pid {pid} of {want}; pane gone")
                return pid
            if time.monotonic() >= deadline:
                _say(f"kill -9 {pid} of {want} returned 0 but the pane is still listed ({left})")
                return None
            time.sleep(0.25)

    CONFIRMED = ("deleted", "absent", "killed+deleted", "deleted+swept", "killed+deleted+swept")

    def teardown_session(self, key: str) -> str:
        """Delete *key* and VERIFY it is gone. Outcome words:
        deleted | absent | killed+deleted | deleted? | absent? | failed:<why>,
        the deleted forms also with "+swept"; CONFIRMED lists the resolved ones.

        The HTTP status is not trusted on its own: the servers pop the manager
        entry before stopping the session, so a failed/timed-out DELETE can
        leave a live pane that the next DELETE reports as 404. After the delete
        (any status, or once Reply.elapsed reached DELETE_DEADLINE) the panes
        are listed; a surviving pane named for this key is killed (verified),
        the delete repeated, the panes listed again. Before all that, any
        abandoned (deadline-expired) request to the key is waited for, up to
        DELETE_DEADLINE: its late POST could still spawn the session after an
        "absent"; if it is still in flight the key stays unresolved. Before
        every successful return the pending list is re-checked (the teardown's
        own DELETE may have been abandoned) and, after a delete, the key's
        worktree generations must be gone from disk. A key whose earlier
        request ended in an ordinary timeout/connection loss (`_uncertain`) is
        re-checked once more after the pass; if something reappeared it is
        deleted again (confirmed), otherwise the outcome carries a "?" and the
        key is no longer uncertain (reported once). A worktree generation left
        on disk after a delete is swept (see sweep_worktrees) and the outcome
        gets "+swept"."""
        if not names.is_ours(key):
            raise ValueError(f"refusing to tear down a session this run did not create: {key!r}")
        if self.pending_requests(key):
            _say(f"{key}: {len(self.pending_requests(key))} abandoned request(s) still in flight; "
                 f"waiting up to {DELETE_DEADLINE:.0f}s before deleting")
            if not self.wait_pending(key, DELETE_DEADLINE):
                return "failed:request-still-pending"
        was_uncertain = key in self._uncertain
        outcome = self._teardown_once(key)
        if outcome not in self.CONFIRMED:
            return outcome
        if not was_uncertain:
            self._uncertain.discard(key)
            return outcome
        # An earlier request may still have landed on the server: look again.
        _say(f"{key}: an earlier request ended uncertain; re-checking health and panes")
        h = self.health()
        if h.status != 200 or h.body is None:
            return "failed:health-unavailable"
        listed = any(sess.get("name") == key for sess in h.body.get("sessions", []))
        pids = self.pane_pids(key)
        if pids is None:
            return "failed:cannot-list-panes"
        if listed or pids:
            _say(f"{key}: reappeared (listed={listed}, pids={pids}); deleting again")
            again = self._teardown_once(key)
            if again in self.CONFIRMED:
                self._uncertain.discard(key)  # the late request landed and was cleaned up
            return again
        # Reported once: a later teardown of the same key says a plain word.
        self._uncertain.discard(key)
        if outcome.startswith("killed+deleted"):
            return outcome
        return outcome + "?"  # nothing visible now, but a late request could still arrive

    def sweep_worktrees(self, key: str, dirs: list[str]) -> list[str]:
        """Remove *key*'s leftover worktree generations *dirs* from the
        container (`git worktree remove --force`, `rm -rf` as the fallback,
        then `git worktree prune`) and return what is still on disk. Only
        paths under this bridge's sessions root are touched; anything else
        is refused and reported as left."""
        root = self.sessions_root + "/"
        safe = [d for d in dirs if d.startswith(root) and ".." not in d.split("/")]
        if len(safe) != len(dirs):
            _say(f"refusing to sweep outside {root}: {[d for d in dirs if d not in safe]}")
            return dirs
        git = f"git -C {REPO_IN_CONTAINER}"
        script = "; ".join(
            f"{git} worktree remove --force {shlex.quote(d)} || rm -rf {shlex.quote(d)}" for d in safe
        ) + f"; {git} worktree prune"
        rc, out = self.docker_exec("sh", "-c", script, timeout=60.0)
        _say(f"swept worktree(s) of {key} (rc={rc}): {out.strip()[:200]}")
        return self.worktree_dirs(key)

    def _finish(self, key: str, outcome: str) -> str:
        """The gate before every successful return of a teardown pass."""
        if self.pending_requests(key):
            return "failed:request-still-pending"
        if outcome in ("deleted", "killed+deleted"):
            left = self.worktree_dirs(key)
            if left:
                _say(f"worktree(s) of {key} left on disk after delete: {left}; sweeping")
                left = self.sweep_worktrees(key, left)
                if left:
                    return f"failed:worktree-left({left})"
                return outcome + "+swept"
        return outcome

    def _teardown_once(self, key: str) -> str:
        """One delete-and-verify pass (see teardown_session)."""
        rep = self.delete(key, timeout=DELETE_DEADLINE)
        timed_out = rep.status is None or rep.elapsed >= DELETE_DEADLINE
        if timed_out:
            _say(f"delete of {key} passed the {DELETE_DEADLINE:.0f}s deadline "
                 f"(status={rep.status}, {rep.elapsed:.1f}s, {rep.error}); checking its pane")
        elif rep.status not in (200, 404):
            _say(f"delete of {key} answered {rep.status}: {rep.text[:200]!r}; checking its pane")

        pids = self.pane_pids(key)
        if pids is None:
            return "failed:cannot-list-panes"
        if not pids:
            if not timed_out and rep.status == 200:
                return self._finish(key, "deleted")
            if not timed_out and rep.status == 404:
                return self._finish(key, "absent")
            # No pane, but the request did not say 200/404: the entry may
            # still be in the manager. Ask /health (it must answer) before
            # calling it done.
            h = self.health()
            if h.status != 200 or h.body is None:
                return "failed:health-unavailable"
            if not any(sess.get("name") == key for sess in h.body.get("sessions", [])):
                return self._finish(key, "deleted" if timed_out else "absent")
            return f"failed:http-{rep.status}-entry-listed"

        _say(f"pane of {key} still alive after delete (pids {pids}); killing it")
        if self.kill_verified_pane(key) is None:
            return "failed:pane-not-killed"
        again = self.delete(key, timeout=DELETE_RETRY_TIMEOUT)
        left = self.pane_pids(key)
        if left is None:
            return "failed:cannot-list-panes-after-kill"
        if left:
            return f"failed:pane-survived({left})"
        if again.status in (200, 404):
            return self._finish(key, "killed+deleted")
        return f"failed:second-delete-{again.status or again.error}"


@dataclass
class SessionRegistry:
    """Keys a story (function scope) or a class (class scope) owns.

    Call it to build-and-register: `k = own_session("chain")` (or
    `own_session.key("chain")`) -> names.key("chain") registered for teardown. `register(k)` adopts a key
    built elsewhere with the helper. Teardown deletes every key, in reverse
    order, with the deadline rule; outcomes are printed, never asserted.
    """
    bridge: Bridge
    label: str = "story"
    keys: list[str] = field(default_factory=list)

    def __call__(self, name: str, base: str | None = "main") -> str:
        return self.register(names.key(name, base))

    def key(self, name: str, base: str | None = "main") -> str:
        """Alias of __call__, mirroring GroupSession.key()."""
        return self(name, base)

    def register(self, key: str) -> str:
        """Adopt *key*: it must carry this run's stamp and must not already be
        live on the bridge (two processes pinned to the same BRIDGE_RUN_STAMP
        would otherwise delete each other's sessions)."""
        if not names.is_ours(key):
            raise ValueError(f"{key!r} was not built by bridgetests.names (no run stamp)")
        if key not in self.keys:
            self.bridge.assert_not_live(key)
            self.keys.append(key)
        return key

    def teardown(self) -> list[tuple[str, str]]:
        """Delete every key (reverse order) with verification; outcomes are
        printed, never asserted. Keys that did not end deleted/absent/
        killed+deleted are appended to the module-level UNRESOLVED."""
        outcomes: list[tuple[str, str]] = []
        for key in reversed(self.keys):
            try:
                outcome = self.bridge.teardown_session(key)
            except Exception as exc:  # teardown must never mask the story's result
                outcome = f"failed:{type(exc).__name__}: {exc}"
            _say(f"teardown[{self.label}] {self.bridge.agent} {key}: {outcome}")
            outcomes.append((key, outcome))
            if outcome not in Bridge.CONFIRMED:
                UNRESOLVED.append((self.bridge.agent, key, outcome))
        self.keys.clear()
        return outcomes


class GroupSession:
    """A class-shared session: the key is built on first use and deleted at
    class teardown. Stories in the group never assert on the turn count. The
    default name is derived from the owning module and class
    ("group-<module>-<classname>"), so two classes never share a key, not
    even same-named ones in different modules."""

    def __init__(self, registry: SessionRegistry, default_name: str = "group"):
        self._registry = registry
        self._key: str | None = None
        self.default_name = default_name

    @classmethod
    def default_name_for(cls, module: str | None, classname: str | None) -> str:
        def part(text: str | None, fallback: str) -> str:
            return re.sub(r"[^A-Za-z0-9_-]", "-", (text or fallback).lower())
        stem = (module or "").rpartition(".")[2].removeprefix("test_")
        return f"group-{part(stem, 'module')}-{part(classname, 'class')}"

    @property
    def bridge(self) -> Bridge:
        return self._registry.bridge

    def key(self, name: str | None = None, base: str | None = "main") -> str:
        if self._key is None:
            self._key = self._registry(name or self.default_name, base)
        return self._key

    def __call__(self, name: str | None = None, base: str | None = "main") -> str:
        return self.key(name, base)


# Every Bridge built so far (weak: instances die with their fixtures; a
# dataclass with eq is unhashable, hence a dict keyed by id).
_BRIDGES: "weakref.WeakValueDictionary[int, Bridge]" = weakref.WeakValueDictionary()


def close_all() -> None:
    """Close the HTTP client of every Bridge (stories/conftest.py, session finish)."""
    for b in list(_BRIDGES.values()):
        b.close()


def all_bridges() -> list[Bridge]:
    return [Bridge.for_agent(a) for a in PORTS]


def health_snapshot(label: str) -> dict[str, Any]:
    """Print (not assert) /health of every port; returns the raw bodies."""
    snap: dict[str, Any] = {}
    for b in all_bridges():
        h = b.health(timeout=5.0)
        snap[b.agent] = h.body if h.body is not None else {"unreachable": h.error or h.status}
        print(f"\n[{_ts()}] health {label} {b.agent}:{b.port} -> {snap[b.agent]}", flush=True)
    return snap
