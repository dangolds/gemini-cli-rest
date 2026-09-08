"""
Unit tests for the shared worktree helper.

These are pure, fast, hermetic tests — no real git, no tmux, no network. The
PURE functions (split_base / safe_name) are exercised directly. The async git
functions (resolve_base / add / remove / prune_stale) are tested by mocking the
single chokepoint they all funnel through — worktree._git — with an AsyncMock,
then asserting the git argv they BUILD. That pins the contract the servers rely
on (e.g. add must stay --detach so two callers can sit on the same commit, and
resolve_base must prefer origin/<base> so a bare name reads the freshest pushed
tip) without ever shelling out.

Run standalone:  .venv/bin/python -m pytest test_worktree.py -q
"""

import asyncio
import logging
import re
from unittest.mock import AsyncMock

import pytest

import worktree


def _run(coro):
    """Drive an async function from a sync test without pytest-asyncio."""
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _fresh_fetch_state(monkeypatch):
    """The fetch coalescer is module state (last-attempt clock + lock); start
    every test with 'never fetched' so ordering can't make a fetch get skipped."""
    monkeypatch.setattr(worktree, "_last_fetch", 0.0)
    monkeypatch.setattr(worktree, "_fetch_lock", None)
    monkeypatch.setattr(worktree, "FETCH_MIN_INTERVAL", 60.0)


# ---------------------------------------------------------------------------
# split_base — the gate that decides whether a caller named a branch at all
# ---------------------------------------------------------------------------

def test_split_base_no_at_signals_branchless():
    """No '@' -> base is None, which is how /chat knows to ask for a branch."""
    assert worktree.split_base("fix") == ("fix", None)


def test_split_base_simple_base():
    assert worktree.split_base("fix@dev") == ("fix", "dev")


def test_split_base_only_first_at_splits():
    """A base may itself contain '@' (e.g. a ref with '@'); only the FIRST '@'
    separates session from base, so the remainder is kept whole."""
    assert worktree.split_base("fix@origin/dev@2") == ("fix", "origin/dev@2")


def test_split_base_base_with_slashes_kept_whole():
    """Remote-style bases carry '/', which must survive into the base part."""
    assert worktree.split_base("fix@origin/dev") == ("fix", "origin/dev")
    assert worktree.split_base("a@feature/x/y/z") == ("a", "feature/x/y/z")


def test_split_base_empty_base_collapses_to_none():
    """A trailing '@' (or whitespace-only base) must not slip past the gate as a
    'present but empty' branch — it collapses to None like no '@' at all."""
    assert worktree.split_base("fix@") == ("fix", None)
    assert worktree.split_base("fix@   ") == ("fix", None)


def test_split_base_strips_surrounding_whitespace_in_base():
    assert worktree.split_base("fix@  dev  ") == ("fix", "dev")


# ---------------------------------------------------------------------------
# safe_name — must yield a token safe for tmux session names AND path parts
# ---------------------------------------------------------------------------

_SAFE_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def test_safe_name_matches_safe_charset():
    """Whatever junk goes in, the output is only chars that are safe in both a
    tmux session name and a single filesystem path component."""
    for raw in ["fix@origin/dev", "a@feature/x/y", "weird name!!", "@@@", "/", ""]:
        assert _SAFE_RE.match(worktree.safe_name(raw)), raw


def test_safe_name_is_deterministic():
    """Same key in -> same token out, every time (the server re-derives the
    tmux name / cwd from the key on each call and they must agree)."""
    assert worktree.safe_name("fix@origin/dev") == worktree.safe_name("fix@origin/dev")


def test_safe_name_distinct_keys_dont_collide():
    """Distinct keys that slug to the SAME readable prefix must still differ,
    because the 8-char hash of the FULL key is appended — otherwise two sessions
    would share one tmux session / worktree path. 'fix@dev', 'fix/dev' and
    'fix-dev' all slug to 'fix-dev', so they exercise exactly this case."""
    keys = ["fix@dev", "fix/dev", "fix-dev"]
    tokens = [worktree.safe_name(k) for k in keys]
    # All share the human-readable slug...
    assert {t.rsplit("-", 1)[0] for t in tokens} == {"fix-dev"}
    # ...but every token is unique thanks to the per-key hash suffix.
    assert len(set(tokens)) == len(keys)


def test_safe_name_ends_in_8_char_hex_hash():
    """The collision-resistant suffix is the first 8 hex chars of sha1(key)."""
    suffix = worktree.safe_name("anything@x").rsplit("-", 1)[1]
    assert re.match(r"^[0-9a-f]{8}$", suffix)


def test_safe_name_empty_input_still_valid():
    """An empty/garbage slug falls back to a placeholder so the token is never
    empty (an empty path component / tmux name would be a hard failure)."""
    out = worktree.safe_name("")
    assert _SAFE_RE.match(out)
    assert out.startswith("x-")


# ---------------------------------------------------------------------------
# resolve_base — fetches then resolves to the freshest ref, or raises
# ---------------------------------------------------------------------------

def _patch_git(monkeypatch, *, repo_ok=True, resolvable=()):
    """Install an AsyncMock for worktree._git that fakes repo_ok + rev-parse.

    `resolvable` is the set of refs whose 'rev-parse --verify' returns rc 0;
    every other rev-parse returns rc 1 (not found). fetch always succeeds.
    Returns the mock so a test can inspect the exact argv sequence built.
    """
    async def fake_git(*args, check=True):
        # repo_ok(): git rev-parse --git-dir
        if args[:2] == ("rev-parse", "--git-dir"):
            return (0 if repo_ok else 128, "")
        # ref existence probe: rev-parse --verify --quiet '<ref>^{commit}'
        if args[0] == "rev-parse" and "--verify" in args:
            spec = args[-1]  # e.g. 'origin/dev^{commit}'
            ref = spec[: -len("^{commit}")]
            return (0, "") if ref in resolvable else (1, "")
        # fetch / anything else: succeed quietly
        return (0, "")

    mock = AsyncMock(side_effect=fake_git)
    monkeypatch.setattr(worktree, "_git", mock)
    return mock


def test_resolve_base_prefers_origin_for_bare_name(monkeypatch):
    """A bare 'dev' resolves to 'origin/dev' (the freshest pushed tip) when the
    remote-tracking ref exists, in preference to a local 'dev'."""
    mock = _patch_git(monkeypatch, resolvable={"origin/dev", "dev"})
    assert _run(worktree.resolve_base("dev")) == "origin/dev"

    calls = [c.args for c in mock.await_args_list]
    # It fetched before resolving (freshness is intentional every spawn).
    assert ("fetch", "--all", "--prune", "--quiet") in calls
    # The FIRST rev-parse probe targeted origin/dev, not bare dev.
    first_probe = next(a for a in calls if a[0] == "rev-parse" and "--verify" in a)
    assert first_probe[-1] == "origin/dev^{commit}"


def test_resolve_base_falls_back_to_bare_when_no_remote(monkeypatch):
    """If origin/<base> does not exist, it falls back to the bare local ref."""
    mock = _patch_git(monkeypatch, resolvable={"dev"})
    assert _run(worktree.resolve_base("dev")) == "dev"

    probes = [c.args[-1] for c in mock.await_args_list
              if c.args[0] == "rev-parse" and "--verify" in c.args]
    # Tried origin/dev first, then bare dev.
    assert probes == ["origin/dev^{commit}", "dev^{commit}"]


def test_resolve_base_explicit_origin_not_double_prefixed(monkeypatch):
    """An explicit 'origin/dev' is used as given — never 'origin/origin/dev'."""
    mock = _patch_git(monkeypatch, resolvable={"origin/dev"})
    assert _run(worktree.resolve_base("origin/dev")) == "origin/dev"

    probes = [c.args[-1] for c in mock.await_args_list
              if c.args[0] == "rev-parse" and "--verify" in c.args]
    assert probes == ["origin/dev^{commit}"]


def test_resolve_base_raises_when_nothing_resolves(monkeypatch):
    """No candidate resolves -> RuntimeError naming the missing base, so the
    server can hand the caller a conversational 'name an existing branch'."""
    _patch_git(monkeypatch, resolvable=set())
    with pytest.raises(RuntimeError, match="not found"):
        _run(worktree.resolve_base("ghost"))


def test_resolve_base_raises_when_repo_absent(monkeypatch):
    """If WORKTREE_REPO is not a git clone, resolve_base raises before any
    fetch — a clear setup error, not a confusing 'branch not found'."""
    mock = _patch_git(monkeypatch, repo_ok=False)
    with pytest.raises(RuntimeError, match="not a git clone"):
        _run(worktree.resolve_base("dev"))
    # It bailed at repo_ok and never fetched.
    assert all(c.args[:1] != ("fetch",) for c in mock.await_args_list)


def test_resolve_base_fetch_timeout_warns_and_continues(monkeypatch, caplog):
    """A fetch that hit the git timeout (rc 124 from _git) must NOT raise out
    of resolve_base: it logs a WARNING carrying the reason and the base still
    resolves against the origin/* refs already in the clone."""
    mock = _patch_git(monkeypatch, resolvable={"origin/dev"})
    real = mock.side_effect

    async def fake_git(*args, check=True):
        if args[0] == "fetch":
            return (124, "timed out after 60s")
        return await real(*args, check=check)

    mock.side_effect = fake_git
    with caplog.at_level(logging.WARNING, logger="worktree"):
        assert _run(worktree.resolve_base("dev")) == "origin/dev"
    warn = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warn) == 1
    assert "timed out after 60s" in warn[0].getMessage()
    assert "rc=124" in warn[0].getMessage()


def test_resolve_base_fetch_failure_warns_with_first_line(monkeypatch, caplog):
    """Non-zero fetch exit (offline, auth, dead remote) -> WARNING with the
    first line of git's output and the elapsed time, then carry on."""
    mock = _patch_git(monkeypatch, resolvable={"origin/dev"})
    real = mock.side_effect

    async def fake_git(*args, check=True):
        if args[0] == "fetch":
            return (128, "ssh: connect to host github.com port 22: Connection timed out\n"
                         "fatal: Could not read from remote repository.\n")
        return await real(*args, check=check)

    mock.side_effect = fake_git
    with caplog.at_level(logging.WARNING, logger="worktree"):
        assert _run(worktree.resolve_base("dev")) == "origin/dev"
    msg = next(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
    assert "ssh: connect to host github.com port 22: Connection timed out" in msg
    assert "fatal:" not in msg  # only the first line
    assert re.search(r"after \d+\.\ds", msg)


def test_resolve_base_fetch_success_logs_info(monkeypatch, caplog):
    """One INFO line per ACTUAL fetch, naming the repo and the elapsed time."""
    _patch_git(monkeypatch, resolvable={"origin/dev"})
    with caplog.at_level(logging.INFO, logger="worktree"):
        _run(worktree.resolve_base("dev"))
    infos = [r.getMessage() for r in caplog.records
             if r.levelno == logging.INFO and "fetch" in r.getMessage()]
    assert len(infos) == 1
    assert str(worktree.WORKTREE_REPO) in infos[0]
    assert re.search(r"in \d+\.\ds", infos[0])


# ---------------------------------------------------------------------------
# fetch coalescing — one fetch per interval, shared by concurrent spawns
# ---------------------------------------------------------------------------

def _fetch_calls(mock):
    return [c.args for c in mock.await_args_list if c.args[:1] == ("fetch",)]


def _fake_clock(monkeypatch, clock):
    """Replace only worktree's view of time.monotonic. Patching the global
    time module would freeze asyncio's loop clock too and hang every sleep."""
    from types import SimpleNamespace
    monkeypatch.setattr(worktree, "time", SimpleNamespace(monotonic=lambda: clock[0]))


def test_concurrent_resolves_share_one_fetch(monkeypatch):
    """Two spawns arriving together (both bridges do this) -> ONE fetch; the
    second waits on the lock and reuses the result instead of racing git."""
    mock = _patch_git(monkeypatch, resolvable={"origin/dev", "origin/main"})
    real = mock.side_effect

    async def slow_git(*args, check=True):
        if args[0] == "fetch":
            await asyncio.sleep(0.05)  # long enough for the 2nd caller to queue
        return await real(*args, check=check)

    mock.side_effect = slow_git

    async def both():
        return await asyncio.gather(
            worktree.resolve_base("dev"), worktree.resolve_base("main")
        )

    assert _run(both()) == ["origin/dev", "origin/main"]
    assert len(_fetch_calls(mock)) == 1


def test_fetch_skipped_within_interval_then_repeats_after(monkeypatch, caplog):
    """A call inside WORKTREE_FETCH_MIN_INTERVAL skips the fetch (DEBUG line);
    once the interval has elapsed the next call fetches again."""
    mock = _patch_git(monkeypatch, resolvable={"origin/dev"})
    clock = [1000.0]
    _fake_clock(monkeypatch, clock)
    monkeypatch.setattr(worktree, "FETCH_MIN_INTERVAL", 60.0)

    with caplog.at_level(logging.DEBUG, logger="worktree"):
        _run(worktree.resolve_base("dev"))          # t=1000: fetches
        clock[0] += 30
        _run(worktree.resolve_base("dev"))          # t=1030: inside interval, skipped
        assert len(_fetch_calls(mock)) == 1
        assert any("fetch skipped" in r.getMessage() and r.levelno == logging.DEBUG
                   for r in caplog.records)
        clock[0] += 31
        _run(worktree.resolve_base("dev"))          # t=1061: interval over, fetch again
    assert len(_fetch_calls(mock)) == 2


def test_waiters_reuse_a_timed_out_fetch(monkeypatch):
    """The interval is stamped when the fetch ENDS: a caller that queued on the
    lock during a fetch that ran the full timeout must reuse it, not fire a
    second 60s fetch of its own (timeout == interval by default)."""
    mock = _patch_git(monkeypatch, resolvable={"origin/dev"})
    real = mock.side_effect
    clock = [1000.0]
    _fake_clock(monkeypatch, clock)
    monkeypatch.setattr(worktree, "FETCH_MIN_INTERVAL", 60.0)

    async def slow_timeout_git(*args, check=True):
        if args[0] == "fetch":
            await asyncio.sleep(0.05)
            clock[0] += 60  # the fetch burned the whole 60s timeout
            return (124, "timed out after 60s")
        return await real(*args, check=check)

    mock.side_effect = slow_timeout_git

    async def both():
        return await asyncio.gather(
            worktree.resolve_base("dev"), worktree.resolve_base("dev")
        )

    _run(both())
    assert len(_fetch_calls(mock)) == 1


def test_failed_fetch_still_counts_toward_interval(monkeypatch):
    """A failed/timed-out fetch marks the interval too, so a dead remote costs
    one timeout per interval rather than one per spawn."""
    mock = _patch_git(monkeypatch, resolvable={"origin/dev"})
    real = mock.side_effect

    async def failing_git(*args, check=True):
        if args[0] == "fetch":
            return (124, "timed out after 60s")
        return await real(*args, check=check)

    mock.side_effect = failing_git
    _run(worktree.resolve_base("dev"))
    _run(worktree.resolve_base("dev"))
    assert len(_fetch_calls(mock)) == 1


# ---------------------------------------------------------------------------
# _git timeout — the only test that spawns a real (non-git) subprocess
# ---------------------------------------------------------------------------

def _patch_subprocess_as_sleep(monkeypatch, seen):
    """Make worktree._git spawn `sleep 30` instead of git, keeping the same
    kwargs, and stash the Process so the test can check it was killed."""
    real_exec = asyncio.create_subprocess_exec

    async def fake_exec(*argv, **kw):
        proc = await real_exec("sleep", "30", **kw)
        seen.append(proc)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)


def test_git_timeout_kills_process_and_returns_rc_124(monkeypatch, caplog):
    seen = []
    _patch_subprocess_as_sleep(monkeypatch, seen)
    monkeypatch.setattr(worktree, "GIT_TIMEOUT", 0.2)

    with caplog.at_level(logging.WARNING, logger="worktree"):
        rc, out = _run(worktree._git("fetch", "--all", check=False))
    assert rc == 124
    assert "timed out after 0s" in out
    assert seen[0].returncode == -9  # SIGKILLed, and reaped (returncode set)
    assert any("timed out" in r.getMessage() for r in caplog.records)


def test_git_caller_cancelled_kills_process(monkeypatch):
    """If whoever awaits _git is cancelled (request torn down), the child must
    not be left running detached: kill + reap, then the cancellation propagates."""
    seen = []
    _patch_subprocess_as_sleep(monkeypatch, seen)
    monkeypatch.setattr(worktree, "GIT_TIMEOUT", 60)

    async def scenario():
        task = asyncio.create_task(worktree._git("fetch", check=False))
        await asyncio.sleep(0.1)  # let it spawn
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    _run(scenario())
    assert seen[0].returncode == -9


def test_git_timeout_raises_when_check(monkeypatch):
    """check=True callers (add) get the normal RuntimeError, message says why."""
    _patch_subprocess_as_sleep(monkeypatch, [])
    monkeypatch.setattr(worktree, "GIT_TIMEOUT", 0.2)
    with pytest.raises(RuntimeError, match="timed out"):
        _run(worktree._git("worktree", "add", "x"))


# ---------------------------------------------------------------------------
# env knobs
# ---------------------------------------------------------------------------

def test_env_seconds_parses_and_falls_back(monkeypatch, caplog):
    monkeypatch.setenv("WORKTREE_GIT_TIMEOUT", "15")
    assert worktree._env_seconds("WORKTREE_GIT_TIMEOUT", 60) == 15.0
    monkeypatch.setenv("WORKTREE_FETCH_MIN_INTERVAL", "0.5")
    assert worktree._env_seconds("WORKTREE_FETCH_MIN_INTERVAL", 60) == 0.5
    monkeypatch.delenv("WORKTREE_GIT_TIMEOUT")
    assert worktree._env_seconds("WORKTREE_GIT_TIMEOUT", 60) == 60
    monkeypatch.setenv("WORKTREE_GIT_TIMEOUT", "soon")
    with caplog.at_level(logging.WARNING, logger="worktree"):
        assert worktree._env_seconds("WORKTREE_GIT_TIMEOUT", 60) == 60
    assert any("not a number" in r.getMessage() for r in caplog.records)


def test_env_knob_defaults_are_60s():
    assert worktree.GIT_TIMEOUT == 60
    assert worktree.FETCH_MIN_INTERVAL == 60


# ---------------------------------------------------------------------------
# add — creates the DETACHED, --force worktree at cwd
# ---------------------------------------------------------------------------

def test_add_builds_detached_force_worktree(monkeypatch, tmp_path):
    mock = AsyncMock(return_value=(0, ""))
    monkeypatch.setattr(worktree, "_git", mock)

    cwd = tmp_path / "sessions" / "fix-deadbeef" / "c0"
    _run(worktree.add(cwd, "origin/dev"))

    mock.assert_awaited_once_with(
        "worktree", "add", "--detach", "--force", str(cwd), "origin/dev"
    )


def test_add_creates_parent_dir(monkeypatch, tmp_path):
    """git worktree add needs the parent to exist; add() must create it so the
    server doesn't have to pre-make the SESSIONS_ROOT subtree."""
    monkeypatch.setattr(worktree, "_git", AsyncMock(return_value=(0, "")))
    cwd = tmp_path / "deep" / "nested" / "c0"
    assert not cwd.parent.exists()
    _run(worktree.add(cwd, "dev"))
    assert cwd.parent.exists()


# ---------------------------------------------------------------------------
# remove — best-effort, idempotent teardown
# ---------------------------------------------------------------------------

def test_remove_uses_force_then_prune_non_fatal(monkeypatch, tmp_path):
    """remove() must force-remove THEN prune, and both with check=False so a
    crash/reset/double-delete on an already-gone checkout is a no-op, not a
    500 (it's called from reset, clear, manager.remove, stop_all)."""
    mock = AsyncMock(return_value=(0, ""))
    monkeypatch.setattr(worktree, "_git", mock)

    cwd = tmp_path / "fix-deadbeef" / "c0"
    _run(worktree.remove(cwd))

    assert mock.await_args_list[0].args == (
        "worktree", "remove", "--force", str(cwd),
    )
    assert mock.await_args_list[0].kwargs == {"check": False}
    assert mock.await_args_list[1].args == ("worktree", "prune")
    assert mock.await_args_list[1].kwargs == {"check": False}


# ---------------------------------------------------------------------------
# prune_stale — startup hygiene, guarded on repo presence
# ---------------------------------------------------------------------------

def test_prune_stale_prunes_when_repo_present(monkeypatch):
    async def fake_git(*args, check=True):
        if args[:2] == ("rev-parse", "--git-dir"):
            return (0, "")  # repo_ok True
        return (0, "")

    mock = AsyncMock(side_effect=fake_git)
    monkeypatch.setattr(worktree, "_git", mock)

    _run(worktree.prune_stale())

    calls = [c.args for c in mock.await_args_list]
    assert ("worktree", "prune") in calls


def test_prune_stale_skips_when_repo_absent(monkeypatch):
    """On a container with no clone yet, prune_stale must NOT try to prune (it
    would error); it just logs and returns so startup proceeds."""
    async def fake_git(*args, check=True):
        if args[:2] == ("rev-parse", "--git-dir"):
            return (128, "")  # repo_ok False
        return (0, "")

    mock = AsyncMock(side_effect=fake_git)
    monkeypatch.setattr(worktree, "_git", mock)

    _run(worktree.prune_stale())

    # Only the repo_ok probe ran; no 'worktree prune' was attempted.
    calls = [c.args for c in mock.await_args_list]
    assert all(c != ("worktree", "prune") for c in calls)
