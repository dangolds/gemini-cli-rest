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

import re
from unittest.mock import AsyncMock

import pytest

import worktree


def _run(coro):
    """Drive an async function from a sync test without pytest-asyncio."""
    import asyncio

    return asyncio.run(coro)


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
