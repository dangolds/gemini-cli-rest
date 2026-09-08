"""Shared git-worktree backing for bridge sessions.

Each bridge session (one warm, READ-ONLY CLI agent) runs inside its own
*detached* git worktree pinned to a caller-named branch, instead of an empty
scratch dir. The agents only ever READ code — they never edit, commit, or push
— so the worktree is purely a per-session read surface that lets many callers
ask questions about many different branches at the same time without colliding.

Detached on purpose: git forbids checking out the same branch in two worktrees,
which would block two callers both asking about `dev`. A detached worktree
(`--detach`, no branch attached) sidesteps that — any number of them can sit on
the same commit. Because nothing is ever written back, reset/delete just remove
the checkout; there is no branch to clean up and nothing to preserve.

Repo-agnostic: server.py (agy) and codex_server.py (codex) both import it, and
any future bridge can too. The only repo it ever mutates is WORKTREE_REPO; the
checkouts live wherever the caller passes (the servers put them under their
ephemeral SESSIONS_ROOT, so a container restart wipes the checkouts while the
repo's worktree registry survives in the named volume — which is why
prune_stale() exists).
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import signal
import time
from pathlib import Path as FsPath

logger = logging.getLogger("worktree")

# The repo worktrees are cut from. Inside the bridge container this is the
# persisted slitled-platform clone (docker-compose named volume at /app/...).
WORKTREE_REPO = FsPath(os.getenv("WORKTREE_REPO", "/app/slitled-platform"))


def _env_seconds(name: str, default: float) -> float:
    """Parse a seconds knob from the environment; garbage falls back to the
    default (with a warning) rather than crashing the bridge at import."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", name, raw, default)
        return default


# Wall-clock cap on EVERY git call (seconds; <= 0 disables). The fetch is the
# one that matters: ssh can stall for hours on a sleeping host or dead link,
# and one hung fetch queued every later spawn on both bridges behind it (a
# request once sat ~19h). On expiry the child is killed and the call fails
# with a clear "timed out" instead of hanging.
GIT_TIMEOUT = _env_seconds("WORKTREE_GIT_TIMEOUT", 60)
# Coalesce fetches: at most one per this many seconds (success or failure),
# so concurrent spawns share one fetch instead of racing for .git locks, and
# a dead remote costs one timeout per interval rather than one per spawn.
# The counter is per process — agy and codex are separate processes, so the
# pair may fetch twice per interval, but never concurrently from one bridge.
FETCH_MIN_INTERVAL = _env_seconds("WORKTREE_FETCH_MIN_INTERVAL", 60)
_TIMEOUT_RC = 124  # coreutils `timeout` convention for "killed on expiry"

# Returned (after .format(name=...)) to a caller who addresses /chat/<name>
# with no @<branch>. Phrased as the agent asking the question, because that is
# how the calling agent will read and act on it.
NEEDS_BRANCH_MSG = (
    "Which branch are we working on? I won't start until you name one.\n"
    "Re-send to /chat/{name}@<branch> — for example /chat/{name}@dev to read "
    "the 'dev' branch, or /chat/{name}@origin/dev for the shared base."
)


def split_base(name: str) -> tuple[str, str | None]:
    """Split a session key '<session>@<base>' into its parts.

    No '@' -> (name, None): the signal that the caller never named a branch.
    Only the FIRST '@' splits, so a base may itself contain '/' (and, rarely,
    '@'): 'fix@origin/dev' -> ('fix', 'origin/dev'). An empty base ('fix@',
    'fix@ ') also collapses to None, so a trailing '@' can't slip past the gate.
    """
    session, sep, base = name.partition("@")
    if not sep:
        return name, None
    base = base.strip()
    return session, base or None


def safe_name(name: str) -> str:
    """A filesystem- and tmux-safe token for a session key.

    Session keys now contain '@' and '/' (from bases like origin/dev), which are
    unsafe in tmux session names and would fragment a filesystem path into
    nested dirs. We slug the readable part for humans/logs and append a short
    hash of the FULL key so two distinct keys that happen to slug alike can't
    collide onto the same tmux session or worktree path.
    """
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._") or "x"
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{digest}"


async def _git(*args: str, check: bool = True) -> tuple[int, str]:
    """Run a git command against WORKTREE_REPO; return (rc, combined output)."""
    proc = await asyncio.create_subprocess_exec(
        "git", "-C", str(WORKTREE_REPO), *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,  # own process group: a kill also reaps git's ssh child
    )
    try:
        out, _ = await asyncio.wait_for(
            proc.communicate(), timeout=GIT_TIMEOUT if GIT_TIMEOUT > 0 else None
        )
    except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
        # Timeout, or our caller was cancelled: either way don't leave git (or
        # the ssh it spawned) running. Kill the whole group — SIGKILL to git
        # alone would orphan the ssh / remote-helper that is actually stuck.
        for kill in (lambda: os.killpg(proc.pid, signal.SIGKILL), proc.kill):
            try:
                kill()
            except ProcessLookupError:
                pass
        # Drain + reap via communicate(), not a bare wait(): wait() only
        # resolves once every pipe hit EOF, and a reader paused on a full
        # buffer (its drain task was just cancelled) would never read it.
        # Bounded so a truly wedged pipe costs seconds, never another hang.
        try:
            await asyncio.wait_for(proc.communicate(), timeout=5)
        except Exception:
            logger.warning("git %s: killed but not reaped within 5s", " ".join(args))
        if isinstance(exc, asyncio.CancelledError):
            raise
        logger.warning("git %s timed out after %.0fs — killed", " ".join(args), GIT_TIMEOUT)
        rc, text = _TIMEOUT_RC, f"timed out after {GIT_TIMEOUT:.0f}s"
    else:
        text = out.decode("utf-8", "replace")
        rc = proc.returncode or 0
    if check and rc != 0:
        raise RuntimeError(f"git {' '.join(args)} failed (rc={rc}): {text.strip()}")
    return rc, text


async def repo_ok() -> bool:
    rc, _ = await _git("rev-parse", "--git-dir", check=False)
    return rc == 0


_fetch_lock: asyncio.Lock | None = None
_fetch_lock_loop: asyncio.AbstractEventLoop | None = None
_last_fetch: float = 0.0  # time.monotonic() when the last fetch attempt ENDED; 0 = never


def _get_fetch_lock() -> asyncio.Lock:
    """One lock per event loop (a Lock binds to the loop that first contends
    it; the bridges run one loop for life, tests spin a new one per case)."""
    global _fetch_lock, _fetch_lock_loop
    loop = asyncio.get_running_loop()
    if _fetch_lock is None or _fetch_lock_loop is not loop:
        _fetch_lock, _fetch_lock_loop = asyncio.Lock(), loop
    return _fetch_lock


async def _fetch() -> None:
    """Refresh origin/* — at most once per FETCH_MIN_INTERVAL, never two at
    once. Callers arriving mid-fetch wait on the lock and then find it fresh;
    callers within the interval skip. Failure (timeout, offline, auth) is a
    WARNING, not an error: the caller carries on with whatever origin/<base>
    the clone already has, and resolve_base's own check reports a truly
    missing ref."""
    global _last_fetch
    async with _get_fetch_lock():
        now = time.monotonic()
        if _last_fetch and now - _last_fetch < FETCH_MIN_INTERVAL:
            logger.debug(
                "worktree fetch skipped: last attempt %.0fs ago (< %.0fs) repo=%s",
                now - _last_fetch, FETCH_MIN_INTERVAL, WORKTREE_REPO,
            )
            return
        rc, out = await _git("fetch", "--all", "--prune", "--quiet", check=False)
        # Stamp at completion (success or not): callers that queued on the lock
        # during a 60s timeout must reuse this attempt, not start another.
        _last_fetch = time.monotonic()
        elapsed = _last_fetch - now
        if rc != 0:
            lines = out.strip().splitlines()
            logger.warning(
                "worktree fetch failed after %.1fs (rc=%d): %s — continuing with "
                "the origin/* refs already in %s",
                elapsed, rc, lines[0] if lines else "(no output)", WORKTREE_REPO,
            )
        else:
            logger.info("worktree fetch %s ok in %.1fs", WORKTREE_REPO, elapsed)


async def resolve_base(base: str) -> str:
    """Fetch, then resolve <base> to the freshest ref to check out. Raise if it
    does not resolve, so the caller can surface a clear message instead of
    spawning against a branch that does not exist.

    A bare name like 'dev' prefers the remote-tracking 'origin/dev' (the latest
    pushed tip), falling back to a local 'dev'. An explicit 'origin/dev', a SHA,
    or any ref already naming a remote is used as given.
    """
    if not await repo_ok():
        raise RuntimeError(
            f"Worktree repo {WORKTREE_REPO} is not a git clone. Clone the target "
            "repo into it before starting branch sessions."
        )
    # Refresh so origin/* are current; offline / stalled / no-remote is
    # non-fatal (logged, then we use what the clone already has).
    await _fetch()
    candidates: list[str] = []
    if not base.startswith("origin/"):
        candidates.append(f"origin/{base}")
    candidates.append(base)
    for ref in candidates:
        rc, _ = await _git(
            "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", check=False
        )
        if rc == 0:
            return ref
    raise RuntimeError(
        f"Base branch '{base}' not found in {WORKTREE_REPO}. Name an existing "
        "local or remote branch (e.g. dev, origin/dev, a feature branch, or a SHA)."
    )


async def add(cwd: FsPath, ref: str) -> None:
    """Create a DETACHED worktree at <cwd> pinned to <ref> (read-only surface)."""
    cwd.parent.mkdir(parents=True, exist_ok=True)
    # --detach: no branch attached, so many worktrees can share one commit.
    # --force: reuse a path the registry still half-remembers after a crash.
    await _git("worktree", "add", "--detach", "--force", str(cwd), ref)
    logger.info("worktree add --detach %s @ %s", cwd, ref)


async def remove(cwd: FsPath) -> None:
    """Tear down a worktree. Best-effort & idempotent — safe on an already-gone
    checkout (reset, delete, crash recovery). No branch to delete (detached)."""
    await _git("worktree", "remove", "--force", str(cwd), check=False)
    await _git("worktree", "prune", check=False)
    # `git worktree remove` deletes the cN checkout but leaves the empty slug
    # parent dir (SESSIONS_ROOT/<safe_name>/) behind; drop it so /tmp doesn't
    # accumulate empty shells. Best-effort: rmdir only removes truly-empty dirs,
    # so a still-live sibling generation is left untouched.
    for d in (cwd, cwd.parent):
        try:
            d.rmdir()
        except OSError:
            pass
    logger.info("worktree remove %s", cwd)


async def prune_stale() -> None:
    """Startup hygiene: drop registry entries whose checkouts vanished (the
    ephemeral SESSIONS_ROOT under /tmp is wiped on container restart while the
    repo's worktree registry survives in the named volume). Detached worktrees
    leave no branches behind, so a prune is all that's needed."""
    if not await repo_ok():
        logger.warning("worktree repo %s absent — skipping prune_stale", WORKTREE_REPO)
        return
    await _git("worktree", "prune", check=False)
    logger.info("worktree prune_stale complete (repo=%s)", WORKTREE_REPO)
