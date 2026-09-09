"""The one session-name helper (TestPRD Rule 2 / FR-3).

Every story builds its session name here, in one of three forms:

    key(name, base="main")  -> "<prefix><name>-<STAMP>@<base>"   (live sessions)
    bare(name)              -> "<prefix><name>-<STAMP>"          (no-base management stories)
    raw(text)               -> text, untouched                   (hermetic grammar probes ONLY)

STAMP is one short token per process: random by default, or BRIDGE_RUN_STAMP
plus a 4-hex nonce ("r1" -> "r1-ab12") so two processes pinned to the same
stamp never share one. Two runs, or a run beside the operator's own sessions,
never collide, and teardown can tell "ours" from "not ours" (is_ours is exact;
matches_stamp also accepts the pinned prefix, so the operator finds every
session of "r1" by "-r1-").

REPO_PREFIX is the single line that changes with PRD-multi-repo.md ("" today,
"slitled/" then). Nothing else in the suite spells a key by hand.
"""
from __future__ import annotations

import os
import re
import secrets

# --- the migration line ------------------------------------------------------
REPO_PREFIX = ""  # becomes "slitled/" with the repository change; touch nothing else

# --- the run stamp -----------------------------------------------------------
_PINNED: str = os.environ.get("BRIDGE_RUN_STAMP", "").strip()
STAMP: str = f"{_PINNED}-{secrets.token_hex(2)}" if _PINNED else f"r{secrets.token_hex(3)}"


def bare(name: str) -> str:
    """Name part only: '<prefix><name>-<STAMP>' (equals the name part of key())."""
    return f"{REPO_PREFIX}{name}-{STAMP}"


def key(name: str, base: str | None = "main") -> str:
    """Full session key '<prefix><name>-<STAMP>@<base>'.

    base may carry a slash ('origin/main'); an empty/None base means
    branchless and yields the bare form (no '@' at all).
    """
    n = bare(name)
    if not base:
        return n
    return f"{n}@{base}"


def raw(text: str) -> str:
    """Return *text* unchanged.

    For hermetic grammar probes only (an empty name, '@@', whitespace, ...):
    such a name carries no stamp, so it must NEVER reach a live bridge.
    """
    return text


def name_part(session_name: str) -> str:
    """The part before the first '@' (worktree.split_base splits the same way)."""
    return session_name.partition("@")[0]


def matches_stamp(session_name: str, stamp: str) -> bool:
    """Does *session_name* (key or bare) carry *stamp* the way bare()/key() put
    it - exactly ("-r1-ab12"), or as the pinned prefix of a nonced stamp
    ("-r1" matches "-r1-ab12")?"""
    if not stamp:
        return False
    return re.search(rf"-{re.escape(stamp)}(-[0-9a-f]{{4}})?$", name_part(session_name)) is not None


def is_ours(session_name: str) -> bool:
    """Does *session_name* carry THIS process's stamp, exactly (no prefix form)?"""
    return name_part(session_name).endswith(f"-{STAMP}")
