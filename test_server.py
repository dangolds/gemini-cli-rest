"""
E2E tests for the Antigravity CLI REST Bridge.

Starts the server as a subprocess (or uses an existing one),
runs real agy sessions, and verifies memory, clear, reset,
isolation, delete, and stop.

Usage:
    pytest test_server.py -v
"""

import time
import uuid
from datetime import datetime

import httpx
import pytest


def _ts() -> str:
    """Short timestamp for progress output."""
    return datetime.now().strftime("%H:%M:%S")

BASE = "http://127.0.0.1:8000"
# agy can be slow — first turn spawns the CLI (~25s) + model response time
TIMEOUT = 180.0


def _unique(prefix: str = "xq") -> str:
    """Generate a unique nonsense token the model can't guess."""
    return f"{prefix}{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Server fixture — starts before all tests, stops after
# ---------------------------------------------------------------------------

def _server_already_running() -> bool:
    """Check if a server is already listening on BASE."""
    try:
        r = httpx.get(f"{BASE}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# NOTE: named `live_server`, NOT `server` — the hermetic unit section below does
# `import server` (the module), which would shadow a fixture named `server` and
# silently disable the live-bridge gate + /stop teardown for the whole file.
@pytest.fixture(autouse=True)
def live_server(request):
    """Gate the E2E tests on a live bridge at BASE; tear sessions down after.

    Hermetic unit classes mock everything and need no bridge, so they return
    early with NO gate. Otherwise, a missing bridge SKIPS (not fails) so the host
    suite stays green whether or not `docker compose up -d` is running.
    """
    # Hermetic unit classes never touch the network — run them unconditionally.
    if request.cls in (TestSpawnWorktree, TestChatBranchGating,
                       TestWorktreeTeardown, TestNameRegex):
        yield
        return

    print(f"\n[{_ts()}] Checking for server at {BASE}...", flush=True)
    if not _server_already_running():
        pytest.skip(f"No bridge at {BASE}; run docker compose up -d")
    print(f"[{_ts()}] Server is up and healthy", flush=True)
    yield
    print(f"\n[{_ts()}] Tearing down — stopping all sessions...", flush=True)
    try:
        httpx.post(f"{BASE}/stop", timeout=10)
    except Exception:
        pass
    print(f"[{_ts()}] Done", flush=True)


def chat(session: str, prompt: str) -> dict:
    """Send a chat message and return the JSON response."""
    # Sessions now spawn inside a worktree keyed "<name>@<base>"; a bare name
    # hits the branchless handshake instead of spawning. Default-base any plain
    # name to "@main" so E2E sessions actually start.
    if "@" not in session:
        session = f"{session}@main"
    print(f"  [{_ts()}] >>> {session}: {prompt}", flush=True)
    t0 = time.monotonic()
    r = httpx.post(
        f"{BASE}/chat/{session}",
        json={"prompt": prompt},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    elapsed = time.monotonic() - t0
    print(f"  [{_ts()}] <<< {session} (turn {data['turn']}, {elapsed:.1f}s): {data['response']}", flush=True)
    return data


def last(session: str, wait: float = 15.0) -> dict:
    """GET /last/{session}?wait=N and return the JSON response."""
    # Same default-basing as chat(): a bare name addresses the wrong key.
    if "@" not in session:
        session = f"{session}@main"
    r = httpx.get(f"{BASE}/last/{session}", params={"wait": wait}, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    print(f"  [{_ts()}] /last {session}: done={data['done']} turn={data['turn']} "
          f"resp={data['response']!r}", flush=True)
    return data


# ---------------------------------------------------------------------------
# Cleanup: each test that creates sessions cleans up after itself
# ---------------------------------------------------------------------------

@pytest.fixture()
def cleanup():
    """Collects session names to delete after test."""
    sessions = []
    yield sessions
    for name in sessions:
        # chat() default-bases bare names to "@main"; mirror that so the DELETE
        # URL addresses the SAME key the session was actually created under.
        if "@" not in name:
            name = f"{name}@main"
        print(f"  [{_ts()}] Cleaning up session '{name}'...", flush=True)
        try:
            httpx.delete(f"{BASE}/chat/{name}", timeout=30)
        except Exception as e:
            print(f"  [{_ts()}] WARNING: cleanup of '{name}' failed: {e}", flush=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChatMemory:
    """Verify that a session remembers context across turns."""

    def test_remembers_fact(self, cleanup):
        print(f"\n[{_ts()}] TEST: Chat memory — does session remember a secret word?", flush=True)
        # Keyed: the response's "session" field echoes the full "<name>@<base>" key.
        name = f"test-memory-{uuid.uuid4().hex[:8]}@main"
        cleanup.append(name)
        token = _unique()

        # Tell it a fact (instruct not to persist anywhere)
        print(f"  [{_ts()}] Step 1/2: Teaching secret word '{token}'...", flush=True)
        chat(name, f"For this conversation only, my secret word is {token}. Do NOT save this to memory, knowledge, or any file. Just confirm you got it.")

        # Ask for it back
        print(f"  [{_ts()}] Step 2/2: Asking for the word back...", flush=True)
        resp = chat(name, "What is my secret word? Reply with just the word.")
        assert token in resp["response"].lower(), (
            f"Expected '{token}' in response, got: {resp['response']!r}"
        )
        assert resp["session"] == name
        assert resp["turn"] == 2
        print(f"  [{_ts()}] PASS: Session correctly remembered '{token}'", flush=True)


class TestLastReadback:
    """Verify the /chat -> /last flow: re-read a completed answer without re-asking."""

    def test_last_recovers_the_completed_answer(self, cleanup):
        print(f"\n[{_ts()}] TEST: /last — recover the answer a /chat just produced", flush=True)
        # Keyed: /last echoes the full "<name>@<base>" key in its "session" field.
        name = f"test-last-{uuid.uuid4().hex[:8]}@main"
        cleanup.append(name)
        token = _unique()

        sent = chat(name, f"Reply with exactly this token and nothing else: {token}")
        got = last(name, wait=15)

        assert got["done"] is True, f"expected done=True, got {got}"
        assert got["session"] == name
        assert got["turn"] == sent["turn"]
        assert token in (got["response"] or "").lower(), (
            f"/last should return the same answer the turn produced, got: {got['response']!r}"
        )
        print(f"  [{_ts()}] PASS: /last returned the completed answer", flush=True)

    def test_last_returns_latest_turn_not_a_previous_one(self, cleanup):
        """The baseline guard, over real HTTP: after a 2nd turn, /last is turn 2 — never turn 1."""
        print(f"\n[{_ts()}] TEST: /last — returns the LATEST turn, not a stale previous one", flush=True)
        name = f"test-last2-{uuid.uuid4().hex[:8]}"
        cleanup.append(name)
        tok1 = _unique("firstx")
        tok2 = _unique("secndx")

        chat(name, f"Reply with exactly this token and nothing else: {tok1}")
        sent2 = chat(name, f"Reply with exactly this token and nothing else: {tok2}")
        got = last(name, wait=15)

        assert got["done"] is True
        assert got["turn"] == sent2["turn"] == 2
        resp = (got["response"] or "").lower()
        assert tok2 in resp, f"expected the latest turn's token {tok2!r}, got: {got['response']!r}"
        assert tok1 not in resp, f"/last must NOT return the previous turn's answer: {got['response']!r}"
        print(f"  [{_ts()}] PASS: /last returned turn 2's answer, not turn 1's", flush=True)

    def test_last_after_clear_does_not_return_precleared_answer(self, cleanup):
        """/clear respawns the session; /last must not surface the pre-clear answer."""
        print(f"\n[{_ts()}] TEST: /last after /clear → no stale pre-clear answer", flush=True)
        # Keyed: chat()/last() leave an already-@based name alone, and the direct
        # /clear URL below must hit the SAME key.
        name = f"test-lastclr-{uuid.uuid4().hex[:8]}@main"
        cleanup.append(name)
        token = _unique()

        chat(name, f"Reply with exactly this token and nothing else: {token}")
        r = httpx.post(f"{BASE}/clear/{name}", timeout=60)
        assert r.status_code == 200

        got = last(name, wait=2)  # no new turn since clear → nothing to return
        assert got["done"] is False, f"expected done=False after clear, got {got}"
        assert got["response"] is None
        print(f"  [{_ts()}] PASS: /last returns done=False after a clear", flush=True)

    def test_last_unknown_session_returns_404(self):
        print(f"\n[{_ts()}] TEST: /last nonexistent session → expect 404", flush=True)
        r = httpx.get(f"{BASE}/last/nonexistent-session-xyz", timeout=10)
        assert r.status_code == 404
        print(f"  [{_ts()}] PASS: Got 404 as expected", flush=True)

    def test_last_invalid_name_returns_422(self):
        print(f"\n[{_ts()}] TEST: /last invalid session name → expect 422", flush=True)
        r = httpx.get(f"{BASE}/last/bad name!", timeout=10)
        assert r.status_code == 422
        print(f"  [{_ts()}] PASS: Got 422 as expected", flush=True)


class TestClear:
    """Verify that /clear wipes conversation context."""

    def test_clear_forgets_context(self, cleanup):
        print(f"\n[{_ts()}] TEST: Clear — does /clear wipe conversation context?", flush=True)
        # Keyed: the direct /clear URL below must address the same key chat() uses.
        name = f"test-clear-{uuid.uuid4().hex[:8]}@main"
        cleanup.append(name)
        token = _unique()

        # Establish context (instruct not to persist anywhere)
        print(f"  [{_ts()}] Step 1/3: Teaching secret code '{token}'...", flush=True)
        chat(name, f"For this conversation only, my secret code is {token}. Do NOT save this to memory, knowledge, or any file. Just confirm you got it.")

        # Clear — should succeed
        print(f"  [{_ts()}] Step 2/3: Sending /clear...", flush=True)
        r = httpx.post(f"{BASE}/clear/{name}", timeout=60)
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        print(f"  [{_ts()}] Clear returned OK", flush=True)

        # Ask for it — should NOT know (context wiped)
        print(f"  [{_ts()}] Step 3/3: Verifying context was wiped...", flush=True)
        resp = chat(name, "What is my secret code? If you don't know, say 'no idea'.")
        assert token not in resp["response"].lower(), (
            f"Expected context to be cleared, but got: {resp['response']}"
        )
        assert resp["turn"] == 1, (
            f"Expected turn 1 after /clear, got turn {resp['turn']}"
        )
        print(f"  [{_ts()}] PASS: Context was properly cleared", flush=True)

    def test_clear_nonexistent_returns_404(self):
        print(f"\n[{_ts()}] TEST: Clear nonexistent session → expect 404", flush=True)
        r = httpx.post(f"{BASE}/clear/nonexistent-session-xyz", timeout=10)
        assert r.status_code == 404
        print(f"  [{_ts()}] PASS: Got 404 as expected", flush=True)


class TestReset:
    """Verify that /reset kills and restarts the session."""

    def test_reset_returns_200_and_resets_turn_count(self, cleanup):
        print(f"\n[{_ts()}] TEST: Reset — does /reset kill and restart with fresh turn count?", flush=True)
        # Keyed: the direct /reset URL below must address the same key chat() uses.
        name = f"test-reset-{uuid.uuid4().hex[:8]}@main"
        cleanup.append(name)

        # Build up some turns
        print(f"  [{_ts()}] Step 1/3: Building up turns...", flush=True)
        chat(name, "Hello.")
        resp2 = chat(name, "Say 'pong'.")
        assert resp2["turn"] == 2

        # Reset
        print(f"  [{_ts()}] Step 2/3: Sending /reset...", flush=True)
        r = httpx.post(f"{BASE}/reset/{name}", timeout=60)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        print(f"  [{_ts()}] Reset returned OK", flush=True)

        # Next chat should be turn 1 (fresh process)
        print(f"  [{_ts()}] Step 3/3: Verifying turn count reset to 1...", flush=True)
        resp3 = chat(name, "Say 'hello'.")
        assert resp3["turn"] == 1
        assert resp3["response"]
        print(f"  [{_ts()}] PASS: Turn count reset to 1 after /reset", flush=True)

    def test_reset_nonexistent_returns_404(self):
        print(f"\n[{_ts()}] TEST: Reset nonexistent session → expect 404", flush=True)
        r = httpx.post(f"{BASE}/reset/nonexistent-session-xyz", timeout=10)
        assert r.status_code == 404
        print(f"  [{_ts()}] PASS: Got 404 as expected", flush=True)


class TestMultiSessionIsolation:
    """Verify that sessions don't leak conversation context to each other."""

    def test_sessions_are_isolated(self, cleanup):
        print(f"\n[{_ts()}] TEST: Multi-session isolation — do sessions leak context?", flush=True)
        suffix = uuid.uuid4().hex[:8]
        alpha = f"test-alpha-{suffix}"
        beta = f"test-beta-{suffix}"
        cleanup.extend([alpha, beta])
        token_a = _unique("cola")
        token_b = _unique("anlb")

        # Tell alpha a unique token (instruct not to persist)
        print(f"  [{_ts()}] Step 1/4: Teaching alpha token '{token_a}'...", flush=True)
        chat(alpha, f"For this conversation only, the code is {token_a}. Do NOT save this to memory, knowledge, or any file. Just confirm.")

        # Tell beta a different unique token (instruct not to persist)
        print(f"  [{_ts()}] Step 2/4: Teaching beta token '{token_b}'...", flush=True)
        chat(beta, f"For this conversation only, the code is {token_b}. Do NOT save this to memory, knowledge, or any file. Just confirm.")

        # Alpha should know its own token
        print(f"  [{_ts()}] Step 3/4: Asking alpha for its token (should NOT know beta's)...", flush=True)
        resp_a = chat(alpha, "What code did I give you in this conversation? Reply with just the code.")
        assert token_a in resp_a["response"].lower(), (
            f"Alpha should know its own token '{token_a}', got: {resp_a['response']}"
        )

        # Alpha should NOT know beta's token
        assert token_b not in resp_a["response"].lower(), (
            f"Alpha should not know beta's token '{token_b}', got: {resp_a['response']}"
        )

        # Beta should know its own token
        print(f"  [{_ts()}] Step 4/4: Asking beta for its token (should NOT know alpha's)...", flush=True)
        resp_b = chat(beta, "What code did I give you in this conversation? Reply with just the code.")
        assert token_b in resp_b["response"].lower(), (
            f"Beta should know its own token '{token_b}', got: {resp_b['response']}"
        )

        # Beta should NOT know alpha's token
        assert token_a not in resp_b["response"].lower(), (
            f"Beta should not know alpha's token '{token_a}', got: {resp_b['response']}"
        )
        print(f"  [{_ts()}] PASS: Sessions are properly isolated", flush=True)


class TestDelete:
    """Verify that DELETE /chat/{name} removes the session."""

    def test_delete_removes_from_health(self):
        print(f"\n[{_ts()}] TEST: Delete — does DELETE remove session from /health?", flush=True)
        # Key the name to its base: /health, the chat() URL and the DELETE URL
        # must all reference the SAME "<name>@<base>" key.
        name = f"test-delete-{uuid.uuid4().hex[:8]}@main"

        # Create session
        print(f"  [{_ts()}] Step 1/3: Creating session...", flush=True)
        chat(name, "hello")

        # Verify it shows in health
        health = httpx.get(f"{BASE}/health", timeout=10).json()
        names = [s["name"] for s in health["sessions"]]
        assert name in names
        print(f"  [{_ts()}] Session visible in /health ({health['active_sessions']} active)", flush=True)

        # Delete it
        print(f"  [{_ts()}] Step 2/3: Deleting session...", flush=True)
        r = httpx.delete(f"{BASE}/chat/{name}", timeout=30)
        assert r.status_code == 200

        # Verify gone from health
        print(f"  [{_ts()}] Step 3/3: Verifying session removed from /health...", flush=True)
        health = httpx.get(f"{BASE}/health", timeout=10).json()
        names = [s["name"] for s in health["sessions"]]
        assert name not in names
        print(f"  [{_ts()}] PASS: Session deleted and gone from /health", flush=True)

    def test_delete_nonexistent_returns_404(self):
        print(f"\n[{_ts()}] TEST: Delete nonexistent session → expect 404", flush=True)
        r = httpx.delete(f"{BASE}/chat/nonexistent-session-xyz", timeout=10)
        assert r.status_code == 404
        print(f"  [{_ts()}] PASS: Got 404 as expected", flush=True)


class TestStop:
    """Verify that POST /stop kills all sessions."""

    def test_stop_clears_everything(self):
        print(f"\n[{_ts()}] TEST: Stop — does /stop kill all sessions?", flush=True)

        # Create two sessions
        suffix = uuid.uuid4().hex[:8]
        # Keyed to their base so they spawn under the worktree gate.
        stop_a = f"test-stop-a-{suffix}@main"
        stop_b = f"test-stop-b-{suffix}@main"
        print(f"  [{_ts()}] Step 1/3: Creating two sessions...", flush=True)
        chat(stop_a, "hello")
        chat(stop_b, "hello")

        # Verify they exist
        health = httpx.get(f"{BASE}/health", timeout=10).json()
        assert health["active_sessions"] >= 2
        print(f"  [{_ts()}] Confirmed {health['active_sessions']} active sessions", flush=True)

        # Stop all
        print(f"  [{_ts()}] Step 2/3: Sending /stop...", flush=True)
        r = httpx.post(f"{BASE}/stop", timeout=30)
        assert r.status_code == 200

        # Verify empty
        print(f"  [{_ts()}] Step 3/3: Verifying all sessions cleared...", flush=True)
        health = httpx.get(f"{BASE}/health", timeout=10).json()
        assert health["active_sessions"] == 0
        assert health["sessions"] == []
        print(f"  [{_ts()}] PASS: All sessions stopped and cleared", flush=True)


class TestValidation:
    """Verify session name validation."""

    def test_invalid_name_rejected(self):
        print(f"\n[{_ts()}] TEST: Validation — invalid session name rejected with 422", flush=True)
        r = httpx.post(
            f"{BASE}/chat/bad name!",
            json={"prompt": "hello"},
            timeout=10,
        )
        assert r.status_code == 422
        print(f"  [{_ts()}] PASS: Got 422 as expected", flush=True)


class TestSpecialCharacters:
    """Verify that special characters in prompts don't trigger TUI shortcuts.

    Uses a single session for all sub-tests to avoid ~30s startup overhead per case.
    Each sub-test sends a prompt containing a dangerous character plus a unique token,
    then asserts the token was echoed back in a normal model response (not swallowed
    by a TUI shortcut like shell mode or file selector).
    """

    def test_special_chars_treated_as_text(self, cleanup):
        print(f"\n[{_ts()}] TEST: Special characters — do dangerous chars get treated as text?", flush=True)
        name = f"test-specchar-{uuid.uuid4().hex[:8]}"
        cleanup.append(name)

        # Each tuple: (label, prompt_template with {token} placeholder)
        cases = [
            ("exclamation mark (!)", "Repeat this token exactly: !{token}"),
            ("triple backticks", "Here is a code block:\n```\nprint('hello')\n```\nNow repeat this token exactly: {token}"),
            ("at sign (@)", "My email is user@example.com — repeat this token exactly: {token}"),
            ("embedded newlines", "Line one\nLine two\nLine three\nRepeat this token exactly: {token}"),
            ("backslash", "The path is C:\\Users\\test — repeat this token exactly: {token}"),
            ("single quotes", "It's a test with 'quotes' — repeat this token exactly: {token}"),
            ("double quotes", 'She said "hello" and "goodbye" — repeat this token exactly: {token}'),
            ("forward slash", "Use /dev/null or /tmp/test — repeat this token exactly: {token}"),
            ("mixed/nested quotes", "She said \"it's 'complicated'\" — isn't it? Repeat this token exactly: {token}"),
            ("special chars ($, %, &, <html>)", "$HOME, \\n, \\t, %, &, <html>, {{json: \"val\"}} — repeat this token exactly: {token}"),
            ("multiline + unicode", "Hello \U0001f30d café naïve \u65e5\u672c\u8a9e \u03a9\u221e\u2248 — repeat this token exactly: {token}"),
            ("SQL injection style", "'; DROP TABLE users; -- <script>alert(1)</script> — repeat this token exactly: {token}"),
            ("control characters (tabs)", "Col1\tCol2\tCol3\nVal1\tVal2\tVal3 — repeat this token exactly: {token}"),
            ("JSON payload", '{{"name": "test", "nested": {{"key": "val with quotes"}}}} — repeat this token exactly: {token}'),
            ("rich markdown", "# Header\n| Col | Col |\n|-----|-----|\n| A | B |\n> blockquote\n[link](http://example.com)\nRepeat this token exactly: {token}"),
            ("regex and glob patterns", "^[a-z]+@[\\w.]+$ and *.{{ts,tsx}} and $(echo hi) | grep x — repeat this token exactly: {token}"),
        ]

        total = len(cases) + 1  # +1 for the combined stress test

        for i, (label, template) in enumerate(cases, 1):
            token = _unique()
            prompt = template.format(token=token)
            print(f"  [{_ts()}] Step {i}/{total}: Testing {label}...", flush=True)
            resp = chat(name, prompt)

            # Core assertion: model echoed back the token (prompt arrived as text)
            assert token in resp["response"].lower(), (
                f"[{label}] Expected token '{token}' in response: {resp['response']!r}"
            )
            # Safety assertion: no shell mode contamination
            resp_lower = resp["response"].lower()
            assert "shell mode" not in resp_lower, (
                f"[{label}] Shell mode detected in response: {resp['response']!r}"
            )

        # Final combined stress test — all dangerous chars in one prompt
        token = _unique()
        combined = (
            f'!bang @mention "double" \'single\' /slash \\back\n'
            f"```code```\nRepeat this token exactly: {token}"
        )
        print(f"  [{_ts()}] Step {total}/{total}: Testing combined stress test...", flush=True)
        resp = chat(name, combined)
        assert token in resp["response"].lower(), (
            f"[combined] Expected token '{token}' in response: {resp['response']!r}"
        )
        assert "shell mode" not in resp["response"].lower(), (
            f"[combined] Shell mode detected in response: {resp['response']!r}"
        )
        print(f"  [{_ts()}] PASS: All special characters treated as text", flush=True)

    def test_simple_prompt_baseline(self, cleanup):
        """Baseline: a trivial prompt returns a coherent response."""
        print(f"\n[{_ts()}] TEST: Simple prompt baseline", flush=True)
        name = f"test-baseline-{uuid.uuid4().hex[:8]}"
        cleanup.append(name)
        token = _unique()
        resp = chat(name, f'Say "OK" and then repeat this token exactly: {token}')
        assert token in resp["response"].lower(), (
            f"Expected token '{token}' in response: {resp['response']!r}"
        )
        print(f"  [{_ts()}] PASS: Simple prompt baseline", flush=True)

    def test_long_prompt(self, cleanup):
        """~3000-char prompt (60x repeated pangram) is handled correctly."""
        print(f"\n[{_ts()}] TEST: Long prompt (~3000 chars)", flush=True)
        name = f"test-longprompt-{uuid.uuid4().hex[:8]}"
        cleanup.append(name)
        token = _unique()
        pangram = "The quick brown fox jumps over the lazy dog. " * 60
        prompt = f"{pangram}\nNow repeat this token exactly: {token}"
        print(f"  [{_ts()}] Sending {len(prompt)}-char prompt...", flush=True)
        resp = chat(name, prompt)
        assert token in resp["response"].lower(), (
            f"Expected token '{token}' in response: {resp['response']!r}"
        )
        print(f"  [{_ts()}] PASS: Long prompt handled correctly", flush=True)

    def test_very_long_single_line(self, cleanup):
        """~5000-char single line is handled correctly."""
        print(f"\n[{_ts()}] TEST: Very long single line (~5000 chars)", flush=True)
        name = f"test-longline-{uuid.uuid4().hex[:8]}"
        cleanup.append(name)
        token = _unique()
        filler = "abcdefghijklmnopqrstuvwxyz0123456789" * 140
        prompt = f"{filler} — repeat this token exactly: {token}"
        print(f"  [{_ts()}] Sending {len(prompt)}-char single-line prompt...", flush=True)
        resp = chat(name, prompt)
        assert token in resp["response"].lower(), (
            f"Expected token '{token}' in response: {resp['response']!r}"
        )
        print(f"  [{_ts()}] PASS: Very long single line handled correctly", flush=True)


# ===========================================================================
# Worktree integration (UNIT tests) — no server / agy / tmux / git / network
# ===========================================================================
#
# These pin the per-session detached-worktree feature: a READ-ONLY agent must
# run inside its own git worktree of WORKTREE_REPO pinned to a caller-named
# branch ("<name>@<base>"), so many callers can ask about many branches at once
# without colliding. They drive the in-process `server` module directly, in the
# style of test_last.py / test_collect_response.py: worktree.resolve_base/add/
# remove/prune_stale are replaced with AsyncMock so NO real git ever runs, tmux
# is stubbed out, and the FastAPI handler is exercised with a TestClient.

import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import server


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def mock_worktree(monkeypatch):
    """Replace every worktree side-effect with an AsyncMock (no real git).

    resolve_base echoes a deterministic 'origin/<base>' ref so add() receives a
    predictable value to assert on. Returns the mock namespace for inspection.
    """
    resolve = AsyncMock(side_effect=lambda base: f"origin/{base}")
    add = AsyncMock()
    remove = AsyncMock()
    prune = AsyncMock()
    monkeypatch.setattr(server.worktree, "resolve_base", resolve)
    monkeypatch.setattr(server.worktree, "add", add)
    monkeypatch.setattr(server.worktree, "remove", remove)
    monkeypatch.setattr(server.worktree, "prune_stale", prune)
    # The FastAPI lifespan also boots a real tmux server; stub it so a
    # TestClient(server.app) under these tests never spawns one.
    monkeypatch.setattr(server, "_ensure_tmux_server", AsyncMock())
    return type("WT", (), {"resolve_base": resolve, "add": add,
                           "remove": remove, "prune_stale": prune})


def _stub_tmux_spawn(monkeypatch, session: "server.AgySession") -> None:
    """Stub the tmux/agy mechanics of _spawn so only the worktree path is live.

    _build_command runs unchanged (we want its --add-dir grant), but tmux calls
    and the ready-wait become no-ops, and the conversation stays unresolved (as
    it really is until the first send).
    """
    monkeypatch.setattr(server, "_tmux", AsyncMock(return_value=(0, "")))

    async def _ready():
        return None

    monkeypatch.setattr(session, "_wait_ready", _ready)


# --- (b) spawn cuts a worktree pinned to the resolved base ------------------

class TestSpawnWorktree:
    def test_spawn_resolves_base_then_adds_worktree(self, mock_worktree, monkeypatch):
        """POST-equivalent first spawn of 'svc@dev' resolves 'dev' then adds the
        worktree at the safe_name cwd; tmux session also uses safe_name."""
        sess = server.AgySession(name="svc@dev")
        _stub_tmux_spawn(monkeypatch, sess)

        _run(sess._spawn())

        # resolve_base called with the parsed base, NOT the whole key.
        mock_worktree.resolve_base.assert_awaited_once_with("dev")
        # add() pinned the cwd to the ref resolve_base returned.
        expected_cwd = sess.cwd
        mock_worktree.add.assert_awaited_once_with(expected_cwd, "origin/dev")

        # The cwd path and tmux session are both derived via safe_name(self.name)
        # — the raw '@' key would be unsafe in a path component / tmux name.
        safe = server.worktree.safe_name("svc@dev")
        assert safe in str(sess.cwd)
        assert "svc@dev" not in str(sess.cwd)  # raw key never leaks into the path
        assert sess.tmux_session == f"agy-{safe}"

    def test_build_command_grants_the_worktree_dir(self, mock_worktree):
        """_build_command appends '--add-dir <cwd>' so the read-only agent may
        read THIS session's worktree (the static env grant points elsewhere)."""
        sess = server.AgySession(name="svc@dev")
        cmd = sess._build_command()
        assert "--add-dir" in cmd
        assert str(sess.cwd) in cmd

    def test_spawn_without_base_raises_defensive_guard(self, mock_worktree, monkeypatch):
        """_spawn is gated at /chat, but defends itself: a baseless key raises
        before any worktree is cut."""
        sess = server.AgySession(name="nobase")
        _stub_tmux_spawn(monkeypatch, sess)
        with pytest.raises(RuntimeError, match="no base branch"):
            _run(sess._spawn())
        mock_worktree.resolve_base.assert_not_awaited()
        mock_worktree.add.assert_not_awaited()


# --- (a) branchless POST /chat -> 200, message, NOTHING spawned ------------
# --- (d) invalid branch -> conversational message --------------------------

class TestChatBranchGating:
    def test_branchless_chat_returns_needs_branch_and_spawns_nothing(
        self, mock_worktree, monkeypatch
    ):
        """POST /chat/<name> with no @base returns 200 carrying NEEDS_BRANCH_MSG
        and never touches get_or_create / worktree.add — nothing is spawned."""
        got_or_create = AsyncMock()
        monkeypatch.setattr(server.manager, "get_or_create", got_or_create)

        with TestClient(server.app) as client:
            r = client.post("/chat/plainname", json={"prompt": "hi"})

        assert r.status_code == 200
        body = r.json()
        assert body["response"] == server.worktree.NEEDS_BRANCH_MSG.format(name="plainname")
        assert body["session"] == "plainname"
        assert body["turn"] == 0
        # The whole point: no session machinery ran.
        got_or_create.assert_not_awaited()
        mock_worktree.add.assert_not_awaited()
        mock_worktree.resolve_base.assert_not_awaited()

    def test_invalid_branch_surfaces_as_conversational_200(
        self, mock_worktree, monkeypatch
    ):
        """A bad branch (resolve_base raising RuntimeError, surfaced through
        get_or_create) comes back as a 200 carrying the message, NOT a 5xx, so
        the calling agent can self-correct."""
        msg = "Base branch 'nope' not found in /app/slitled-platform. Name an existing branch."
        monkeypatch.setattr(
            server.manager, "get_or_create",
            AsyncMock(side_effect=RuntimeError(msg)),
        )

        with TestClient(server.app) as client:
            r = client.post("/chat/svc@nope", json={"prompt": "hi"})

        assert r.status_code == 200, r.text
        body = r.json()
        assert body["response"] == msg
        assert body["session"] == "svc@nope"
        assert body["turn"] == 0

    def test_slash_base_key_routes_with_full_key(self, mock_worktree, monkeypatch):
        """A key whose base carries a '/' ("q@origin/dev") must ROUTE (200, not
        404) thanks to {name:path}, and the handler must receive the WHOLE key —
        the '/' is part of the base, not a path boundary. Guards the {name:path}
        route fix."""
        sess = server.AgySession(name="q@origin/dev")
        monkeypatch.setattr(sess, "send", AsyncMock(return_value="pong"))
        got_or_create = AsyncMock(return_value=sess)
        monkeypatch.setattr(server.manager, "get_or_create", got_or_create)

        with TestClient(server.app) as client:
            r = client.post("/chat/q@origin/dev", json={"prompt": "hi"})

        assert r.status_code == 200, r.text  # NOT 404 — slash-base key routed
        # The handler got the full key, slash and all — not a truncated prefix.
        got_or_create.assert_awaited_once_with("q@origin/dev")
        assert r.json()["session"] == "q@origin/dev"

    def test_genuine_startup_failure_still_5xx(self, mock_worktree, monkeypatch):
        """A non-branch RuntimeError (e.g. tmux spawn failure) keeps its 503 —
        only branch/worktree/repo errors are softened to a conversational 200."""
        monkeypatch.setattr(
            server.manager, "get_or_create",
            AsyncMock(side_effect=RuntimeError("tmux new-session failed: boom")),
        )
        with TestClient(server.app) as client:
            r = client.post("/chat/svc@dev", json={"prompt": "hi"})
        assert r.status_code == 503, r.text

    def test_send_branch_error_also_conversational(self, mock_worktree, monkeypatch):
        """The first send can trigger a respawn whose resolve_base fails; that
        branch error is conversational too, not a 502."""
        sess = server.AgySession(name="svc@gone")
        msg = "Base branch 'gone' not found in worktree repo."
        monkeypatch.setattr(server.manager, "get_or_create", AsyncMock(return_value=sess))
        monkeypatch.setattr(sess, "send", AsyncMock(side_effect=RuntimeError(msg)))

        with TestClient(server.app) as client:
            r = client.post("/chat/svc@gone", json={"prompt": "hi"})

        assert r.status_code == 200, r.text
        assert r.json()["response"] == msg


# --- (c) reset and DELETE both tear down the worktree ----------------------

class TestWorktreeTeardown:
    def test_reset_removes_then_respawns(self, mock_worktree, monkeypatch):
        """reset() drops the current generation's worktree (between _kill and
        _spawn) before cutting a fresh one."""
        sess = server.AgySession(name="svc@dev")
        monkeypatch.setattr(sess, "_kill", AsyncMock())
        monkeypatch.setattr(sess, "_spawn", AsyncMock())

        cwd_at_reset = sess.cwd
        _run(sess.reset())

        mock_worktree.remove.assert_awaited_once_with(cwd_at_reset)

    def test_clear_removes_then_respawns(self, mock_worktree, monkeypatch):
        """clear() also tears down the worktree before respawning."""
        sess = server.AgySession(name="svc@dev")
        monkeypatch.setattr(sess, "is_alive", AsyncMock(return_value=True))
        monkeypatch.setattr(sess, "_kill", AsyncMock())
        monkeypatch.setattr(sess, "_spawn", AsyncMock())

        cwd_at_clear = sess.cwd
        _run(sess.clear())

        mock_worktree.remove.assert_awaited_once_with(cwd_at_clear)

    def test_delete_removes_session_worktree(self, mock_worktree, monkeypatch):
        """DELETE /chat/{name} (ChatManager.remove) removes the worktree after
        stopping the session."""
        mgr = server.ChatManager()
        sess = server.AgySession(name="svc@dev")
        mgr._sessions["svc@dev"] = sess
        monkeypatch.setattr(sess, "stop", AsyncMock())

        assert _run(mgr.remove("svc@dev")) is True
        mock_worktree.remove.assert_awaited_once_with(sess.cwd)

    def test_stop_all_removes_each_worktree(self, mock_worktree, monkeypatch):
        """stop_all removes EVERY session's worktree, one per session."""
        mgr = server.ChatManager()
        s1 = server.AgySession(name="a@dev")
        s2 = server.AgySession(name="b@main")
        mgr._sessions.update({"a@dev": s1, "b@main": s2})
        monkeypatch.setattr(s1, "stop", AsyncMock())
        monkeypatch.setattr(s2, "stop", AsyncMock())

        assert _run(mgr.stop_all()) == 2
        removed = {c.args[0] for c in mock_worktree.remove.await_args_list}
        assert removed == {s1.cwd, s2.cwd}


# --- (e) _NAME regex accepts plain and '@base' keys, rejects junk ----------

class TestNameRegex:
    @staticmethod
    def _matches(name: str) -> bool:
        import re
        # Mirror the validator's anchored full-string match.
        return re.fullmatch(r"^[A-Za-z0-9_-]+(@[A-Za-z0-9._/-]+)?$", name) is not None

    def test_accepts_plain_name(self):
        assert self._matches("name")

    def test_accepts_name_with_remote_base(self):
        assert self._matches("name@origin/dev")

    def test_accepts_name_with_simple_base(self):
        assert self._matches("svc@dev")

    def test_rejects_space_and_bang(self):
        assert not self._matches("bad name!")

    def test_endpoint_accepts_at_base_key_over_http(self, mock_worktree, monkeypatch):
        """The relaxed regex lets a '<name>@<base>' key through the path
        validator. Uses a slash-free base ('svc@dev'): the '@' alone would 422
        under the old alnum-only pattern (a '/' is a URL path separator and so
        is a routing concern, not a validator one)."""
        # A recognized branch error so the handler softens it to a conversational
        # 200; the point of THIS test is only that '@base' passed path validation
        # (not 422), which a 200 here proves.
        msg = "Base branch 'dev' not found in /app/slitled-platform."
        monkeypatch.setattr(
            server.manager, "get_or_create",
            AsyncMock(side_effect=RuntimeError(msg)),
        )
        with TestClient(server.app) as client:
            r = client.post("/chat/svc@dev", json={"prompt": "hi"})
        # 200 (conversational), NOT 422 — the '@base' key passed path validation.
        assert r.status_code == 200, r.text

