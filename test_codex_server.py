"""
E2E tests for the Codex CLI REST Bridge.

Requires a running codex-rest server (e.g. Docker on port 8001),
runs real codex sessions, and verifies memory, clear, reset,
isolation, delete, and stop.

Usage:
    pytest test_codex_server.py -v
"""

import time
import uuid
from datetime import datetime

import httpx
import pytest


def _ts() -> str:
    """Short timestamp for progress output."""
    return datetime.now().strftime("%H:%M:%S")

BASE = "http://127.0.0.1:8001"
# codex can be slow — first turn spawns the TUI in tmux, plus high reasoning +
# model response time per turn.
TIMEOUT = 300.0


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


@pytest.fixture(autouse=True)
def server(request):
    """Require a running server (e.g. Docker) for the E2E suite.

    Autouse, but a no-op for the mocked worktree unit tests (TestWorktreeBranching,
    appended at the end of this file) — those mock tmux/git and need NO live
    server. For every E2E test it SKIPs (not fails) when no server is up, so
    running this file without Docker still exercises the mocked unit tests rather
    than erroring the whole file out.
    """
    if request.cls is TestWorktreeBranching:
        yield  # mocked unit tests — no live bridge needed, no teardown
        return
    print(f"\n[{_ts()}] Checking for server at {BASE}...", flush=True)
    if not _server_already_running():
        pytest.skip(
            f"No server running at {BASE}. Start it first with: docker compose up -d"
        )
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
    # A bare session name (no "@base") now hits the branchless handshake gate
    # instead of spawning, so default-base E2E sessions to "@main" (main exists)
    # — callers that already key their name (e.g. for /health membership) keep it.
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
    # Mirror chat(): a bare name is keyed to "@main" so /last addresses the same
    # session the chat() helper spawned under that key.
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
        # Key with "@main": the handler echoes back the full session key, so the
        # resp["session"] == name assertion below must compare against that key.
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
        # Key with "@main": both chat() and the resp["session"] == name assertion
        # use the full key the handler echoes.
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
        # Key with "@main" so chat()/last() and cleanup's DELETE all address the
        # same session key.
        name = f"test-last2-{uuid.uuid4().hex[:8]}@main"
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
        """/clear respawns the codex session; /last must not surface the pre-clear answer."""
        print(f"\n[{_ts()}] TEST: /last after /clear → no stale pre-clear answer", flush=True)
        # Key with "@main": /clear addresses by the full session key.
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
        # Key with "@main": /clear addresses by the full session key.
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
    """Verify that /reset starts a fresh conversation."""

    def test_reset_returns_200_and_resets_turn_count(self, cleanup):
        print(f"\n[{_ts()}] TEST: Reset — does /reset start fresh with a reset turn count?", flush=True)
        # Key with "@main": /reset addresses by the full session key.
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

        # Next chat should be turn 1 (fresh session)
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
        # Key with "@main" so cleanup's DELETE addresses the spawned session keys.
        alpha = f"test-alpha-{suffix}@main"
        beta = f"test-beta-{suffix}@main"
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
        # Key the name with "@main": the bridge stores the session under the full
        # key, so /health membership and the DELETE URL must use that same key.
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

        # Create two sessions (keyed with "@main" so they spawn, not handshake)
        suffix = uuid.uuid4().hex[:8]
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
    """Verify that special characters in prompts survive the tmux paste.

    codex now receives the prompt as a tmux bracketed paste (like the agy
    bridge), so we verify that dangerous characters, newlines, and unicode
    arrive as literal text rather than triggering a TUI shortcut (e.g. a leading
    '/' becoming a slash command). Uses a single session for all sub-tests to
    avoid per-case startup overhead. Each sub-test sends a prompt containing a
    dangerous character plus a unique token, then asserts the token was echoed
    back in a normal model response.
    """

    def test_special_chars_treated_as_text(self, cleanup):
        print(f"\n[{_ts()}] TEST: Special characters — do dangerous chars get treated as text?", flush=True)
        name = f"test-specchar-{uuid.uuid4().hex[:8]}@main"
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
            ("multiline + unicode", "Hello \U0001f30d café naïve 日本語 Ω∞≈ — repeat this token exactly: {token}"),
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

        # Final combined stress test — all dangerous chars in one prompt.
        # NOTE: the prompt must NOT *begin* with '!' or '/'. codex's TUI reads a
        # leading '!' as Shell mode (runs the rest as bash) and a leading '/' as a
        # slash-command (verified codex 0.141) — a leading command char is
        # interpreted, never sent to the model, and bracketed paste / a leading
        # space can't bypass it (codex trims whitespace). That's codex UX, not a
        # bridge defect. Mid-prompt '!' '/' backticks etc. ARE treated as literal
        # text (the 16 cases above prove it), so the stress prompt keeps them all
        # but leads with normal text.
        token = _unique()
        combined = (
            f'Repeat this token exactly: {token}\n'
            f'Treat the rest as literal text: !bang @mention "double" \'single\' '
            f"/slash \\back\n```code```"
        )
        print(f"  [{_ts()}] Step {total}/{total}: Testing combined stress test...", flush=True)
        resp = chat(name, combined)
        assert token in resp["response"].lower(), (
            f"[combined] Expected token '{token}' in response: {resp['response']!r}"
        )
        print(f"  [{_ts()}] PASS: All special characters treated as text", flush=True)

    def test_simple_prompt_baseline(self, cleanup):
        """Baseline: a trivial prompt returns a coherent response."""
        print(f"\n[{_ts()}] TEST: Simple prompt baseline", flush=True)
        name = f"test-baseline-{uuid.uuid4().hex[:8]}@main"
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
        name = f"test-longprompt-{uuid.uuid4().hex[:8]}@main"
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
        name = f"test-longline-{uuid.uuid4().hex[:8]}@main"
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
# Worktree branching (mocked — no live server, no real git/tmux/network)
# ===========================================================================
#
# These pin the per-session git-worktree backing added to the codex bridge.
# They mock worktree.resolve_base/add/remove/prune_stale with AsyncMock so NO
# real git runs, and (where a session must spawn) stub the tmux/codex layer the
# same way the rollout/diagnostic unit tests do — so a "spawn" creates no real
# process. Run:  ./.venv/bin/python -m pytest test_codex_server.py -q
#
# The E2E classes above require a live bridge (their autouse `server` fixture
# SKIPs when none is up); this class is exempt from that fixture, so it runs
# standalone with everything mocked.

import asyncio
import re
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

import codex_server
import worktree


def _run(coro):
    return asyncio.run(coro)


class TestWorktreeBranching:
    """The detached-worktree session backing: branchless gate, spawn wiring,
    safe-name derivation, teardown on reset/delete, and the conversational
    branch-error path. All git/tmux is mocked — nothing real is spawned."""

    @pytest.fixture()
    def wt(self, monkeypatch):
        """Mock every worktree side-effect with AsyncMock (no real git)."""
        resolve = AsyncMock(return_value="origin/dev-resolved")
        add = AsyncMock()
        remove = AsyncMock()
        prune = AsyncMock()
        monkeypatch.setattr(codex_server.worktree, "resolve_base", resolve)
        monkeypatch.setattr(codex_server.worktree, "add", add)
        monkeypatch.setattr(codex_server.worktree, "remove", remove)
        monkeypatch.setattr(codex_server.worktree, "prune_stale", prune)

        # The FastAPI lifespan (run by TestClient) bootstraps a real tmux server;
        # stub it so these unit tests never touch tmux.
        async def fake_ensure():
            return None

        monkeypatch.setattr(codex_server, "_ensure_tmux_server", fake_ensure)
        return {"resolve": resolve, "add": add, "remove": remove, "prune": prune}

    @pytest.fixture()
    def no_tmux(self, monkeypatch):
        """Stub the tmux/codex layer so a 'spawn' starts no real process.

        _build_command still runs (we want it to exercise --add-dir), but the
        tmux new-session, ready-wait, and liveness check are all no-ops. is_alive
        reports True so a spawned session looks live to send()/health.
        """
        async def fake_tmux(*args, **kwargs):
            return (0, "")

        async def fake_wait_ready(self):
            return None

        async def fake_is_alive(self):
            return True

        monkeypatch.setattr(codex_server, "_tmux", fake_tmux)
        monkeypatch.setattr(codex_server.CodexSession, "_wait_ready", fake_wait_ready)
        monkeypatch.setattr(codex_server.CodexSession, "is_alive", fake_is_alive)

    @pytest.fixture()
    def spawn_path(self, monkeypatch):
        """Like no_tmux, but is_alive starts False so get_or_create -> start()
        actually reaches _spawn (the path where a branch error is raised)."""
        async def fake_tmux(*args, **kwargs):
            return (0, "")

        async def fake_wait_ready(self):
            return None

        async def fake_is_alive(self):
            return False  # force start() to spawn rather than short-circuit

        monkeypatch.setattr(codex_server, "_tmux", fake_tmux)
        monkeypatch.setattr(codex_server.CodexSession, "_wait_ready", fake_wait_ready)
        monkeypatch.setattr(codex_server.CodexSession, "is_alive", fake_is_alive)

    # --- (e) regex --------------------------------------------------------

    def test_name_regex_accepts_bare_and_based_keys(self):
        raw = next(m.pattern for m in codex_server._NAME.metadata
                   if getattr(m, "pattern", None))
        pat = re.compile(raw)
        assert pat.fullmatch("name")
        assert pat.fullmatch("name@origin/dev")
        # and still rejects an unsafe key / a trailing-only '@'
        assert not pat.fullmatch("bad name!")
        assert not pat.fullmatch("name@")

    # --- safe_name derivation --------------------------------------------

    def test_cwd_and_tmux_name_use_safe_name(self):
        """A key with '@' and '/' must not leak into the tmux name or fragment
        the path — both derive from worktree.safe_name(self.name)."""
        name = "feat@origin/dev"
        sess = codex_server.CodexSession(name=name)
        safe = worktree.safe_name(name)
        assert sess.tmux_session == f"codex-{safe}"
        # path is one component (no nested origin/dev dir) ending in the gen dir
        assert sess.cwd == codex_server.SESSIONS_ROOT / safe / "c0"
        assert "@" not in sess.tmux_session and "/" not in safe

    # --- (a) branchless POST /chat/{name} ---------------------------------

    def test_branchless_chat_returns_needs_branch_and_spawns_nothing(self, wt, monkeypatch):
        """A key with no @<base> -> 200 NEEDS_BRANCH_MSG, and NOTHING is started:
        get_or_create and worktree.add are never called."""
        goc = AsyncMock()
        monkeypatch.setattr(codex_server.manager, "get_or_create", goc)

        with TestClient(codex_server.app) as client:
            r = client.post("/chat/lonely", json={"prompt": "hi"})

        assert r.status_code == 200
        body = r.json()
        assert body["response"] == worktree.NEEDS_BRANCH_MSG.format(name="lonely")
        assert body["session"] == "lonely"
        assert body["turn"] == 0
        goc.assert_not_called()       # nothing spawned
        wt["add"].assert_not_called()  # no worktree cut

    def test_slash_base_key_routes_and_passes_full_key(self, wt, monkeypatch):
        """A key whose base contains '/' ('q@origin/dev') must route via
        {name:path} -> 200 (NOT a 404 from a non-greedy path converter), and the
        handler must hand get_or_create the WHOLE key, not a truncated prefix."""
        sess = AsyncMock()
        sess.send = AsyncMock(return_value="ok")
        sess.turn_count = 1
        # Real sessions carry str|None here; a bare Mock attribute would fail
        # the response model's `via` validation and turn the 200 into a 500.
        sess._last_exit_reason = "notify"
        goc = AsyncMock(return_value=sess)
        monkeypatch.setattr(codex_server.manager, "get_or_create", goc)

        with TestClient(codex_server.app) as client:
            r = client.post("/chat/q@origin/dev", json={"prompt": "hi"})

        assert r.status_code == 200, r.status_code  # routed, not 404
        body = r.json()
        assert body["session"] == "q@origin/dev"
        goc.assert_awaited_once_with("q@origin/dev")  # full slash-base key received

    # --- (b) spawn wires resolve_base -> add(cwd, ref) --------------------

    def test_spawn_resolves_base_then_adds_worktree(self, wt, no_tmux):
        """_spawn parses the base, resolves it, and cuts the worktree at cwd
        pinned to the resolved ref — using the safe_name-derived cwd."""
        sess = codex_server.CodexSession(name="work@dev")
        _run(sess._spawn())

        wt["resolve"].assert_awaited_once_with("dev")
        # add(cwd, ref) — cwd uses safe_name, ref is resolve_base's return
        (cwd_arg, ref_arg), _ = wt["add"].call_args
        assert cwd_arg == sess.cwd
        assert cwd_arg == codex_server.SESSIONS_ROOT / worktree.safe_name("work@dev") / "c1"
        assert ref_arg == "origin/dev-resolved"
        # the worktree is also granted to codex dynamically via --add-dir
        assert f"--add-dir {sess.cwd}" in sess._build_command()

    def test_spawn_without_base_raises_defensively(self, wt, no_tmux):
        """_spawn is gated by /chat, but defends itself if reached branchless."""
        sess = codex_server.CodexSession(name="nobranch")
        with pytest.raises(RuntimeError, match="no base branch"):
            _run(sess._spawn())
        wt["resolve"].assert_not_called()

    # --- (c) reset and DELETE both tear down the worktree -----------------

    def test_reset_removes_then_respawns_worktree(self, wt, no_tmux):
        sess = codex_server.CodexSession(name="r@dev")
        _run(sess._spawn())            # generation 1 cut
        wt["remove"].reset_mock()
        wt["add"].reset_mock()

        _run(sess.reset())

        wt["remove"].assert_awaited_once_with(sess.cwd.parent / "c1")  # old gen torn down
        wt["add"].assert_awaited_once()                                 # fresh gen re-cut

    def test_delete_removes_worktree(self, wt, no_tmux):
        """DELETE /chat/{name} stops the session AND removes its worktree."""
        sess = codex_server.CodexSession(name="d@dev")
        _run(sess._spawn())
        _run(codex_server.manager._lock.acquire())
        codex_server.manager._sessions["d@dev"] = sess
        codex_server.manager._lock.release()
        cwd = sess.cwd

        with TestClient(codex_server.app) as client:
            r = client.delete("/chat/d@dev")

        assert r.status_code == 200
        wt["remove"].assert_awaited_with(cwd)

    # --- (d) invalid branch surfaces as the conversational message --------

    def test_invalid_branch_is_conversational_200(self, wt, spawn_path):
        """resolve_base raising (bad branch) -> 200 carrying that message, so the
        agent can self-correct, NOT a 5xx it would read as the bridge being down."""
        _run(codex_server.manager.stop_all())  # no leftover session short-circuits start()
        wt["resolve"].side_effect = RuntimeError(
            "Base branch 'nope' not found in /app/slitled-platform. Name an existing branch."
        )
        with TestClient(codex_server.app) as client:
            r = client.post("/chat/ask@nope", json={"prompt": "hi"})

        assert r.status_code == 200
        body = r.json()
        assert "not found in" in body["response"]
        assert body["session"] == "ask@nope"
        assert body["turn"] == 0

    def test_genuine_startup_failure_stays_5xx(self, wt, spawn_path, monkeypatch):
        """A non-branch RuntimeError during start keeps its 503 (bridge fault)."""
        _run(codex_server.manager.stop_all())
        async def boom(self):
            raise RuntimeError("tmux new-session failed: server not running")

        monkeypatch.setattr(codex_server.CodexSession, "_spawn", boom)
        with TestClient(codex_server.app) as client:
            r = client.post("/chat/boom@dev", json={"prompt": "hi"})
        assert r.status_code == 503

    # --- (8) startup prune --------------------------------------------------

    def test_lifespan_prunes_stale_worktrees(self, wt, monkeypatch):
        """The FastAPI lifespan runs worktree.prune_stale() once on startup."""
        async def fake_ensure():
            return None

        monkeypatch.setattr(codex_server, "_ensure_tmux_server", fake_ensure)
        with TestClient(codex_server.app):
            pass
        wt["prune"].assert_awaited()
