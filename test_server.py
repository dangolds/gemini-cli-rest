"""
E2E tests for the Gemini CLI REST Bridge.

Starts the server as a subprocess (or uses an existing one),
runs real Gemini sessions, and verifies memory, clear, reset,
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
# Gemini can be slow — first turn spawns the CLI (~30s) + model response time
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


@pytest.fixture(scope="session", autouse=True)
def server():
    """Require a running server (e.g. Docker). Fails fast if none is found."""
    print(f"\n[{_ts()}] Checking for server at {BASE}...", flush=True)
    if not _server_already_running():
        pytest.fail(
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
        name = f"test-memory-{uuid.uuid4().hex[:8]}"
        cleanup.append(name)
        token = _unique()

        # Tell it a fact (instruct not to persist to GEMINI.md)
        print(f"  [{_ts()}] Step 1/2: Teaching secret word '{token}'...", flush=True)
        chat(name, f"For this conversation only, my secret word is {token}. Do NOT save this to memory or GEMINI.md. Just confirm you got it.")

        # Ask for it back
        print(f"  [{_ts()}] Step 2/2: Asking for the word back...", flush=True)
        resp = chat(name, "What is my secret word? Reply with just the word.")
        assert token in resp["response"].lower(), (
            f"Expected '{token}' in response, got: {resp['response']!r}"
        )
        assert resp["session"] == name
        assert resp["turn"] == 2
        print(f"  [{_ts()}] PASS: Session correctly remembered '{token}'", flush=True)


class TestClear:
    """Verify that /clear wipes conversation context."""

    def test_clear_forgets_context(self, cleanup):
        print(f"\n[{_ts()}] TEST: Clear — does /clear wipe conversation context?", flush=True)
        name = f"test-clear-{uuid.uuid4().hex[:8]}"
        cleanup.append(name)
        token = _unique()

        # Establish context (instruct not to persist to GEMINI.md)
        print(f"  [{_ts()}] Step 1/3: Teaching secret code '{token}'...", flush=True)
        chat(name, f"For this conversation only, my secret code is {token}. Do NOT save this to memory or GEMINI.md. Just confirm you got it.")

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
        name = f"test-reset-{uuid.uuid4().hex[:8]}"
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
        chat(alpha, f"For this conversation only, the code is {token_a}. Do NOT save this to memory or GEMINI.md. Just confirm.")

        # Tell beta a different unique token (instruct not to persist)
        print(f"  [{_ts()}] Step 2/4: Teaching beta token '{token_b}'...", flush=True)
        chat(beta, f"For this conversation only, the code is {token_b}. Do NOT save this to memory or GEMINI.md. Just confirm.")

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
        name = f"test-delete-{uuid.uuid4().hex[:8]}"

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
        stop_a = f"test-stop-a-{suffix}"
        stop_b = f"test-stop-b-{suffix}"
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
