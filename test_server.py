"""
E2E tests for the Gemini CLI REST Bridge.

Starts the server as a subprocess (or uses an existing one),
runs real Gemini sessions, and verifies memory, clear, reset,
isolation, delete, and stop.

Usage:
    pytest test_server.py -v
"""

import subprocess
import sys
import time
import uuid

import httpx
import pytest

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
    """Use existing server (e.g. Docker) or launch uvicorn locally."""
    if _server_already_running():
        # Server already up (Docker, manual, etc.) — just use it
        yield None
        try:
            httpx.post(f"{BASE}/stop", timeout=10)
        except Exception:
            pass
        return

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server:app",
         "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(__import__("pathlib").Path(__file__).parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait for server to be reachable
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{BASE}/health", timeout=2)
            if r.status_code == 200:
                break
        except httpx.ConnectError:
            time.sleep(0.5)
    else:
        proc.kill()
        raise RuntimeError("Server did not start within 30s")

    yield proc

    # Cleanup: stop all sessions then kill server
    try:
        httpx.post(f"{BASE}/stop", timeout=10)
    except Exception:
        pass
    proc.terminate()
    proc.wait(timeout=10)


def chat(session: str, prompt: str) -> dict:
    """Send a chat message and return the JSON response."""
    r = httpx.post(
        f"{BASE}/chat/{session}",
        json={"prompt": prompt},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Cleanup: each test that creates sessions cleans up after itself
# ---------------------------------------------------------------------------

@pytest.fixture()
def cleanup():
    """Collects session names to delete after test."""
    sessions = []
    yield sessions
    for name in sessions:
        try:
            httpx.delete(f"{BASE}/chat/{name}", timeout=30)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChatMemory:
    """Verify that a session remembers context across turns."""

    def test_remembers_fact(self, cleanup):
        name = "test-memory"
        cleanup.append(name)
        token = _unique()

        # Tell it a fact (instruct not to persist to GEMINI.md)
        chat(name, f"For this conversation only, my secret word is {token}. Do NOT save this to memory or GEMINI.md. Just confirm you got it.")

        # Ask for it back
        resp = chat(name, "What is my secret word? Reply with just the word.")
        assert token in resp["response"].lower(), (
            f"Expected '{token}' in response, got: {resp['response']!r}"
        )
        assert resp["session"] == name
        assert resp["turn"] == 2


class TestClear:
    """Verify that /clear wipes conversation context."""

    def test_clear_forgets_context(self, cleanup):
        name = "test-clear"
        cleanup.append(name)
        token = _unique()

        # Establish context (instruct not to persist to GEMINI.md)
        chat(name, f"For this conversation only, my secret code is {token}. Do NOT save this to memory or GEMINI.md. Just confirm you got it.")

        # Clear — should succeed
        r = httpx.post(f"{BASE}/clear/{name}", timeout=60)
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

        # Ask for it — should NOT know (context wiped)
        resp = chat(name, "What is my secret code? If you don't know, say 'no idea'.")
        assert token not in resp["response"].lower(), (
            f"Expected context to be cleared, but got: {resp['response']}"
        )

    def test_clear_nonexistent_returns_404(self):
        r = httpx.post(f"{BASE}/clear/nonexistent-session-xyz", timeout=10)
        assert r.status_code == 404


class TestReset:
    """Verify that /reset kills and restarts the session."""

    def test_reset_returns_200_and_resets_turn_count(self, cleanup):
        name = "test-reset"
        cleanup.append(name)

        # Build up some turns
        chat(name, "Hello.")
        resp2 = chat(name, "Say 'pong'.")
        assert resp2["turn"] == 2

        # Reset
        r = httpx.post(f"{BASE}/reset/{name}", timeout=60)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"

        # Next chat should be turn 1 (fresh process)
        resp3 = chat(name, "Say 'hello'.")
        assert resp3["turn"] == 1
        assert resp3["response"]

    def test_reset_nonexistent_returns_404(self):
        r = httpx.post(f"{BASE}/reset/nonexistent-session-xyz", timeout=10)
        assert r.status_code == 404


class TestMultiSessionIsolation:
    """Verify that sessions don't leak conversation context to each other."""

    def test_sessions_are_isolated(self, cleanup):
        cleanup.extend(["test-alpha", "test-beta"])
        token_a = _unique("cola")
        token_b = _unique("anlb")

        # Tell alpha a unique token (instruct not to persist)
        chat("test-alpha", f"For this conversation only, the code is {token_a}. Do NOT save this to memory or GEMINI.md. Just confirm.")

        # Tell beta a different unique token (instruct not to persist)
        chat("test-beta", f"For this conversation only, the code is {token_b}. Do NOT save this to memory or GEMINI.md. Just confirm.")

        # Alpha should know its own token
        resp_a = chat("test-alpha", "What code did I give you in this conversation? Reply with just the code.")
        assert token_a in resp_a["response"].lower(), (
            f"Alpha should know its own token '{token_a}', got: {resp_a['response']}"
        )

        # Alpha should NOT know beta's token
        assert token_b not in resp_a["response"].lower(), (
            f"Alpha should not know beta's token '{token_b}', got: {resp_a['response']}"
        )

        # Beta should know its own token
        resp_b = chat("test-beta", "What code did I give you in this conversation? Reply with just the code.")
        assert token_b in resp_b["response"].lower(), (
            f"Beta should know its own token '{token_b}', got: {resp_b['response']}"
        )

        # Beta should NOT know alpha's token
        assert token_a not in resp_b["response"].lower(), (
            f"Beta should not know alpha's token '{token_a}', got: {resp_b['response']}"
        )


class TestDelete:
    """Verify that DELETE /chat/{name} removes the session."""

    def test_delete_removes_from_health(self):
        name = "test-delete"

        # Create session
        chat(name, "hello")

        # Verify it shows in health
        health = httpx.get(f"{BASE}/health", timeout=10).json()
        names = [s["name"] for s in health["sessions"]]
        assert name in names

        # Delete it
        r = httpx.delete(f"{BASE}/chat/{name}", timeout=30)
        assert r.status_code == 200

        # Verify gone from health
        health = httpx.get(f"{BASE}/health", timeout=10).json()
        names = [s["name"] for s in health["sessions"]]
        assert name not in names

    def test_delete_nonexistent_returns_404(self):
        r = httpx.delete(f"{BASE}/chat/nonexistent-session-xyz", timeout=10)
        assert r.status_code == 404


class TestStop:
    """Verify that POST /stop kills all sessions."""

    def test_stop_clears_everything(self):
        # Create two sessions
        chat("test-stop-a", "hello")
        chat("test-stop-b", "hello")

        # Verify they exist
        health = httpx.get(f"{BASE}/health", timeout=10).json()
        assert health["active_sessions"] >= 2

        # Stop all
        r = httpx.post(f"{BASE}/stop", timeout=30)
        assert r.status_code == 200

        # Verify empty
        health = httpx.get(f"{BASE}/health", timeout=10).json()
        assert health["active_sessions"] == 0
        assert health["sessions"] == []


class TestValidation:
    """Verify session name validation."""

    def test_invalid_name_rejected(self):
        r = httpx.post(
            f"{BASE}/chat/bad name!",
            json={"prompt": "hello"},
            timeout=10,
        )
        assert r.status_code == 422
