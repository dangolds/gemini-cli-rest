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

    def test_large_code_io(self, cleanup):
        """~5000-line code input expecting large (~6000-line) code output.

        Stress-tests the PTY I/O pipeline: bracketed paste for large input,
        and response collection loop for sustained large output.
        """
        print(f"\n[{_ts()}] TEST: Large code I/O (~5k lines in, expecting ~6k out)", flush=True)
        name = f"test-largecodeio-{uuid.uuid4().hex[:8]}"
        cleanup.append(name)
        token = _unique()

        # Generate ~5000 lines of Python code (1000 functions × 5 lines each)
        code_lines = []
        for i in range(1000):
            code_lines.append(f"def process_item_{i}(data):")
            code_lines.append(f"    value = data.get('field_{i}', {i})")
            code_lines.append(f"    result = value * {i + 1} + {i % 7}")
            code_lines.append(f"    return {{'id': {i}, 'result': result}}")
            code_lines.append("")

        code_block = "\n".join(code_lines)
        prompt = (
            f"Here is a Python file with {len(code_lines)} lines of code:\n"
            f"```python\n{code_block}\n```\n\n"
            f"Rewrite ALL of these functions adding: "
            f"(1) Google-style docstrings with Args and Returns sections, "
            f"(2) full type hints on parameters and return types, "
            f"(3) input validation that raises ValueError for bad input. "
            f"Output the COMPLETE rewritten file in a single code block — "
            f"do NOT skip, abbreviate, or summarize any functions. "
            f"At the very end of the code block, add this comment: # TOKEN: {token}"
        )

        print(
            f"  [{_ts()}] Sending {len(code_lines)}-line, {len(prompt)}-char prompt...",
            flush=True,
        )
        resp = chat(name, prompt)

        # Assert token present — proves the full prompt was received and
        # the model reached the end of its output without truncation
        assert token in resp["response"].lower(), (
            f"Expected token '{token}' in response (last 300 chars): "
            f"{resp['response'][-300:]!r}"
        )

        # Assert response is substantial (model didn't summarize/skip)
        resp_lines = resp["response"].strip().splitlines()
        resp_chars = len(resp["response"])
        print(
            f"  [{_ts()}] Got {len(resp_lines)} lines, {resp_chars} chars back",
            flush=True,
        )
        assert len(resp_lines) >= 2000, (
            f"Expected at least 2000 lines in response, got {len(resp_lines)}. "
            f"Model may have summarized instead of outputting all functions."
        )

        print(f"  [{_ts()}] PASS: Large code I/O handled correctly", flush=True)
