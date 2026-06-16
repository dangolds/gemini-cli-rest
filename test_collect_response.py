"""
Unit tests for AgySession._collect_response (progress-aware turn completion).

These pin the fix for the third failure class we hit: a long agentic review
(file reads, command runs, a slowly-streamed answer) that ran past the old flat
140s wall. The bridge used to give up while agy was still actively working and
hadn't yet flushed its answer to the transcript, returning "0 message(s)".

The turn must now end on INACTIVITY, not a flat clock: keep waiting while agy
is busy or the transcript is still growing; only give up after a real stall or
the absolute hard ceiling. A non-clean end also drops a local diagnostic file.

No server / agy / Docker — transcript reads and the screen are mocked.
Run:  ./.venv/bin/python -m pytest test_collect_response.py -v
"""

import asyncio
import time

import pytest

import server


def _run(coro):
    return asyncio.run(coro)


def _answer(idx, content="the answer"):
    return {
        "step_index": idx, "source": "MODEL", "type": "PLANNER_RESPONSE",
        "status": "DONE", "content": content, "created_at": "t",
    }


@pytest.fixture()
def fastpoll(monkeypatch):
    monkeypatch.setattr(server, "RESPONSE_POLL_INTERVAL", 0.02)
    monkeypatch.setattr(server, "RESPONSE_MIN_WAIT", 0.0)


def _session(monkeypatch, *, transcript, screen):
    """Build a session whose transcript/screen are driven by callables."""
    sess = server.AgySession(name="unit")
    sess._conversation_id = "conv"
    sess._turn_count = 1
    monkeypatch.setattr(server, "_read_transcript", lambda cid: transcript())
    monkeypatch.setattr(sess, "_transcript_mtime", lambda: 0.0)  # isolate: progress via busy/steps only

    async def capture():
        return screen()

    monkeypatch.setattr(sess, "_capture", capture)
    return sess


# --- normal completion ------------------------------------------------------

def test_completes_when_answer_present_and_idle(fastpoll, monkeypatch):
    monkeypatch.setattr(server, "RESPONSE_STALL_TIMEOUT", 5.0)
    monkeypatch.setattr(server, "RESPONSE_HARD_TIMEOUT", 10.0)
    sess = _session(
        monkeypatch,
        transcript=lambda: [_answer(0, "final review text")],
        screen=lambda: "all done  ? for shortcuts",
    )
    assert _run(sess._collect_response(baseline=-1)) == "final review text"


# --- the fix: a busy turn past the stall window is NOT cut off --------------

def test_busy_turn_is_not_cut_off_and_then_completes(fastpoll, monkeypatch):
    # stall window is tiny; a flat-timeout impl would give up at 0.3s and lose
    # the answer. Progress-awareness must keep waiting while agy is busy.
    monkeypatch.setattr(server, "RESPONSE_STALL_TIMEOUT", 0.3)
    monkeypatch.setattr(server, "RESPONSE_HARD_TIMEOUT", 10.0)
    t0 = time.monotonic()
    sess = _session(
        monkeypatch,
        transcript=lambda: [_answer(0, "late answer")] if time.monotonic() - t0 > 0.8 else [],
        screen=lambda: ("? for shortcuts" if time.monotonic() - t0 > 0.8
                        else "Generating...  esc to cancel"),
    )
    assert _run(sess._collect_response(baseline=-1)) == "late answer"


# --- genuine stall: idle, no progress, no answer -> give up + diagnostic ----

def test_stalls_when_idle_and_writes_diagnostic(fastpoll, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "RESPONSE_STALL_TIMEOUT", 0.3)
    monkeypatch.setattr(server, "RESPONSE_HARD_TIMEOUT", 10.0)
    monkeypatch.setattr(server, "TIMEOUT_LOG_DIR", tmp_path / "timeouts")
    # a frozen transcript at step 5, no content-bearing planner response
    sess = _session(
        monkeypatch,
        transcript=lambda: [{"step_index": 5, "source": "SYSTEM", "type": "X", "status": "DONE"}],
        screen=lambda: "idle  ? for shortcuts",
    )
    assert _run(sess._collect_response(baseline=5)) == ""
    dumps = list((tmp_path / "timeouts").glob("*.log"))
    assert len(dumps) == 1
    body = dumps[0].read_text()
    assert "reason=stalled" in body and "conversation_id=conv" in body


# --- hard ceiling: busy forever still gets bounded --------------------------

def test_hard_timeout_caps_a_perpetually_busy_turn(fastpoll, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "RESPONSE_STALL_TIMEOUT", 10.0)  # never fires
    monkeypatch.setattr(server, "RESPONSE_HARD_TIMEOUT", 0.4)
    monkeypatch.setattr(server, "TIMEOUT_LOG_DIR", tmp_path / "timeouts")
    sess = _session(
        monkeypatch,
        transcript=lambda: [],
        screen=lambda: "Generating...  esc to cancel",  # always busy
    )
    assert _run(sess._collect_response(baseline=-1)) == ""
    dumps = list((tmp_path / "timeouts").glob("*.log"))
    assert dumps and "reason=hard_timeout" in dumps[0].read_text()
