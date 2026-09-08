"""
Unit tests for AgySession._submit_first (agy's account-verification gate).

agy verifies account eligibility once per process launch, and a prompt
submitted during that window is CONSUMED: the TUI prints "Verifying your
account", returns to an empty input box and never starts a turn — no brain dir,
no transcript — so turn-1 conversation detection burned its window and the
request 502'd. Recovery cannot be a second Enter (the text is gone from the
input box, which is why _submit_confirmed's retry is useless here): the whole
prompt has to be re-pasted.

The safety property under test is NO DUPLICATE TURNS. The notice STAYS on the
screen once printed — even after a later submit succeeds — so the marker alone
must never trigger a re-paste; it only counts together with an idle screen and
no candidate transcript.

No server / agy / Docker: tmux and the screen are mocked, brain dirs are real
temp dirs (same fixtures as test_detect_conversation.py / test_submit_confirm.py).
Run:  ./.venv/bin/python -m pytest test_verify_gate.py -v
"""

import asyncio
import threading
import time
from pathlib import Path

import pytest

import server


# What agy renders when the gate eats a prompt (idle input box underneath), and
# the same screen once agy is actually working on the turn.
GATE_SCREEN = (
    "⚠ Verifying your account...\n"
    "  ⎿  We're finishing verifying your account eligibility.\n"
    "     This usually takes a moment. Please try again shortly.\n"
    "                                                 ? for shortcuts"
)
GATE_SCREEN_BUSY = GATE_SCREEN + "\nGenerating...  esc to cancel"


def _run(coro):
    """Drive an async method from a sync test without pytest-asyncio."""
    return asyncio.run(coro)


def _make_brain(state_dir, cid):
    """Create brain/<cid>/ with a populated transcript.jsonl (a real turn)."""
    logs = Path(state_dir) / "brain" / cid / ".system_generated" / "logs"
    logs.mkdir(parents=True)
    (logs / "transcript.jsonl").write_text(
        '{"step_index":0,"source":"USER_EXPLICIT","type":"USER_INPUT",'
        '"status":"DONE","content":"hi"}\n'
    )
    return cid


@pytest.fixture()
def state_dir(tmp_path, monkeypatch):
    """Point the module's AGY_STATE_DIR at an isolated temp tree."""
    monkeypatch.setattr(server, "AGY_STATE_DIR", tmp_path)
    monkeypatch.setattr(server, "TIMEOUT_LOG_DIR", tmp_path / "timeouts")
    return tmp_path


@pytest.fixture()
def fast(monkeypatch):
    """Shrink the timing constants so windows elapse in fractions of a second."""
    monkeypatch.setattr(server, "CONVERSATION_DETECT_TIMEOUT", 1.0)
    monkeypatch.setattr(server, "CONVERSATION_DETECT_MAX", 1.5)
    monkeypatch.setattr(server, "VERIFY_RESUBMIT_DELAY", 0.0)
    monkeypatch.setattr(server, "VERIFY_RESUBMIT_MAX", 2)


class _Pastes(list):
    """Recorded paste-buffer calls, with a hook fired on each one."""
    on_paste = None


@pytest.fixture()
def pastes(monkeypatch):
    """Replace _tmux with a recorder; returns the list of prompt pastes.

    One entry per _submit (load-buffer + paste-buffer + Enter), so its length
    is exactly the number of times the prompt reached agy — the duplicate-turn
    counter every test below asserts on. Assign `pastes.on_paste` to make the
    Nth paste have a side effect (e.g. agy finally starting the turn).
    """
    sent = _Pastes()

    async def fake_tmux(*args, **kwargs):
        if args and args[0] == "paste-buffer":
            sent.append(args)
            if sent.on_paste is not None:
                sent.on_paste(len(sent))
        return (0, "")

    monkeypatch.setattr(server, "_tmux", fake_tmux)
    return sent


def _session(monkeypatch, screen):
    """AgySession whose screen is *screen* (a str, or a callable per capture)."""
    sess = server.AgySession(name="unit")

    async def capture():
        return screen() if callable(screen) else screen

    monkeypatch.setattr(sess, "_capture", capture)
    return sess


# --- the gate ate the prompt: re-paste it -----------------------------------

def test_gate_drop_is_resubmitted_and_then_detected(fast, pastes, state_dir, monkeypatch):
    """The gate drops paste 1; paste 2 lands and agy writes a transcript."""
    def turn_starts_on_the_resubmit(n):
        if n == 2:
            _make_brain(state_dir, "real")

    pastes.on_paste = turn_starts_on_the_resubmit
    # The notice is still on screen after the resubmit lands (it never clears),
    # which must not cost a third paste.
    sess = _session(monkeypatch, GATE_SCREEN)
    assert _run(sess._submit_first("hi", before=set())) == "real"
    assert len(pastes) == 2, "exactly one re-paste: the original plus one retry"


def test_send_resolves_the_conversation_after_a_gate_drop(fast, pastes, state_dir, monkeypatch):
    """Turn 1 of send() goes through the recovery and answers normally."""
    def turn_starts_on_the_resubmit(n):
        if n == 2:
            _make_brain(state_dir, "real")

    pastes.on_paste = turn_starts_on_the_resubmit
    monkeypatch.setattr(server, "_SPAWN_LOCK", asyncio.Lock())  # fresh per loop
    sess = _session(monkeypatch, GATE_SCREEN)

    async def collect(baseline, **kw):
        return "answer"

    monkeypatch.setattr(sess, "_collect_response", collect)
    assert _run(sess.send("hi")) == "answer"
    assert sess._conversation_id == "real"
    assert len(pastes) == 2


# --- agy took the prompt: never re-paste (would duplicate the turn) ---------

def test_busy_screen_is_never_resubmitted(fast, pastes, state_dir, monkeypatch):
    """Marker on screen but agy is generating -> it took the prompt."""
    sess = _session(monkeypatch, GATE_SCREEN_BUSY)
    with pytest.raises(RuntimeError, match="no new transcript appeared"):
        _run(sess._submit_first("hi", before=set()))
    assert len(pastes) == 1, "busy means agy accepted it; re-pasting would duplicate"


def test_existing_candidate_is_never_resubmitted(fast, pastes, state_dir, monkeypatch):
    """A transcript exists, so the notice on screen is leftover, not a drop."""
    _make_brain(state_dir, "real")
    sess = _session(monkeypatch, GATE_SCREEN)
    assert _run(sess._submit_first("hi", before=set())) == "real"
    assert len(pastes) == 1


def test_settle_grace_lets_a_late_transcript_win(pastes, state_dir, monkeypatch):
    """The idle notice is on screen from the first poll while agy is still
    starting the turn; the grace must hold the re-paste back long enough for
    the transcript to show up."""
    monkeypatch.setattr(server, "CONVERSATION_DETECT_TIMEOUT", 3.0)
    monkeypatch.setattr(server, "VERIFY_RESUBMIT_DELAY", 1.0)

    def _delayed_real():
        time.sleep(0.2)
        _make_brain(state_dir, "real")

    worker = threading.Thread(target=_delayed_real)
    worker.start()
    try:
        sess = _session(monkeypatch, GATE_SCREEN)
        assert _run(sess._submit_first("hi", before=set())) == "real"
    finally:
        worker.join()
    assert len(pastes) == 1


# --- bounded retries, and the kill switch -----------------------------------

def test_gives_up_after_max_resubmits(fast, pastes, state_dir, monkeypatch):
    """The gate never lifts: bounded re-pastes, then an error that names it."""
    sess = _session(monkeypatch, GATE_SCREEN)
    with pytest.raises(RuntimeError, match="account-verification gate"):
        _run(sess._submit_first("hi", before=set()))
    assert len(pastes) == 1 + server.VERIFY_RESUBMIT_MAX


def test_max_zero_restores_the_old_single_submit_behavior(fast, pastes, state_dir, monkeypatch):
    monkeypatch.setattr(server, "VERIFY_RESUBMIT_MAX", 0)
    sess = _session(monkeypatch, GATE_SCREEN)
    with pytest.raises(RuntimeError, match="no new transcript appeared"):
        _run(sess._submit_first("hi", before=set()))
    assert len(pastes) == 1
