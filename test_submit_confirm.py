"""
Unit tests for AgySession._submit_confirmed / _await_ingest.

These pin the behavior that actually bit the review harness: agy sometimes
swallows the submit Enter (the pasted prompt sits unsubmitted while the screen
looks idle), so the bridge polled an unchanging transcript until the 140s max
timeout and logged "0 message(s)" — even though agy answered instantly the
moment the input finally registered.

The safety property under test is NO DUPLICATE MESSAGES: we only re-press Enter
when agy is idle AND has ingested nothing for a full confirm window. If agy
took the input (a new transcript step, or a busy screen) we must never re-send.

No server / agy / Docker — everything agy-facing is mocked.
Run:  ./.venv/bin/python -m pytest test_submit_confirm.py -v
"""

import asyncio

import pytest

import server


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def fast(monkeypatch):
    """Shrink the timing constants so windows elapse in fractions of a second."""
    monkeypatch.setattr(server, "SUBMIT_CONFIRM_WAIT", 0.25)
    monkeypatch.setattr(server, "RESPONSE_POLL_INTERVAL", 0.02)


@pytest.fixture()
def enters(monkeypatch):
    """Replace _tmux with a recorder; returns the list of Enter keystrokes."""
    sent = []

    async def fake_tmux(*args, **kwargs):
        if args and args[0] == "send-keys" and args[-1] == "Enter":
            sent.append(args)
        return (0, "")

    monkeypatch.setattr(server, "_tmux", fake_tmux)
    return sent


def _session(monkeypatch, *, screen):
    sess = server.AgySession(name="unit")
    sess._conversation_id = "conv"

    async def capture():
        return screen

    monkeypatch.setattr(sess, "_capture", capture)
    return sess


# --- the input was accepted: never re-send -> no duplicate ------------------

def test_accepted_via_new_step_does_not_resend(fast, enters, monkeypatch):
    # transcript already shows a step beyond baseline on the first read
    monkeypatch.setattr(server, "_read_transcript", lambda cid: [{"step_index": 6}])
    sess = _session(monkeypatch, screen="all idle  ? for shortcuts")
    _run(sess._submit_confirmed("hi", baseline=5))
    assert len(enters) == 1, "accepted submit must not be re-sent (would duplicate)"


def test_busy_screen_counts_as_accepted_does_not_resend(fast, enters, monkeypatch):
    # no new transcript step, but agy is visibly generating -> it took the input
    monkeypatch.setattr(server, "_read_transcript", lambda cid: [{"step_index": 5}])
    sess = _session(monkeypatch, screen="Generating...  esc to cancel")
    _run(sess._submit_confirmed("hi", baseline=5))
    assert len(enters) == 1, "busy means agy accepted the turn; re-sending would duplicate"


# --- the Enter was dropped: re-press exactly once ---------------------------

def test_dropped_submit_is_resent_once_then_ingested(fast, enters, monkeypatch):
    # the transcript only advances once a *second* Enter (the resend) lands —
    # exactly how a dropped submit behaves: nothing is logged until it's resent
    def transcript(cid):
        return [{"step_index": 6 if len(enters) >= 2 else 5}]

    monkeypatch.setattr(server, "_read_transcript", transcript)
    sess = _session(monkeypatch, screen="idle, no markers  ? for shortcuts")
    _run(sess._submit_confirmed("hi", baseline=5))
    assert len(enters) == 2, "exactly one resend: the original Enter plus one retry"


def test_gives_up_after_max_retries_without_spamming(fast, enters, monkeypatch):
    # agy never ingests (transcript frozen, screen idle): bounded resends only
    monkeypatch.setattr(server, "_read_transcript", lambda cid: [{"step_index": 5}])
    sess = _session(monkeypatch, screen="idle  ? for shortcuts")
    _run(sess._submit_confirmed("hi", baseline=5))
    # 1 original + SUBMIT_MAX_RETRIES resends, then it stops (no infinite loop)
    assert len(enters) == 1 + server.SUBMIT_MAX_RETRIES
