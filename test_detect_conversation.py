"""
Unit tests for AgySession._detect_new_conversation.

These are pure-filesystem tests — no live server, no agy, no Docker. They
pin down the conversation-id detection that the response loop depends on:
the bridge polls ONE brain dir's transcript.jsonl for the whole turn, so if
detection picks the wrong dir every poll returns nothing and the turn only
ends at the max timeout ("0 message(s) via max_timeout") while the TUI already
showed the reply. agy creates empty placeholder brain dirs (no transcript ever
written); detection must skip them and select the dir that actually holds a
transcript.

Run standalone:  ./.venv/bin/python -m pytest test_detect_conversation.py -v
"""

import asyncio
import os
import threading
import time
from pathlib import Path

import pytest

import server


def _run(coro):
    """Drive an async method from a sync test without pytest-asyncio."""
    return asyncio.run(coro)


def _make_brain(state_dir, cid, *, with_transcript, mtime=None):
    """Create brain/<cid>/, optionally with a populated transcript.jsonl.

    When *with_transcript* is False this mimics the empty placeholder dirs agy
    leaves behind. *mtime* (epoch seconds) lets a test force one dir to look
    newer than another so we can prove selection ignores raw dir mtime.
    """
    bdir = Path(state_dir) / "brain" / cid
    if with_transcript:
        logs = bdir / ".system_generated" / "logs"
        logs.mkdir(parents=True)
        tpath = logs / "transcript.jsonl"
        tpath.write_text(
            '{"step_index":0,"source":"USER_EXPLICIT","type":"USER_INPUT",'
            '"status":"DONE","content":"hi"}\n'
        )
        if mtime is not None:
            os.utime(tpath, (mtime, mtime))
    else:
        bdir.mkdir(parents=True)
    if mtime is not None:
        os.utime(bdir, (mtime, mtime))
    return cid


@pytest.fixture()
def state_dir(tmp_path, monkeypatch):
    """Point the module's AGY_STATE_DIR at an isolated temp tree."""
    monkeypatch.setattr(server, "AGY_STATE_DIR", tmp_path)
    return tmp_path


# --- The 90% happy path must keep working -----------------------------------

def test_returns_the_single_new_conversation(state_dir):
    _make_brain(state_dir, "real", with_transcript=True)
    sess = server.AgySession(name="unit")
    assert _run(sess._detect_new_conversation(before=set())) == "real"


def test_ignores_preexisting_conversations(state_dir):
    """Dirs present before this turn (in `before`) are never selected."""
    _make_brain(state_dir, "old", with_transcript=True)
    _make_brain(state_dir, "new", with_transcript=True)
    sess = server.AgySession(name="unit")
    assert _run(sess._detect_new_conversation(before={"old"})) == "new"


def test_newest_transcript_wins_among_real_conversations(state_dir):
    now = time.time()
    _make_brain(state_dir, "older", with_transcript=True, mtime=now - 10)
    _make_brain(state_dir, "newer", with_transcript=True, mtime=now)
    sess = server.AgySession(name="unit")
    assert _run(sess._detect_new_conversation(before=set())) == "newer"


# --- The 10% bug this fix targets -------------------------------------------

def test_skips_empty_placeholder_even_when_it_is_the_newest_dir(state_dir):
    """Regression: the empty placeholder has the newest *dir* mtime, so the
    old 'newest dir wins' logic latched onto it and every transcript read came
    back empty -> max_timeout / 0 messages. Detection must pick the dir that
    has a transcript instead."""
    now = time.time()
    _make_brain(state_dir, "real", with_transcript=True, mtime=now - 5)
    _make_brain(state_dir, "ghost", with_transcript=False, mtime=now)  # newest dir
    sess = server.AgySession(name="unit")
    assert _run(sess._detect_new_conversation(before=set())) == "real"


def test_waits_for_a_late_transcript_past_an_empty_placeholder(state_dir, monkeypatch):
    """The empty placeholder shows up first; the real transcript appears a
    beat later. Detection must keep polling and return the real one, not bail
    early on the placeholder."""
    monkeypatch.setattr(server, "CONVERSATION_DETECT_TIMEOUT", 5.0)
    _make_brain(state_dir, "ghost", with_transcript=False)

    def _delayed_real():
        time.sleep(0.7)
        _make_brain(state_dir, "real", with_transcript=True)

    worker = threading.Thread(target=_delayed_real)
    worker.start()
    try:
        sess = server.AgySession(name="unit")
        assert _run(sess._detect_new_conversation(before=set())) == "real"
    finally:
        worker.join()


def test_raises_when_only_an_empty_placeholder_ever_appears(state_dir, monkeypatch):
    """No transcript ever arrives -> a clear error rather than a silent wrong
    pick that would later time out as '0 message(s)'."""
    monkeypatch.setattr(server, "CONVERSATION_DETECT_TIMEOUT", 0.3)
    _make_brain(state_dir, "ghost", with_transcript=False)
    sess = server.AgySession(name="unit")
    with pytest.raises(RuntimeError, match="conversation id"):
        _run(sess._detect_new_conversation(before=set()))
