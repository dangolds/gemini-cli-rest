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
import types
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
    monkeypatch.setattr(server, "TIMEOUT_LOG_DIR", tmp_path / "timeouts")
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


# --- Past the window: keep waiting while agy is visibly working -------------
#
# agy 1.1.27 can sit 20s+ on a post-login loadCodeAssist call between taking
# the prompt and writing the transcript (bridge log 2026-09-06: three 502s at
# 20.7s, agy's own log "Forwarding user message" only 20s after "Streaming
# conversation"). Past CONVERSATION_DETECT_TIMEOUT the wait now continues while
# the process is alive and the screen shows work — a turn spinner or the
# auth/backend "Signing in..." — up to CONVERSATION_DETECT_MAX. Time is virtual:
# asyncio.sleep advances a fake monotonic clock, so a 90s bound runs instantly.

SIGNING_IN_SCREEN = (
    " Welcome to the Antigravity CLI. You are currently not signed in.\n"
    " ⢿  Signing in..."
)
GENERATING_SCREEN = "> hi\n⣻  Generating...\n>\nesc to cancel"
IDLE_SCREEN = "> hi\n>\n? for shortcuts"


class _Clock:
    """Virtual monotonic clock driven by the patched asyncio.sleep."""

    def __init__(self):
        self.now = 1000.0
        self.on_tick = None  # callable(now) run after every sleep

    def monotonic(self):
        return self.now

    async def sleep(self, secs):
        self.now += secs
        if self.on_tick is not None:
            self.on_tick(self.now)


@pytest.fixture()
def clock(monkeypatch):
    c = _Clock()
    fake_time = types.SimpleNamespace(
        monotonic=c.monotonic, time=time.time, mktime=time.mktime,
        strptime=time.strptime, sleep=time.sleep,
    )
    monkeypatch.setattr(server, "time", fake_time)
    monkeypatch.setattr(server.asyncio, "sleep", c.sleep)
    monkeypatch.setattr(server, "CONVERSATION_DETECT_TIMEOUT", 20.0)
    monkeypatch.setattr(server, "CONVERSATION_DETECT_MAX", 90.0)
    return c


def _session(monkeypatch, screen, alive=None):
    """AgySession whose screen is *screen* (str or callable) and whose process
    liveness is *alive* (bool or callable); records every screen capture."""
    sess = server.AgySession(name="unit")
    captures = []

    async def capture():
        captures.append(server.time.monotonic())
        return screen() if callable(screen) else screen

    async def is_alive():
        return True if alive is None else (alive() if callable(alive) else alive)

    monkeypatch.setattr(sess, "_capture", capture)
    monkeypatch.setattr(sess, "is_alive", is_alive)
    return sess, captures


def test_busy_agy_is_waited_for_past_the_window(state_dir, clock, monkeypatch):
    """(a) The transcript shows up at 35s while the screen still shows the
    auth/backend spinner: the id is returned, no error."""
    start = clock.now

    def late_transcript(now):
        if now - start >= 35 and not (state_dir / "brain" / "real").exists():
            _make_brain(state_dir, "real", with_transcript=True)

    clock.on_tick = late_transcript
    sess, captures = _session(monkeypatch, SIGNING_IN_SCREEN)
    assert _run(sess._detect_new_conversation(before=set())) == "real"
    assert 35 <= clock.now - start < 37
    # The extension looks at the screen about once a second, not every poll.
    assert captures and len(captures) <= 17
    assert all(t - start >= 20 for t in captures), "no screen reads inside the window"


def test_idle_screen_past_the_window_gives_up_as_before(state_dir, clock, monkeypatch):
    """(b) Ready marker, no spinner, no transcript: the prompt was dropped —
    fail at the window, exactly as before the extension existed."""
    start = clock.now
    sess, captures = _session(monkeypatch, IDLE_SCREEN)
    with pytest.raises(RuntimeError, match="no new transcript appeared.*screen idle"):
        _run(sess._detect_new_conversation(before=set()))
    assert 20 <= clock.now - start < 21.5
    assert len(captures) == 2  # one look at the window, one for the dump


def test_busy_forever_gives_up_at_the_hard_bound_with_a_dump(state_dir, clock, monkeypatch):
    """(c) agy stays busy but never writes a transcript: give up at
    CONVERSATION_DETECT_MAX, with a diagnostic dump named in the error."""
    start = clock.now
    sess, _ = _session(monkeypatch, GENERATING_SCREEN)
    with pytest.raises(RuntimeError, match="hard bound 90s") as exc:
        _run(sess._detect_new_conversation(before=set()))
    assert 90 <= clock.now - start < 92
    dumps = list((state_dir / "timeouts").glob("unit-*detect-g0-turn0.log"))
    assert len(dumps) == 1 and str(dumps[0]) in str(exc.value)
    text = dumps[0].read_text(encoding="utf-8")
    assert "reason=conversation_detect_timeout" in text
    assert "window=20s max=90s" in text
    assert "Generating..." in text                 # the rendered screen
    assert "=== agy process log ===" in text       # agy's own log section


def test_extend_false_keeps_the_plain_window(state_dir, clock, monkeypatch):
    """(d) The re-paste probe in _submit_first passes extend=False: a busy
    screen does not stretch its window and the screen is not even consulted."""
    start = clock.now
    sess, captures = _session(monkeypatch, GENERATING_SCREEN)
    with pytest.raises(RuntimeError, match="no new transcript appeared"):
        _run(sess._detect_new_conversation(before=set(), timeout=10.0, extend=False))
    assert 10 <= clock.now - start < 11
    assert len(captures) == 1  # only the dump's capture, no busy look


def test_dead_process_during_the_extension_gives_up_at_once(state_dir, clock, monkeypatch):
    """(e) agy exits while we are extending: no point waiting for the bound."""
    start = clock.now
    sess, _ = _session(
        monkeypatch, GENERATING_SCREEN, alive=lambda: clock.now - start < 25
    )
    with pytest.raises(RuntimeError, match="agy process exited"):
        _run(sess._detect_new_conversation(before=set()))
    assert 25 <= clock.now - start < 26.5


def test_submit_first_retry_probe_does_not_extend(state_dir, clock, monkeypatch):
    """The half-window re-paste probe keeps its short window even on a busy
    screen (the initial window may already have spent the extension budget)."""
    monkeypatch.setattr(server, "VERIFY_RESUBMIT_MAX", 1)
    monkeypatch.setattr(server, "VERIFY_RESUBMIT_DELAY", 0.0)
    seen = []
    orig = server.AgySession._detect_new_conversation

    async def spy(self, before, timeout=None, **kw):
        seen.append((timeout, kw.get("extend")))
        if len(seen) == 1:
            return None  # the gate ate paste 1
        return await orig(self, before, timeout, **kw)

    monkeypatch.setattr(server.AgySession, "_detect_new_conversation", spy)

    async def fake_tmux(*args, **kwargs):
        return (0, "")

    monkeypatch.setattr(server, "_tmux", fake_tmux)
    sess, _ = _session(monkeypatch, GENERATING_SCREEN)
    start = clock.now
    with pytest.raises(RuntimeError, match="no new transcript appeared"):
        _run(sess._submit_first("hi", before=set()))
    assert seen == [(None, True), (10.0, False)]
    assert clock.now - start < 11


def test_max_wait_caps_the_extension_never_the_window(state_dir, clock, monkeypatch):
    """send() passes what is left of the /chat budget: the busy extension stops
    there, while the base window is always granted."""
    start = clock.now
    sess, _ = _session(monkeypatch, GENERATING_SCREEN)
    with pytest.raises(RuntimeError, match="hard bound 30s"):
        _run(sess._detect_new_conversation(before=set(), max_wait=30.0))
    assert 30 <= clock.now - start < 32
    start = clock.now
    sess, _ = _session(monkeypatch, GENERATING_SCREEN)
    with pytest.raises(RuntimeError, match="no new transcript appeared"):
        _run(sess._detect_new_conversation(before=set(), max_wait=5.0))
    assert 20 <= clock.now - start < 21.5  # never below the window
