"""
Unit tests for the codex bridge's rollout-driven internals.

These mirror the agy bridge's unit tests (test_collect_response /
test_detect_conversation / test_submit_confirm) but against codex's rollout
JSONL instead of agy's transcript. They cover the three things correctness
rides on:

  * parsing — extracting the turn's final answer from rollout events
  * _collect_response — completion on a NEW task_complete, in-flight turns are
    not cut off, genuine stalls and the hard cap are bounded + dumped
  * _detect_new_session — pick the new rollout tagged with this session's cwd
  * _submit_confirmed/_await_ingest — re-press a dropped Enter at most once,
    never duplicate an accepted submit

No server / codex / Docker — the rollout, the screen, and tmux are mocked.
Run:  ./.venv/bin/python -m pytest test_codex_rollout.py -v
"""

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

import codex_server


def _run(coro):
    return asyncio.run(coro)


# --- event builders ---------------------------------------------------------

def _meta(session_id="s1", cwd="/x"):
    return {"type": "session_meta", "payload": {"id": session_id, "cwd": cwd}}


def _start(turn_id="t1"):
    return {"type": "event_msg", "payload": {"type": "task_started", "turn_id": turn_id},
            "timestamp": "t"}


def _complete(turn_id="t1", last="the answer"):
    return {"type": "event_msg",
            "payload": {"type": "task_complete", "turn_id": turn_id, "last_agent_message": last},
            "timestamp": "t"}


def _assistant(text):
    return {"type": "response_item",
            "payload": {"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": text}]},
            "timestamp": "t"}


def _user(text):
    return {"type": "response_item",
            "payload": {"type": "message", "role": "user",
                        "content": [{"type": "input_text", "text": text}]},
            "timestamp": "t"}


# ===========================================================================
# Parsing
# ===========================================================================

class TestParsing:
    def test_count_helpers(self):
        events = [_meta(), _start("t1"), _complete("t1"), _start("t2")]
        assert codex_server._count_task_starts(events) == 2
        assert codex_server._count_task_completes(events) == 1

    def test_answer_prefers_last_agent_message(self):
        events = [_meta(), _start(), _assistant("streamed text"), _complete(last="final answer")]
        assert codex_server._answer_for_new_turn(events, baseline_completes=0) == "final answer"

    def test_answer_falls_back_to_assistant_text_when_null(self):
        events = [_meta(), _start(), _assistant("visible reply"), _complete(last=None)]
        assert codex_server._answer_for_new_turn(events, baseline_completes=0) == "visible reply"

    def test_answer_scopes_to_the_new_turn_only(self):
        """With a prior completed turn, baseline=1 must return only turn 2's answer."""
        events = [
            _meta(),
            _start("t1"), _assistant("turn one reply"), _complete("t1", last="answer one"),
            _start("t2"), _assistant("turn two reply"), _complete("t2", last="answer two"),
        ]
        assert codex_server._answer_for_new_turn(events, baseline_completes=1) == "answer two"

    def test_rollout_meta_reads_session_meta(self, tmp_path):
        p = tmp_path / "rollout-x.jsonl"
        p.write_text(json.dumps(_meta("sess-9", "/work/dir")) + "\n" + json.dumps(_start()) + "\n")
        meta = codex_server._rollout_meta(p)
        assert meta and meta["id"] == "sess-9" and meta["cwd"] == "/work/dir"


# ===========================================================================
# _collect_response
# ===========================================================================

@pytest.fixture()
def fastpoll(monkeypatch):
    monkeypatch.setattr(codex_server, "RESPONSE_POLL_INTERVAL", 0.02)
    monkeypatch.setattr(codex_server, "RESPONSE_MIN_WAIT", 0.0)
    # These tests pin the legacy fallback path: wake fast and run the full
    # check on every wake, so the notify cadence knobs can't starve it.
    monkeypatch.setattr(codex_server, "RESPONSE_FAST_POLL", 0.02)
    monkeypatch.setattr(codex_server, "RESPONSE_FULL_CHECK_EVERY", 1)


def _collect_session(monkeypatch, *, events, screen="idle  Context 0% used", mtime=0.0):
    """A session whose rollout/screen/mtime are driven by callables."""
    sess = codex_server.CodexSession(name="unit")
    sess._rollout_path = Path("/tmp/does-not-matter/rollout-x.jsonl")
    sess._session_id = "s1"
    sess._turn_count = 1
    monkeypatch.setattr(codex_server, "_read_rollout", lambda path: events())
    monkeypatch.setattr(sess, "_rollout_mtime", (mtime if callable(mtime) else (lambda: mtime)))

    async def capture():
        return screen() if callable(screen) else screen

    monkeypatch.setattr(sess, "_capture", capture)
    return sess


def test_completes_when_new_task_complete_appears(fastpoll, monkeypatch):
    monkeypatch.setattr(codex_server, "RESPONSE_STALL_TIMEOUT", 5.0)
    monkeypatch.setattr(codex_server, "RESPONSE_HARD_TIMEOUT", 10.0)
    sess = _collect_session(
        monkeypatch,
        events=lambda: [_meta(), _start(), _assistant("x"), _complete(last="final review text")],
    )
    assert _run(sess._collect_response(baseline_completes=0, baseline_starts=0)) == "final review text"


def test_in_flight_turn_is_not_cut_off_then_completes(fastpoll, monkeypatch):
    # A started-but-not-completed turn writes nothing else (long thinking). The
    # rollout mtime is frozen, so ONLY in-flight detection keeps it alive past a
    # tiny stall window; a naive impl would give up at 0.3s and lose the answer.
    monkeypatch.setattr(codex_server, "RESPONSE_STALL_TIMEOUT", 0.3)
    monkeypatch.setattr(codex_server, "RESPONSE_HARD_TIMEOUT", 10.0)
    t0 = time.monotonic()

    def events():
        if time.monotonic() - t0 > 0.8:
            return [_meta(), _start(), _complete(last="late answer")]
        return [_meta(), _start()]  # in-flight: started, not completed

    sess = _collect_session(monkeypatch, events=events, mtime=0.0)  # frozen mtime
    assert _run(sess._collect_response(baseline_completes=0, baseline_starts=0)) == "late answer"


def test_stalls_when_idle_and_writes_diagnostic(fastpoll, monkeypatch, tmp_path):
    monkeypatch.setattr(codex_server, "RESPONSE_STALL_TIMEOUT", 0.3)
    monkeypatch.setattr(codex_server, "RESPONSE_HARD_TIMEOUT", 10.0)
    monkeypatch.setattr(codex_server, "TIMEOUT_LOG_DIR", tmp_path / "timeouts")
    # nothing ever started (a dropped submit that confirm logic let through):
    # no task_started, no task_complete, frozen mtime -> genuine stall.
    sess = _collect_session(monkeypatch, events=lambda: [_meta()], mtime=0.0)
    assert _run(sess._collect_response(baseline_completes=0, baseline_starts=0)) == ""
    dumps = list((tmp_path / "timeouts").glob("*.log"))
    assert len(dumps) == 1
    assert "reason=stalled" in dumps[0].read_text()


def test_hard_timeout_caps_a_perpetually_in_flight_turn(fastpoll, monkeypatch, tmp_path):
    monkeypatch.setattr(codex_server, "RESPONSE_STALL_TIMEOUT", 10.0)  # never fires
    monkeypatch.setattr(codex_server, "RESPONSE_HARD_TIMEOUT", 0.4)
    monkeypatch.setattr(codex_server, "TIMEOUT_LOG_DIR", tmp_path / "timeouts")
    # started but never completes -> in-flight forever; only the hard cap bounds it
    sess = _collect_session(monkeypatch, events=lambda: [_meta(), _start()], mtime=0.0)
    assert _run(sess._collect_response(baseline_completes=0, baseline_starts=0)) == ""
    dumps = list((tmp_path / "timeouts").glob("*.log"))
    assert dumps and "reason=hard_timeout" in dumps[0].read_text()


# ===========================================================================
# _detect_new_session
# ===========================================================================

def _write_rollout(sessions_dir, name, *, cwd, mtime=None):
    sub = Path(sessions_dir) / "2026" / "06" / "22"
    sub.mkdir(parents=True, exist_ok=True)
    p = sub / f"rollout-{name}.jsonl"
    p.write_text(json.dumps(_meta(session_id=name, cwd=cwd)) + "\n")
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return str(p)


@pytest.fixture()
def sessions_dir(tmp_path, monkeypatch):
    d = tmp_path / "sessions"
    d.mkdir()
    monkeypatch.setattr(codex_server, "CODEX_SESSIONS_DIR", d)
    return d


def test_returns_the_new_rollout_for_our_cwd(sessions_dir, monkeypatch):
    monkeypatch.setattr(codex_server, "SESSION_DETECT_TIMEOUT", 2.0)
    sess = codex_server.CodexSession(name="unit")
    _write_rollout(sessions_dir, "ours", cwd=str(sess.cwd))
    got = _run(sess._detect_new_session(before=set()))
    assert codex_server._rollout_meta(got)["id"] == "ours"


def test_ignores_rollouts_present_before_and_other_cwds(sessions_dir, monkeypatch):
    monkeypatch.setattr(codex_server, "SESSION_DETECT_TIMEOUT", 2.0)
    sess = codex_server.CodexSession(name="unit")
    pre = _write_rollout(sessions_dir, "old", cwd=str(sess.cwd))      # excluded by `before`
    _write_rollout(sessions_dir, "other", cwd="/some/other/cwd")       # wrong cwd
    new = _write_rollout(sessions_dir, "new", cwd=str(sess.cwd))
    got = _run(sess._detect_new_session(before={pre}))
    assert str(got) == new


def test_newest_rollout_wins_among_matches(sessions_dir, monkeypatch):
    monkeypatch.setattr(codex_server, "SESSION_DETECT_TIMEOUT", 2.0)
    sess = codex_server.CodexSession(name="unit")
    now = time.time()
    _write_rollout(sessions_dir, "older", cwd=str(sess.cwd), mtime=now - 10)
    newer = _write_rollout(sessions_dir, "newer", cwd=str(sess.cwd), mtime=now)
    got = _run(sess._detect_new_session(before=set()))
    assert str(got) == newer


def test_raises_when_no_matching_rollout_appears(sessions_dir, monkeypatch):
    monkeypatch.setattr(codex_server, "SESSION_DETECT_TIMEOUT", 0.3)
    sess = codex_server.CodexSession(name="unit")
    _write_rollout(sessions_dir, "other", cwd="/not/ours")
    with pytest.raises(RuntimeError, match="codex session"):
        _run(sess._detect_new_session(before=set()))


# ===========================================================================
# _submit_confirmed / _await_ingest
# ===========================================================================

@pytest.fixture()
def fast_submit(monkeypatch):
    monkeypatch.setattr(codex_server, "SUBMIT_CONFIRM_WAIT", 0.25)
    monkeypatch.setattr(codex_server, "RESPONSE_POLL_INTERVAL", 0.02)


@pytest.fixture()
def enters(monkeypatch):
    """Replace _tmux with a recorder; returns the list of Enter keystrokes."""
    sent = []

    async def fake_tmux(*args, **kwargs):
        if args and args[0] == "send-keys" and args[-1] == "Enter":
            sent.append(args)
        return (0, "")

    monkeypatch.setattr(codex_server, "_tmux", fake_tmux)
    return sent


def _submit_session(monkeypatch, *, screen):
    sess = codex_server.CodexSession(name="unit")
    sess._rollout_path = Path("/tmp/does-not-matter/rollout-x.jsonl")

    async def capture():
        return screen

    monkeypatch.setattr(sess, "_capture", capture)
    return sess


def test_accepted_via_new_start_does_not_resend(fast_submit, enters, monkeypatch):
    # a new task_started is already present on the first read -> ingested
    monkeypatch.setattr(codex_server, "_read_rollout", lambda path: [_start("t1"), _start("t2")])
    sess = _submit_session(monkeypatch, screen="idle  Context 0% used")
    _run(sess._submit_confirmed("hi", baseline_starts=1))
    assert len(enters) == 1, "accepted submit must not be re-sent (would duplicate)"


def test_busy_screen_counts_as_accepted_does_not_resend(fast_submit, enters, monkeypatch):
    # no new task_started, but codex is visibly working -> it took the input
    monkeypatch.setattr(codex_server, "_read_rollout", lambda path: [_start("t1")])
    sess = _submit_session(monkeypatch, screen="thinking…  Esc to interrupt")
    _run(sess._submit_confirmed("hi", baseline_starts=1))
    assert len(enters) == 1, "busy means codex accepted the turn; re-sending would duplicate"


def test_dropped_submit_is_resent_once_then_ingested(fast_submit, enters, monkeypatch):
    # task_started count only advances once a SECOND Enter (the resend) lands —
    # exactly how a dropped submit behaves: nothing logged until it is resent.
    def read_rollout(path):
        return [_start("t1"), _start("t2")] if len(enters) >= 2 else [_start("t1")]

    monkeypatch.setattr(codex_server, "_read_rollout", read_rollout)
    sess = _submit_session(monkeypatch, screen="idle, no markers  Context 0% used")
    _run(sess._submit_confirmed("hi", baseline_starts=1))
    assert len(enters) == 2, "exactly one resend: the original Enter plus one retry"


def test_gives_up_after_max_retries_without_spamming(fast_submit, enters, monkeypatch):
    # codex never ingests (no new task_started, screen idle): bounded resends only
    monkeypatch.setattr(codex_server, "_read_rollout", lambda path: [_start("t1")])
    sess = _submit_session(monkeypatch, screen="idle  Context 0% used")
    _run(sess._submit_confirmed("hi", baseline_starts=1))
    assert len(enters) == 1 + codex_server.SUBMIT_MAX_RETRIES
