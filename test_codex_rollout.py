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
  * _submit_first — turn 1 has no rollout to confirm against, so detection IS
    the confirmation: a miss means the prompt never reached codex and is
    recovered (Enter re-press, or a bounded re-paste when the composer is
    empty) instead of 502'ing with the prompt left sitting in the TUI

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
    monkeypatch.setattr(codex_server, "PASTE_VISIBLE_WAIT", 0.05)


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


# ===========================================================================
# The dropped submit on screen: the paste landed, the Enter was eaten
# ===========================================================================

# codex's composer: the caret line holds the paste placeholder when a prompt is
# waiting unsubmitted, and a rotating placeholder HINT when it is empty (an
# empty composer is not a blank line — which is why "empty" can only be read as
# "the pasted prompt is not there").
PROMPT = "please review the diff I just sent"
STATUS = "  gpt-5.6-sol high · /tmp/codex-rest-sessions/x/c1"
UNSUBMITTED = f"• some earlier output\n\n» [Pasted Content 6534 chars]\n\n{STATUS}"
EMPTY_COMPOSER = f"• some earlier output\n\n› Improve documentation in @filename\n\n{STATUS}"


@pytest.fixture()
def tmux(monkeypatch):
    """Recorder for BOTH corrective actions: .enters and .pastes.

    A paste too many is a duplicated turn, an Enter too many is free — so these
    tests assert on both counts. `on_enter` fires per Enter so a test can make
    the Nth one land (codex finally starting the turn).
    """
    class _Rec:
        def __init__(self):
            self.enters: list = []
            self.pastes: list = []
            self.on_enter = None

    rec = _Rec()

    async def fake_tmux(*args, **kwargs):
        if args and args[0] == "send-keys" and args[-1] == "Enter":
            rec.enters.append(args)
            if rec.on_enter is not None:
                rec.on_enter(len(rec.enters))
        elif args and args[0] == "paste-buffer":
            rec.pastes.append(args)
        return (0, "")

    monkeypatch.setattr(codex_server, "_tmux", fake_tmux)
    return rec


def test_unsubmitted_paste_is_recovered_by_enter_not_a_repaste(fast_submit, tmux, monkeypatch):
    """The incident shape: "» [Pasted Content 6534 chars]" sits in the composer
    with nothing in the rollout. The prompt is THERE, so Enter is the fix — a
    re-paste would submit it twice once the first Enter finally registers."""
    def read_rollout(path):
        return [_start("t1"), _start("t2")] if len(tmux.enters) >= 2 else [_start("t1")]

    monkeypatch.setattr(codex_server, "_read_rollout", read_rollout)
    sess = _submit_session(monkeypatch, screen=UNSUBMITTED)
    _run(sess._submit_confirmed(PROMPT, baseline_starts=1))
    assert len(tmux.pastes) == 1, "the prompt is still in the composer; re-pasting duplicates it"
    assert len(tmux.enters) == 2, "exactly one re-press: the original Enter plus one retry"


def test_empty_composer_is_repasted_once_then_ingested(fast_submit, tmux, monkeypatch):
    # Composer empty and the rollout frozen: an Enter would send nothing.
    monkeypatch.setattr(codex_server, "SUBMIT_REPASTE_DELAY", 0.0)

    def read_rollout(path):
        return [_start("t1"), _start("t2")] if len(tmux.pastes) >= 2 else [_start("t1")]

    monkeypatch.setattr(codex_server, "_read_rollout", read_rollout)
    sess = _submit_session(monkeypatch, screen=EMPTY_COMPOSER)
    _run(sess._submit_confirmed(PROMPT, baseline_starts=1))
    assert len(tmux.pastes) == 2, "exactly one re-paste: the original plus one retry"


def test_empty_composer_repaste_is_aborted_by_a_new_task_started(fast_submit, tmux, monkeypatch):
    """The duplicate guard: codex starts the turn during the settle delay, so
    the re-paste decision is stale by the time it would fire and is dropped."""
    monkeypatch.setattr(codex_server, "SUBMIT_REPASTE_DELAY", 0.5)
    t0 = time.monotonic()
    monkeypatch.setattr(
        codex_server, "_read_rollout",
        lambda path: [_start("t1"), _start("t2")] if time.monotonic() - t0 > 0.4
        else [_start("t1")],
    )
    sess = _submit_session(monkeypatch, screen=EMPTY_COMPOSER)
    _run(sess._submit_confirmed(PROMPT, baseline_starts=1))
    assert len(tmux.pastes) == 1, "codex took the prompt after all — re-pasting duplicates it"


class TestComposerEmpty:
    """Conservative by contract: True only when the pasted prompt is provably
    NOT in the composer, because a True becomes a re-paste."""

    def test_paste_placeholder_means_not_empty(self):
        assert codex_server.CodexSession._composer_empty(UNSUBMITTED, PROMPT) is False

    def test_placeholder_hint_means_empty(self):
        assert codex_server.CodexSession._composer_empty(EMPTY_COMPOSER, PROMPT) is True

    def test_small_prompt_shown_verbatim_means_not_empty(self):
        screen = f"• output\n\n› review the diff I just sent\n\n{STATUS}"
        assert codex_server.CodexSession._composer_empty(
            screen, "review the diff I just sent") is False

    def test_history_echo_above_the_composer_does_not_answer_for_it(self):
        # The prompt is echoed in the transcript ABOVE the composer, which is
        # empty: only the last caret line counts, so this is still a drop.
        screen = f"› review the diff I just sent\n• reply\n\n› Try /skills\n\n{STATUS}"
        assert codex_server.CodexSession._composer_empty(
            screen, "review the diff I just sent") is True

    def test_no_composer_on_screen_is_not_empty(self):
        assert codex_server.CodexSession._composer_empty("Update available\n[y/n]", PROMPT) is False


# ===========================================================================
# _submit_first — turn 1, where detection IS the ingestion check
# ===========================================================================

def _first_session(monkeypatch, *, screen):
    sess = codex_server.CodexSession(name="unit")

    async def capture():
        return screen() if callable(screen) else screen

    monkeypatch.setattr(sess, "_capture", capture)
    return sess


def test_first_prompt_dropped_at_startup_is_recovered_by_enter(
        fast_submit, tmux, sessions_dir, monkeypatch):
    """The codex incident: the paste landed into a 0.5s-old TUI, the Enter was
    eaten, no rollout was ever written and the request 502'd. The retry must
    recover the turn instead."""
    monkeypatch.setattr(codex_server, "SESSION_DETECT_TIMEOUT", 0.6)
    sess = _first_session(monkeypatch, screen=UNSUBMITTED)
    tmux.on_enter = lambda n: (
        _write_rollout(sessions_dir, "ours", cwd=str(sess.cwd)) if n == 2 else None
    )
    got = _run(sess._submit_first(PROMPT, before=set()))
    assert codex_server._rollout_meta(got)["id"] == "ours"
    assert len(tmux.pastes) == 1, "the paste is visibly in the composer; re-pasting duplicates it"
    assert len(tmux.enters) == 2


def test_first_prompt_consumed_whole_is_repasted(
        fast_submit, tmux, sessions_dir, monkeypatch):
    # Composer empty after the submit: the paste itself was consumed, so the
    # prompt goes back in — and it lands.
    monkeypatch.setattr(codex_server, "SESSION_DETECT_TIMEOUT", 0.6)
    monkeypatch.setattr(codex_server, "SUBMIT_REPASTE_DELAY", 0.0)
    sess = _first_session(monkeypatch, screen=EMPTY_COMPOSER)
    tmux.on_enter = lambda n: (
        _write_rollout(sessions_dir, "ours", cwd=str(sess.cwd)) if n == 2 else None
    )
    got = _run(sess._submit_first(PROMPT, before=set()))
    assert codex_server._rollout_meta(got)["id"] == "ours"
    assert len(tmux.pastes) == 2


def test_first_prompt_never_lands_raises_after_bounded_retries(
        fast_submit, tmux, sessions_dir, monkeypatch):
    monkeypatch.setattr(codex_server, "SESSION_DETECT_TIMEOUT", 0.3)
    monkeypatch.setattr(codex_server, "SUBMIT_REPASTE_DELAY", 0.0)
    sess = _first_session(monkeypatch, screen=EMPTY_COMPOSER)
    with pytest.raises(RuntimeError, match="codex session"):
        _run(sess._submit_first(PROMPT, before=set()))
    assert len(tmux.pastes) == 1 + codex_server.SUBMIT_REPASTE_MAX
    assert len(tmux.enters) == len(tmux.pastes) + codex_server.SUBMIT_MAX_RETRIES


def test_busy_screen_is_never_nudged_on_the_first_prompt(
        fast_submit, tmux, sessions_dir, monkeypatch):
    """codex is visibly working, so the rollout is just slow to appear — nudging
    would duplicate the turn."""
    monkeypatch.setattr(codex_server, "SESSION_DETECT_TIMEOUT", 0.3)
    sess = _first_session(monkeypatch, screen="• Working (12s • esc to interrupt)")
    with pytest.raises(RuntimeError, match="codex session"):
        _run(sess._submit_first(PROMPT, before=set()))
    assert len(tmux.pastes) == 1 and len(tmux.enters) == 1


def test_startup_grace_holds_the_first_paste_of_a_fresh_session(
        fast_submit, tmux, sessions_dir, monkeypatch):
    # codex's ready marker precedes real input readiness, so a session that just
    # became ready waits out the grace before pasting; one ready long ago does not.
    monkeypatch.setattr(codex_server, "SUBMIT_GRACE", 0.4)
    monkeypatch.setattr(codex_server, "SESSION_DETECT_TIMEOUT", 2.0)
    sess = _first_session(monkeypatch, screen=UNSUBMITTED)
    _write_rollout(sessions_dir, "ours", cwd=str(sess.cwd))

    sess._ready_at = time.monotonic()
    t0 = time.monotonic()
    _run(sess._submit_first(PROMPT, before=set()))
    assert time.monotonic() - t0 >= 0.4 - 0.05

    sess._ready_at = time.monotonic() - 60  # warm session: grace already paid
    t0 = time.monotonic()
    _run(sess._submit_first(PROMPT, before=set()))
    assert time.monotonic() - t0 < 0.3


def test_a_retried_first_turn_reuses_the_original_scan_set(sessions_dir, monkeypatch):
    """A first turn that failed to bind may still have CREATED the rollout; a
    freshly taken `before` on the retry would exclude that file forever, so the
    session keeps the scan set its first submit captured."""
    sess = codex_server.CodexSession(name="unit")
    rollout = _write_rollout(sessions_dir, "ours", cwd=str(sess.cwd))
    sess._rollout_before = set()  # what the failed attempt captured
    seen = []

    async def submit_first(prompt, before):
        seen.append(set(before))
        return Path(rollout)

    async def collect(baseline_completes, baseline_starts):
        return "ok"

    async def alive():
        return True

    monkeypatch.setattr(sess, "_submit_first", submit_first)
    monkeypatch.setattr(sess, "_collect_response", collect)
    monkeypatch.setattr(sess, "is_alive", alive)
    monkeypatch.setattr(codex_server, "_SPAWN_LOCK", asyncio.Lock())  # fresh per loop
    assert _run(sess.send("q")) == "ok"
    assert seen == [set()], "a fresh scan would exclude the rollout already on disk"
    assert sess._session_id == "ours"


def test_startup_grace_zero_is_off(monkeypatch):
    monkeypatch.setattr(codex_server, "SUBMIT_GRACE", 0.0)
    sess = codex_server.CodexSession(name="unit")
    sess._ready_at = time.monotonic()
    t0 = time.monotonic()
    _run(sess._startup_grace())
    assert time.monotonic() - t0 < 0.1
