"""
Unit tests for the /last read-back path on BOTH bridges (codex + agy).

/last re-reads the most recent turn's answer so a client can recover a reply its
/chat call never received (the 3-min hard cap, or a connectivity blip after the
CLI had already finished) WITHOUT re-asking. The contract is binary: return the
answer only once the turn is DONE; never a partial, and never a *previous*
turn's answer in place of one that has not landed yet.

The two bridges decide "done" differently, and these tests pin both:
  * codex — from the rollout alone: a new task_complete beyond the submit
    baseline. No screen needed.
  * agy   — from the transcript (a new completed answer past the baseline step)
    AND the rendered screen being idle, because agy writes every step as DONE
    and so carries no in-flight marker in the file. A dead process counts as
    "not working", so its last answer is still returned.

No server / codex / agy / Docker — rollout, transcript, and screen are mocked.
Run:  ./.venv/bin/python -m pytest test_last.py -v
"""

import asyncio
import time
from pathlib import Path

import pytest

import codex_server
import server


def _run(coro):
    return asyncio.run(coro)


# ===========================================================================
# codex: done is read from the rollout (task_complete count) alone
# ===========================================================================

def _ev_meta():
    return {"type": "session_meta", "payload": {"id": "s1", "cwd": "/x"}}


def _ev_start(turn_id="t1"):
    return {"type": "event_msg", "payload": {"type": "task_started", "turn_id": turn_id}}


def _ev_complete(turn_id="t1", last="the answer"):
    return {"type": "event_msg",
            "payload": {"type": "task_complete", "turn_id": turn_id, "last_agent_message": last}}


def _ev_assistant(text):
    return {"type": "response_item",
            "payload": {"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": text}]}}


@pytest.fixture()
def codex_fastpoll(monkeypatch):
    monkeypatch.setattr(codex_server, "RESPONSE_POLL_INTERVAL", 0.02)


def _codex_session(monkeypatch, *, events, baseline=0, turn=1, rollout=Path("/tmp/x/rollout.jsonl")):
    sess = codex_server.CodexSession(name="unit")
    sess._rollout_path = rollout
    sess._last_baseline_completes = baseline
    sess._turn_count = turn
    monkeypatch.setattr(codex_server, "_read_rollout", lambda path: events())
    return sess


class TestCodexLast:
    def test_done_returns_answer_past_baseline(self, codex_fastpoll, monkeypatch):
        sess = _codex_session(
            monkeypatch,
            events=lambda: [_ev_meta(), _ev_start("t1"), _ev_complete("t1", last="answer one")],
            baseline=0, turn=1,
        )
        assert _run(sess.last(0)) == (True, "answer one", 1)

    def test_in_flight_turn_is_not_done(self, codex_fastpoll, monkeypatch):
        # started but not completed -> still running -> done=False, no text
        sess = _codex_session(monkeypatch, events=lambda: [_ev_meta(), _ev_start("t1")], baseline=0)
        assert _run(sess.last(0)) == (False, "", 1)

    def test_never_returns_a_previous_turn_as_the_new_one(self, codex_fastpoll, monkeypatch):
        # One completed turn exists ("answer one"); baseline=1 means the caller's
        # NEW turn (turn 2) has not completed. /last must refuse, NOT hand back
        # turn 1 dressed up as turn 2. This is the stale-answer guard.
        sess = _codex_session(
            monkeypatch,
            events=lambda: [_ev_meta(), _ev_start("t1"), _ev_complete("t1", last="answer one")],
            baseline=1, turn=2,
        )
        assert _run(sess.last(0)) == (False, "", 2)

    def test_waits_then_returns_late_completion(self, codex_fastpoll, monkeypatch):
        t0 = time.monotonic()

        def events():
            if time.monotonic() - t0 > 0.3:
                return [_ev_meta(), _ev_start("t1"), _ev_complete("t1", last="late answer")]
            return [_ev_meta(), _ev_start("t1")]  # in-flight until 0.3s

        sess = _codex_session(monkeypatch, events=events, baseline=0)
        done, ans, _ = _run(sess.last(5))
        assert done and ans == "late answer"

    def test_no_rollout_bound_yet_is_not_done(self, codex_fastpoll, monkeypatch):
        sess = codex_server.CodexSession(name="unit")
        sess._rollout_path = None  # never sent a turn
        assert _run(sess.last(0)) == (False, "", 0)

    def test_falls_back_to_assistant_text_when_last_message_null(self, codex_fastpoll, monkeypatch):
        # task_complete with last_agent_message=null still has a visible reply in
        # the assistant output_text — /last must surface it, not an empty string.
        sess = _codex_session(
            monkeypatch,
            events=lambda: [_ev_meta(), _ev_start("t1"),
                            _ev_assistant("visible reply"), _ev_complete("t1", last=None)],
            baseline=0,
        )
        assert _run(sess.last(0)) == (True, "visible reply", 1)

    def test_returns_most_recent_of_several_completed_turns(self, codex_fastpoll, monkeypatch):
        # Three turns completed; the caller's turn was the 3rd (baseline=2).
        # /last returns turn 3's answer, never an earlier turn's.
        events = [
            _ev_meta(),
            _ev_start("t1"), _ev_complete("t1", last="answer one"),
            _ev_start("t2"), _ev_complete("t2", last="answer two"),
            _ev_start("t3"), _ev_complete("t3", last="answer three"),
        ]
        sess = _codex_session(monkeypatch, events=lambda: events, baseline=2, turn=3)
        done, ans, turn = _run(sess.last(0))
        assert (done, ans, turn) == (True, "answer three", 3)
        assert "answer one" not in ans and "answer two" not in ans

    def test_done_with_empty_answer_when_turn_produced_no_text(self, codex_fastpoll, monkeypatch):
        # A completed turn with neither last_agent_message nor assistant text is
        # legitimately DONE (just empty) — /last reports done, not "still running".
        sess = _codex_session(
            monkeypatch,
            events=lambda: [_ev_meta(), _ev_start("t1"), _ev_complete("t1", last=None)],
            baseline=0,
        )
        assert _run(sess.last(0)) == (True, "", 1)

    def test_wait_elapses_then_reports_not_done(self, codex_fastpoll, monkeypatch):
        # Perpetually in-flight: /last must honor `wait` then give up (done=False),
        # not block forever.
        sess = _codex_session(monkeypatch, events=lambda: [_ev_meta(), _ev_start("t1")], baseline=0)
        t0 = time.monotonic()
        result = _run(sess.last(0.3))
        assert result == (False, "", 1)
        assert time.monotonic() - t0 >= 0.3 - 0.05  # roughly honored the wait


# ===========================================================================
# agy: done needs a new answer past baseline AND an idle (or gone) screen
# ===========================================================================

def _step(idx, content="the answer"):
    return {"step_index": idx, "source": "MODEL", "type": "PLANNER_RESPONSE",
            "status": "DONE", "content": content, "created_at": "t"}


@pytest.fixture()
def agy_fastpoll(monkeypatch):
    monkeypatch.setattr(server, "RESPONSE_POLL_INTERVAL", 0.02)


def _agy_session(monkeypatch, *, transcript, screen="done  ? for shortcuts",
                 alive=True, baseline=-1, turn=1):
    sess = server.AgySession(name="unit")
    sess._conversation_id = "conv"
    sess._last_baseline_step = baseline
    sess._turn_count = turn
    monkeypatch.setattr(server, "_read_transcript", lambda cid: transcript())

    async def capture():
        return screen() if callable(screen) else screen

    async def is_alive():
        return alive() if callable(alive) else alive

    monkeypatch.setattr(sess, "_capture", capture)
    monkeypatch.setattr(sess, "is_alive", is_alive)
    return sess


class TestAgyLast:
    def test_done_when_answer_past_baseline_and_idle(self, agy_fastpoll, monkeypatch):
        sess = _agy_session(
            monkeypatch,
            transcript=lambda: [_step(0, "final review")],
            screen="all done  ? for shortcuts",
            baseline=-1,
        )
        assert _run(sess.last(0)) == (True, "final review", 1)

    def test_busy_screen_is_not_done(self, agy_fastpoll, monkeypatch):
        # An answer is in the transcript but agy is still generating: agy emits
        # intermediate planner responses mid-turn, so a busy screen means the
        # turn is NOT over. Must not return the partial.
        sess = _agy_session(
            monkeypatch,
            transcript=lambda: [_step(0, "partial")],
            screen="Generating...  esc to cancel",
        )
        assert _run(sess.last(0)) == (False, "", 1)

    def test_previous_turn_answer_is_not_returned(self, agy_fastpoll, monkeypatch):
        # The only answer sits at step 3, but the baseline is 5 -> it belongs to
        # a PREVIOUS turn. Even with an idle screen, /last must refuse.
        sess = _agy_session(
            monkeypatch,
            transcript=lambda: [_step(3, "old answer")],
            screen="idle  ? for shortcuts",
            baseline=5,
        )
        assert _run(sess.last(0)) == (False, "", 1)

    def test_done_when_process_gone(self, agy_fastpoll, monkeypatch):
        # A DONE planner response is itself complete; a dead process means
        # nothing more is coming, so its last answer is returned.
        sess = _agy_session(
            monkeypatch,
            transcript=lambda: [_step(0, "final")],
            alive=False,
        )
        assert _run(sess.last(0)) == (True, "final", 1)

    def test_joins_multiple_messages_of_the_turn(self, agy_fastpoll, monkeypatch):
        sess = _agy_session(
            monkeypatch,
            transcript=lambda: [_step(0, "part one"), _step(2, "part two")],
            screen="idle  ? for shortcuts",
        )
        assert _run(sess.last(0)) == (True, "part one\n\npart two", 1)

    def test_waits_until_idle_then_returns(self, agy_fastpoll, monkeypatch):
        t0 = time.monotonic()
        sess = _agy_session(
            monkeypatch,
            transcript=lambda: [_step(0, "the answer")],
            screen=lambda: ("? for shortcuts" if time.monotonic() - t0 > 0.3
                            else "Generating...  esc to cancel"),
        )
        done, ans, _ = _run(sess.last(5))
        assert done and ans == "the answer"

    def test_no_conversation_bound_yet_is_not_done(self, agy_fastpoll, monkeypatch):
        sess = server.AgySession(name="unit")
        sess._conversation_id = None  # never sent a turn
        assert _run(sess.last(0)) == (False, "", 0)

    def test_capture_error_mid_check_counts_as_done(self, agy_fastpoll, monkeypatch):
        # The pane vanishes right as we glance at it (process died between the
        # is_alive check and the capture). A DONE answer is already complete, so
        # _turn_over_evidence treats the dead pane as "not working" and returns it.
        def boom():
            raise RuntimeError("pane gone")

        sess = _agy_session(
            monkeypatch,
            transcript=lambda: [_step(0, "final")],
            screen=boom, alive=True,
        )
        assert _run(sess.last(0)) == (True, "final", 1)

    def test_contentless_steps_do_not_count_as_an_answer(self, agy_fastpoll, monkeypatch):
        # Mid-turn agy writes planner-response steps with no content; these are
        # NOT answers. With only contentless steps past the baseline, /last must
        # report not-done even on an idle screen.
        sess = _agy_session(
            monkeypatch,
            transcript=lambda: [_step(0, None), _step(1, "")],
            screen="idle  ? for shortcuts",
        )
        assert _run(sess.last(0)) == (False, "", 1)

    def test_busy_via_esc_to_cancel_marker_is_not_done(self, agy_fastpoll, monkeypatch):
        sess = _agy_session(
            monkeypatch,
            transcript=lambda: [_step(0, "partial")],
            screen="some output  esc to cancel",  # busy marker, no ready marker
        )
        assert _run(sess.last(0)) == (False, "", 1)

    def test_wait_elapses_while_busy_reports_not_done(self, agy_fastpoll, monkeypatch):
        # An answer exists but the screen stays busy forever: /last honors `wait`
        # then gives up rather than returning the in-flight partial.
        sess = _agy_session(
            monkeypatch,
            transcript=lambda: [_step(0, "partial")],
            screen="Generating...  esc to cancel",
        )
        t0 = time.monotonic()
        assert _run(sess.last(0.3)) == (False, "", 1)
        assert time.monotonic() - t0 >= 0.3 - 0.05


# ===========================================================================
# /last observability (audit-log improvements): model-timing derivation,
# per-turn attempt count, and the dump completion footer. Pure helpers — no
# server / CLI / Docker. The exact log-line WORDING is intentionally not
# asserted; these pin the data the lines are built from.
# ===========================================================================

class TestParseIsoTs:
    """Both bridges turn the transcript/rollout's own timestamps into the
    durations /last logs, so the parse must accept 'Z' and fractional seconds."""

    def test_z_equals_explicit_offset_and_orders_correctly(self):
        for mod in (server, codex_server):
            assert mod._parse_iso_ts("2026-06-23T02:38:57Z") == \
                mod._parse_iso_ts("2026-06-23T02:38:57+00:00")
            a = mod._parse_iso_ts("2026-06-23T02:34:23.965Z")
            b = mod._parse_iso_ts("2026-06-23T02:34:24.965Z")
            assert b - a == pytest.approx(1.0)

    def test_missing_or_junk_is_none(self):
        for mod in (server, codex_server):
            assert mod._parse_iso_ts(None) is None
            assert mod._parse_iso_ts("not-a-timestamp") is None


class TestAgyResponseBounds:
    def test_start_is_earliest_complete_is_last_done_answer(self):
        steps = [
            {"step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT",
             "status": "DONE", "content": "q", "created_at": "2026-06-23T02:00:00Z"},
            {"step_index": 1, "source": "MODEL", "type": "PLANNER_RESPONSE",
             "status": "DONE", "content": "interim", "created_at": "2026-06-23T02:01:00Z"},
            {"step_index": 2, "source": "MODEL", "type": "PLANNER_RESPONSE",
             "status": "DONE", "content": "final", "created_at": "2026-06-23T02:03:00Z"},
        ]
        assert server._new_response_bounds(steps, baseline=-1) == \
            ("2026-06-23T02:00:00Z", "2026-06-23T02:03:00Z")

    def test_baseline_excludes_a_prior_turn(self):
        steps = [
            {"step_index": 5, "source": "MODEL", "type": "PLANNER_RESPONSE",
             "status": "DONE", "content": "prev", "created_at": "2026-06-23T01:00:00Z"},
            {"step_index": 9, "source": "MODEL", "type": "PLANNER_RESPONSE",
             "status": "DONE", "content": "this", "created_at": "2026-06-23T02:00:00Z"},
        ]
        assert server._new_response_bounds(steps, baseline=5) == \
            ("2026-06-23T02:00:00Z", "2026-06-23T02:00:00Z")

    def test_no_new_steps_returns_none(self):
        assert server._new_response_bounds([], baseline=-1) == (None, None)


class TestCodexTurnBounds:
    def test_start_and_complete_for_first_turn(self):
        events = [
            {"type": "event_msg", "timestamp": "2026-06-23T02:00:00Z",
             "payload": {"type": "task_started", "turn_id": "t1"}},
            {"type": "event_msg", "timestamp": "2026-06-23T02:05:00Z",
             "payload": {"type": "task_complete", "turn_id": "t1"}},
        ]
        assert codex_server._turn_bounds(events, baseline_completes=0) == \
            ("2026-06-23T02:00:00Z", "2026-06-23T02:05:00Z")

    def test_baseline_skips_an_already_completed_turn(self):
        events = [
            {"type": "event_msg", "timestamp": "2026-06-23T01:00:00Z",
             "payload": {"type": "task_started", "turn_id": "t1"}},
            {"type": "event_msg", "timestamp": "2026-06-23T01:01:00Z",
             "payload": {"type": "task_complete", "turn_id": "t1"}},
            {"type": "event_msg", "timestamp": "2026-06-23T02:00:00Z",
             "payload": {"type": "task_started", "turn_id": "t2"}},
            {"type": "event_msg", "timestamp": "2026-06-23T02:09:00Z",
             "payload": {"type": "task_complete", "turn_id": "t2"}},
        ]
        assert codex_server._turn_bounds(events, baseline_completes=1) == \
            ("2026-06-23T02:00:00Z", "2026-06-23T02:09:00Z")


_SESSIONS = ((server, server.AgySession), (codex_server, codex_server.CodexSession))


class TestNextAttempt:
    def test_increments_and_send_resets_it(self):
        for _mod, cls in _SESSIONS:
            s = cls(name="unit")
            assert s.next_attempt() == 1
            assert s.next_attempt() == 2
            s._last_attempt = 0  # what send() does at each new turn boundary
            assert s.next_attempt() == 1


class TestRecordRecovery:
    def test_appends_footer_once_for_the_matching_turn(self, tmp_path):
        for _mod, cls in _SESSIONS:
            dump = tmp_path / f"{cls.__name__}.log"
            dump.write_text("=== give-up snapshot ===\n")
            s = cls(name="unit")
            s._turn_count = 3
            s._last_dump_path = str(dump)
            s._last_dump_turn = 3
            s._last_dump_finished = False
            s.record_recovery("2026-06-23T02:05:00Z", 300.0)
            text = dump.read_text()
            assert "COMPLETED after give-up" in text
            assert "answer_completed_at=2026-06-23T02:05:00Z" in text
            assert "model_turn_secs=300.0" in text
            s.record_recovery("2026-06-23T02:05:00Z", 300.0)  # idempotent
            assert dump.read_text().count("COMPLETED after give-up") == 1

    def test_leaves_a_dump_from_a_different_turn_untouched(self, tmp_path):
        for _mod, cls in _SESSIONS:
            dump = tmp_path / f"{cls.__name__}-mismatch.log"
            dump.write_text("snapshot\n")
            s = cls(name="unit")
            s._turn_count = 5            # current turn
            s._last_dump_path = str(dump)
            s._last_dump_turn = 4        # dump belongs to a previous turn
            s.record_recovery("2026-06-23T02:05:00Z", 1.0)
            assert dump.read_text() == "snapshot\n"
