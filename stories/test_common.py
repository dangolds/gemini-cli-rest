"""Common live stories: every story here runs on both ports from one code path.

Stage 1 adds the groups of TestPRD section 4 to this file (mark each class
with its group). Stage 0 ships one placeholder story for group A.
"""
from __future__ import annotations

import time

import pytest


@pytest.mark.group("A")
class TestFirstTurn:
    @pytest.mark.units(1)
    def test_first_turn_answers(self, bridge, own_session):
        # given: a fresh run-stamped key on an existing base
        key = own_session("first-turn")
        # when: the first turn is sent
        rep = bridge.chat(key, "Reply with exactly the word: pong")
        # then: it answers as turn 1 under that key, and health lists the session alive
        assert rep.status == 200, f"{rep.status} {rep.error or rep.text[:200]}"
        assert rep["session"] == key
        assert rep["turn"] == 1
        assert rep["response"].strip()
        listed = bridge.sessions().get(key)
        assert listed and listed["alive"] is True and listed["turn_count"] == 1


@pytest.mark.group("C")
class TestTeardownPaths:
    """The teardown machinery of bridgetests.live, proven on the real container.

    The 200 s teardown deadline itself cannot fire on a healthy bridge (a delete
    waits at most the 180 s turn cap), so that branch stays hermetic; what CAN
    be proven live is every piece it is built from: pane listing, the verified
    pane kill, and the "uncertain request" bookkeeping after a client timeout.
    """

    @pytest.mark.units(1)
    def test_verified_pane_kill_then_delete(self, bridge, own_session):
        # given: a live session with one answered turn
        key = own_session("kill-path")
        rep = bridge.chat(key, "Reply with exactly the word: pong")
        assert rep.status == 200, f"{rep.status} {rep.error or rep.text[:200]}"
        assert bridge.pane_pids(key), "the session's pane must be listed while it is alive"
        # when: its pane is killed the way the deadline branch does it
        pid = bridge.kill_verified_pane(key)
        # then: a pid came back only once the pane was verified gone, the bridge
        # still lists the (now dead) session, and a normal teardown confirms it
        assert isinstance(pid, int) and pid > 0
        assert bridge.pane_pids(key) == []
        assert key in bridge.sessions()
        outcome = bridge.teardown_session(key)
        assert outcome in bridge.CONFIRMED, outcome
        assert key not in bridge.sessions()
        assert bridge.worktree_dirs(key) == []

    @pytest.mark.units(1)
    def test_client_timeout_is_uncertain_until_rechecked(self, bridge, own_session):
        # given: a first turn whose client gives up after half a second
        key = own_session("uncertain")
        rep = bridge.chat(key, "Reply with exactly the word: pong", timeout=0.5)
        assert rep.status is None and rep.error, rep
        # when: the bridge finishes the turn anyway (visible through /last)
        deadline = time.monotonic() + 240
        while True:
            last = bridge.last(key, wait=30)
            if last.status == 200 and last.get("done") is True:
                break
            assert time.monotonic() < deadline, f"turn never completed: {last.status} {last.error}"
        assert last["turn"] == 1
        # then: the teardown re-checks and reports the once-uncertain key with a
        # "?", nothing of it is left, and a second teardown is a plain "absent"
        outcome = bridge.teardown_session(key)
        assert outcome == "deleted?", outcome
        assert key not in bridge.sessions() and bridge.pane_pids(key) == []
        assert bridge.teardown_session(key) == "absent"
