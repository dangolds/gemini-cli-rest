"""Hermetic smoke: fake CLI + fake git + fake clock drive a real first turn on
each server, over HTTP, with no real time spent."""
from __future__ import annotations

import time

import pytest

from bridgetests import names

pytestmark = [pytest.mark.hermetic, pytest.mark.group("D")]


def test_first_turn_cuts_worktree_and_answers_after_fake_delay(fake_bridge):
    fb = fake_bridge
    key = names.key("smoke")
    # given: origin/main points at a known commit, and the next turn takes 7 fake seconds
    sha = fb.git.advance("origin/main")
    fb.cli.answer("pong", after=7.0)
    real0, fake0 = time.perf_counter(), fb.clock.now

    # when: the first turn on the key is sent over HTTP
    with fb.client() as client:
        r = client.post(f"/chat/{key}", json={"prompt": "ping"})
        health = client.get("/health").json()
        on_disk = fb.git.commit_of(fb.worktree_dir(key, generation=1))

    # then: the answer is the scripted one, turn 1, completed through the push signal
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["session"] == key
    assert body["turn"] == 1
    assert body["response"] == "pong"
    assert body["via"] == fb.expected_via

    # the CLI received exactly that prompt, in this key's tmux session
    [turn] = fb.cli.turns
    assert turn.prompt == "ping"
    assert turn.tmux_session == fb.tmux_session_name(key)

    # the worktree was cut from the fake ref at the server's own path, and holds the commit
    [(cwd, ref, commit)] = fb.git.added
    assert ref == "origin/main"
    assert commit == sha
    assert cwd == fb.worktree_dir(key, generation=1)
    assert on_disk == sha
    assert fb.git.fetches == 1
    # leaving the client ran the lifespan shutdown: the worktree is gone again
    assert fb.git.removed == [cwd] and not cwd.exists()

    # the session is listed alive with the count advanced
    assert health["sessions"] == [{"name": key, "alive": True, "turn_count": 1}]

    # and no real time was spent: the fake clock moved by at least the delay, the wall barely
    assert fb.clock.now - fake0 >= 7.0
    assert time.perf_counter() - real0 < 5.0


@pytest.mark.group("A")
def test_dotted_base_gets_a_tmux_session_and_answers(fake_bridge):
    """features.md #16: tmux stores '.' in a session name as '_' and cannot
    resolve the dotted target (the FakeCLI mirrors that), so the servers derive
    the tmux name with worktree.tmux_safe_name()."""
    fb = fake_bridge
    key = names.key("dotted", "release/1.2")
    # given: origin/release/1.2 exists, and the next turn takes 3 fake seconds
    fb.git.set_ref("origin/release/1.2", fb.git.advance("origin/main"))
    fb.cli.answer("pong", after=3.0)

    # when: the first turn on the dotted key is sent over HTTP
    with fb.client() as client:
        r = client.post(f"/chat/{key}", json={"prompt": "ping"})

    # then: it answers as turn 1, in a tmux session whose name carries no dot
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["session"] == key
    assert body["turn"] == 1
    assert body["response"] == "pong"
    [turn] = fb.cli.turns
    assert turn.prompt == "ping"
    assert "." not in turn.tmux_session
    assert turn.tmux_session == fb.tmux_session_name(key)
