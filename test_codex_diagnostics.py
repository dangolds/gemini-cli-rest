"""
Unit tests for the codex bridge's local diagnostic dumps.

The codex 502s we hit were invisible because the failure reason only ever went
into the HTTP body, never a log. These pin the new dumps: a failure dump (argv
+ reason + stderr) and a slow-success breakdown (a tally of what the turn did,
from the --json stream) so a slow-but-fine turn is explainable.

No server / codex / Docker.
Run:  ./.venv/bin/python -m pytest test_codex_diagnostics.py -v
"""

import codex_server


def test_failure_dump_records_reason_and_stderr(tmp_path, monkeypatch):
    monkeypatch.setattr(codex_server, "LOG_DIR", tmp_path)
    sess = codex_server.CodexSession(name="cdxfail")
    sess._dump_failure(
        2, ["codex", "exec", "-"], 12.3,
        reason="rc=1", rc=1, stdout="some out", stderr="boom: it broke",
    )
    body = (tmp_path / "timeouts" / "codex-cdxfail-turn2.log").read_text()
    assert "reason=rc=1" in body
    assert "boom: it broke" in body
    assert "argv=codex exec -" in body


def test_slow_dump_tallies_what_the_turn_did(tmp_path, monkeypatch):
    monkeypatch.setattr(codex_server, "LOG_DIR", tmp_path)
    sess = codex_server.CodexSession(name="cdxslow")
    sess._session_id = "thread-1"
    stdout = "\n".join([
        '{"type":"thread.started","thread_id":"thread-1"}',
        '{"type":"item.completed","item":{"type":"command_execution"}}',
        '{"type":"item.completed","item":{"type":"command_execution"}}',
        '{"type":"item.completed","item":{"type":"file_change"}}',
        '{"type":"item.completed","item":{"type":"agent_message","text":"done"}}',
        '{"type":"turn.completed"}',
    ])
    sess._dump_slow(turn=3, elapsed=265.8, stdout=stdout)
    body = (tmp_path / "timeouts" / "codex-slow-cdxslow-turn3.log").read_text()
    assert "reason=slow_success" in body
    assert "elapsed=265.8s" in body and "events=6" in body
    # the tally must show the 2 command executions (the "where did the time go")
    assert "item:command_execution" in body
    assert "   2  item:command_execution" in body
