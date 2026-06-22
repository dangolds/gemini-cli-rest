"""
Unit tests for the codex bridge's local diagnostic dumps.

A non-clean turn end (stall / hard timeout) or a slow-but-successful turn writes
a self-contained snapshot to LOG_DIR/timeouts/ so it can be investigated without
docker-exec'ing a live container. These pin that the dump records the reason,
the resolved session id/rollout, the rollout event tail, and the rendered screen.

No server / codex / Docker — the rollout and the screen are mocked.
Run:  ./.venv/bin/python -m pytest test_codex_diagnostics.py -v
"""

import asyncio
import json

import codex_server


def _run(coro):
    return asyncio.run(coro)


def _evt(ptype: str, **payload) -> str:
    """An event_msg JSONL line with the given payload type."""
    return json.dumps({"type": "event_msg", "payload": {"type": ptype, **payload},
                       "timestamp": "t"})


def test_diagnostic_records_reason_session_and_rollout_tail(tmp_path, monkeypatch):
    monkeypatch.setattr(codex_server, "TIMEOUT_LOG_DIR", tmp_path / "timeouts")
    rollout = tmp_path / "rollout-2026-01-01T00-00-00-abc.jsonl"
    rollout.write_text("\n".join([
        json.dumps({"type": "session_meta", "payload": {"id": "sess-1", "cwd": "/x"}}),
        _evt("task_started", turn_id="t1"),
        _evt("task_complete", turn_id="t1", last_agent_message="done"),
    ]) + "\n")

    sess = codex_server.CodexSession(name="cdxdiag")
    sess._rollout_path = rollout
    sess._session_id = "sess-1"
    sess._turn_count = 3

    async def fake_capture():
        return "RENDERED SCREEN HERE\n  gpt-5.5 · Context 0% used"

    monkeypatch.setattr(sess, "_capture", fake_capture)

    _run(sess._dump_diagnostic("stalled", baseline=0, elapsed=123.4, response=""))

    dumps = list((tmp_path / "timeouts").glob("*.log"))
    assert len(dumps) == 1
    body = dumps[0].read_text()
    assert "reason=stalled" in body
    assert "session_id=sess-1" in body
    assert "elapsed=123.4s" in body
    # rollout tail must render the turn boundary events
    assert "event_msg/task_started" in body
    assert "event_msg/task_complete" in body
    assert "turn=t1" in body
    # the screen at give-up time is captured for context
    assert "RENDERED SCREEN HERE" in body


def test_diagnostic_survives_unresolved_rollout(tmp_path, monkeypatch):
    """A first turn that timed out before the rollout was resolved still dumps."""
    monkeypatch.setattr(codex_server, "TIMEOUT_LOG_DIR", tmp_path / "timeouts")
    sess = codex_server.CodexSession(name="cdxearly")
    sess._rollout_path = None  # never resolved
    sess._turn_count = 1

    async def fake_capture():
        return "stuck on something"

    monkeypatch.setattr(sess, "_capture", fake_capture)

    _run(sess._dump_diagnostic("hard_timeout", baseline=0, elapsed=180.0, response=""))

    dumps = list((tmp_path / "timeouts").glob("*.log"))
    assert len(dumps) == 1
    body = dumps[0].read_text()
    assert "reason=hard_timeout" in body
    assert "rollout=None" in body
    assert "stuck on something" in body
