"""
Unit tests for the codex bridge's notify fast path (push-based completion).

codex's `notify` hook runs a program with the turn's summary JSON as argv[1]
the moment a turn completes; the bridge's hook appends that payload to
NOTIFY_LOG, and _collect_response uses the file as a "check the rollout NOW"
signal instead of waiting out a full poll interval. The properties correctness
rides on:

  * the push NEVER completes a turn by itself — a new task_complete must be
    durable in the rollout, which stays the single source of truth for the
    answer text (the payload's last-assistant-message is never used)
  * events are attributed per session (cwd / thread-id) and per turn (baseline
    captured at submit), so a neighbor's or a previous turn's event never fires
  * a push credits stall progress on its EDGE only — one stray event can never
    disable the stall timeout — and the count is incremental: the shared log
    grows for the whole server run, so a glance reads only newly appended bytes
  * a hook path unsafe for the TOML notify override is never wired in: both the
    install and the -c flag decline, degrading to pure polling
  * with CODEX_NOTIFY off — or the signal never arriving — behavior is exactly
    the legacy polling path (rollout_done, same cadence, same timeouts)

No server / codex / Docker — the rollout, the screen, and the notify log are
mocked or tmp files. Run:  ./.venv/bin/python -m pytest test_notify.py -v
"""

import asyncio
import json
import os
import shlex
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


def _notify_event(*, cwd="/x", thread_id="thread-1", last="the answer"):
    """One NOTIFY_LOG line, shaped exactly like codex's notify argv[1]
    (flat object, kebab-case keys — verified against codex 0.141)."""
    return {"type": "agent-turn-complete", "thread-id": thread_id, "turn-id": "t1",
            "cwd": cwd, "client": "codex-tui", "input-messages": ["q"],
            "last-assistant-message": last}


def _write_events(path: Path, *objs, raw=()):
    lines = [json.dumps(o) for o in objs] + list(raw)
    path.write_text("\n".join(lines) + "\n")


# --- fixtures / session helper (mirrors test_codex_rollout) -----------------

@pytest.fixture()
def notify_log(tmp_path, monkeypatch):
    """Fast-path environment: feature on, NOTIFY_LOG in a tmp dir, quick wakes,
    and TIMEOUT_LOG_DIR redirected so a stray dump can't touch /app/logs."""
    log = tmp_path / "events.jsonl"
    monkeypatch.setattr(codex_server, "CODEX_NOTIFY", True)
    monkeypatch.setattr(codex_server, "NOTIFY_LOG", log)
    monkeypatch.setattr(codex_server, "RESPONSE_FAST_POLL", 0.02)
    monkeypatch.setattr(codex_server, "RESPONSE_MIN_WAIT", 0.0)
    monkeypatch.setattr(codex_server, "TIMEOUT_LOG_DIR", tmp_path / "timeouts")
    return log


def _collect_session(monkeypatch, *, events, screen="idle  Context 0% used", mtime=0.0,
                     notify_baseline=0):
    """A session whose rollout/screen/mtime are driven by callables."""
    sess = codex_server.CodexSession(name="unit")
    sess._rollout_path = Path("/tmp/does-not-matter/rollout-x.jsonl")
    sess._session_id = "thread-1"
    sess._turn_count = 1
    sess._last_notify_baseline = notify_baseline
    monkeypatch.setattr(codex_server, "_read_rollout", lambda path: events())
    monkeypatch.setattr(sess, "_rollout_mtime", (mtime if callable(mtime) else (lambda: mtime)))

    async def capture():
        return screen() if callable(screen) else screen

    monkeypatch.setattr(sess, "_capture", capture)
    return sess


# ===========================================================================
# Fast path: a matching push completes the turn quickly — via the rollout
# ===========================================================================

class TestFastPath:
    def test_matching_notify_completes_via_notify_fast(self, notify_log, tmp_path, monkeypatch):
        # polling cadence is slow (1s): only the fast path can finish this fast
        monkeypatch.setattr(codex_server, "RESPONSE_POLL_INTERVAL", 1.0)
        monkeypatch.setattr(codex_server, "RESPONSE_STALL_TIMEOUT", 5.0)
        monkeypatch.setattr(codex_server, "RESPONSE_HARD_TIMEOUT", 10.0)
        sess = _collect_session(
            monkeypatch,
            events=lambda: [_meta(), _start(), _complete(last="fast answer")],
        )
        # garbage alongside the real event: the glance must shrug it off
        _write_events(notify_log, _notify_event(cwd=str(sess.cwd)), raw=["not json"])
        t0 = time.monotonic()
        assert _run(sess._collect_response(baseline_completes=0, baseline_starts=0)) \
            == "fast answer"
        assert time.monotonic() - t0 < 0.5, "fast path must beat the polling cadence"
        assert sess._last_exit_reason == "notify"
        # a notify completion is CLEAN — it must never leave a diagnostic dump
        assert not list((tmp_path / "timeouts").glob("*.log"))

    def test_foreign_notify_is_ignored_legacy_completes(self, notify_log, monkeypatch):
        monkeypatch.setattr(codex_server, "RESPONSE_POLL_INTERVAL", 0.25)
        monkeypatch.setattr(codex_server, "RESPONSE_STALL_TIMEOUT", 5.0)
        monkeypatch.setattr(codex_server, "RESPONSE_HARD_TIMEOUT", 10.0)
        sess = _collect_session(
            monkeypatch,
            events=lambda: [_meta(), _start(), _complete(last="poll answer")],
        )
        # a NEIGHBOR session's turn: different cwd AND different thread-id
        _write_events(notify_log, _notify_event(cwd="/neighbor/cwd", thread_id="other"))
        t0 = time.monotonic()
        assert _run(sess._collect_response(baseline_completes=0, baseline_starts=0)) \
            == "poll answer"
        assert time.monotonic() - t0 >= 0.2, "must have waited out the polling cadence"
        assert sess._last_exit_reason == "rollout_done"

    def test_notify_before_rollout_flush_counts_as_progress(self, notify_log, monkeypatch):
        # The verified race: the push can land a hair BEFORE the task_complete
        # flush. A tiny stall window would otherwise cut the turn off; the
        # push's EDGE must reset the stall clock (buying the rollout a fresh
        # window to catch up) — and the payload alone must never complete the
        # turn. Timeline: stall clock starts at 0, push at ~0.25 (resetting the
        # clock), flush at 0.6 — inside the pushed-out window (0.25+0.5) yet
        # past the un-credited one (0.5).
        monkeypatch.setattr(codex_server, "RESPONSE_POLL_INTERVAL", 10.0)  # slow path never runs
        monkeypatch.setattr(codex_server, "RESPONSE_STALL_TIMEOUT", 0.5)
        monkeypatch.setattr(codex_server, "RESPONSE_HARD_TIMEOUT", 10.0)
        t0 = time.monotonic()

        def events():
            if time.monotonic() - t0 > 0.6:  # flush lands after the base stall window
                return [_meta(), _start(), _complete(last="late flush")]
            return [_meta()]  # rollout hasn't caught up with the push yet

        sess = _collect_session(monkeypatch, events=events)

        async def scenario():
            task = asyncio.create_task(
                sess._collect_response(baseline_completes=0, baseline_starts=0)
            )
            await asyncio.sleep(0.25)  # push lands mid-turn, before any rollout write
            _write_events(notify_log, _notify_event(cwd=str(sess.cwd)))
            return await task

        assert _run(scenario()) == "late flush"
        assert sess._last_exit_reason == "notify"


# ===========================================================================
# Stall timeout: a stray push must never disable the no-progress cut
# ===========================================================================

class TestStallTimeout:
    def test_stray_notify_does_not_disable_stall_timeout(self, notify_log, monkeypatch):
        # One matching event past the baseline, but the rollout NEVER gains a
        # task_complete (hook double-fire, recycled cwd — a push with no turn
        # behind it). Level-triggered progress would re-credit that event at
        # EVERY wake, pinning last_progress to now and riding the turn all the
        # way to the hard cap; the edge-triggered credit buys exactly one
        # stall window, so the turn must end "stalled" on time.
        monkeypatch.setattr(codex_server, "RESPONSE_POLL_INTERVAL", 0.05)
        monkeypatch.setattr(codex_server, "RESPONSE_STALL_TIMEOUT", 0.3)
        monkeypatch.setattr(codex_server, "RESPONSE_HARD_TIMEOUT", 3.0)
        sess = _collect_session(monkeypatch, events=lambda: [_meta()])
        _write_events(notify_log, _notify_event(cwd=str(sess.cwd)))
        t0 = time.monotonic()
        assert _run(sess._collect_response(baseline_completes=0, baseline_starts=0)) == ""
        elapsed = time.monotonic() - t0
        assert sess._last_exit_reason == "stalled"
        assert elapsed < 1.5, "one stray event buys one stall window, not the hard cap"


# ===========================================================================
# Kill switch: CODEX_NOTIFY off must be byte-for-byte the legacy path
# ===========================================================================

class TestKillSwitch:
    def test_disabled_never_reads_the_log_and_polls(self, notify_log, monkeypatch):
        monkeypatch.setattr(codex_server, "CODEX_NOTIFY", False)
        monkeypatch.setattr(codex_server, "RESPONSE_POLL_INTERVAL", 0.02)
        monkeypatch.setattr(codex_server, "RESPONSE_STALL_TIMEOUT", 5.0)
        monkeypatch.setattr(codex_server, "RESPONSE_HARD_TIMEOUT", 10.0)
        sess = _collect_session(
            monkeypatch,
            events=lambda: [_meta(), _start(), _complete(last="legacy answer")],
        )
        _write_events(notify_log, _notify_event(cwd=str(sess.cwd)))  # present, must be inert

        def boom():
            raise AssertionError("kill switch off: the notify log must never be read")

        monkeypatch.setattr(sess, "_notify_count", boom)
        assert _run(sess._collect_response(baseline_completes=0, baseline_starts=0)) \
            == "legacy answer"
        assert sess._last_exit_reason == "rollout_done"


# ===========================================================================
# Baselines: only events appended AFTER this turn's submit may fast-path it
# ===========================================================================

class TestBaselines:
    _EVENTS = [
        _meta(),
        _start("t1"), _complete("t1", last="one"),
        _start("t2"), _complete("t2", last="two"),
    ]

    def test_turn1_event_does_not_fast_path_turn2(self, notify_log, monkeypatch):
        monkeypatch.setattr(codex_server, "RESPONSE_POLL_INTERVAL", 0.25)
        monkeypatch.setattr(codex_server, "RESPONSE_STALL_TIMEOUT", 5.0)
        monkeypatch.setattr(codex_server, "RESPONSE_HARD_TIMEOUT", 10.0)
        sess = _collect_session(
            monkeypatch, events=lambda: self._EVENTS,
            notify_baseline=1,  # send() counted turn 1's event before submitting turn 2
        )
        _write_events(notify_log, _notify_event(cwd=str(sess.cwd)))  # turn 1's event only
        t0 = time.monotonic()
        assert _run(sess._collect_response(baseline_completes=1, baseline_starts=1)) == "two"
        assert time.monotonic() - t0 >= 0.2, "a stale event must not trigger the fast path"
        assert sess._last_exit_reason == "rollout_done"

    def test_turn2_own_event_fast_paths_turn2(self, notify_log, monkeypatch):
        monkeypatch.setattr(codex_server, "RESPONSE_POLL_INTERVAL", 1.0)
        monkeypatch.setattr(codex_server, "RESPONSE_STALL_TIMEOUT", 5.0)
        monkeypatch.setattr(codex_server, "RESPONSE_HARD_TIMEOUT", 10.0)
        sess = _collect_session(
            monkeypatch, events=lambda: self._EVENTS, notify_baseline=1,
        )
        _write_events(notify_log,
                      _notify_event(cwd=str(sess.cwd)),   # turn 1's
                      _notify_event(cwd=str(sess.cwd)))   # turn 2's — past the baseline
        t0 = time.monotonic()
        assert _run(sess._collect_response(baseline_completes=1, baseline_starts=1)) == "two"
        assert time.monotonic() - t0 < 0.5
        assert sess._last_exit_reason == "notify"

    def test_kill_resets_the_notify_baseline_and_cursor(self, notify_log, monkeypatch):
        async def fake_tmux(*args, **kwargs):
            return (0, "")

        monkeypatch.setattr(codex_server, "_tmux", fake_tmux)
        sess = codex_server.CodexSession(name="unit")
        _write_events(notify_log, _notify_event(cwd=str(sess.cwd)))
        assert sess._notify_count() == 1  # cursor now sits mid-file
        sess._last_notify_baseline = 7
        _run(sess._kill())
        assert sess._last_notify_baseline == 0
        assert (sess._notify_pos, sess._notify_matched) == (0, 0)
        # The respawn bumps the generation (fresh cwd): the reset cursor makes
        # the next baseline a fresh scan, and the dead generation's events —
        # matched at parse time against the OLD cwd — no longer count.
        sess._generation += 1  # what the following _spawn() does
        assert sess._notify_count() == 0


# ===========================================================================
# _notify_count: per-session attribution and hostile-input tolerance
# ===========================================================================

class TestNotifyCount:
    def test_missing_file_counts_zero(self, notify_log):
        # notify_log was never written -> no events, not an error
        assert codex_server.CodexSession(name="unit")._notify_count() == 0

    def test_malformed_lines_are_skipped(self, notify_log):
        sess = codex_server.CodexSession(name="unit")
        _write_events(
            notify_log,
            _notify_event(cwd=str(sess.cwd)),
            raw=["not json at all", '{"type": "agent-turn-complete", "cwd": ', "", "[1, 2]"],
        )
        assert sess._notify_count() == 1

    def test_cwd_matches_before_binding(self, notify_log):
        # first turn: _session_id is still None, so cwd is the only key
        sess = codex_server.CodexSession(name="unit")
        assert sess._session_id is None
        _write_events(notify_log, _notify_event(cwd=str(sess.cwd), thread_id="whoever"))
        assert sess._notify_count() == 1

    def test_thread_id_matches_when_cwd_differs(self, notify_log):
        # post-binding: the bound session id attributes the event to us even if
        # codex reports a different cwd view
        sess = codex_server.CodexSession(name="unit")
        sess._session_id = "thread-9"
        _write_events(notify_log, _notify_event(cwd="/somewhere/else", thread_id="thread-9"))
        assert sess._notify_count() == 1

    def test_unbound_session_never_matches_a_null_thread_id(self, notify_log):
        # pre-binding (_session_id None), a null thread-id must not match: that
        # would attribute EVERY malformed event to every unbound session
        sess = codex_server.CodexSession(name="unit")
        _write_events(notify_log, _notify_event(cwd="/somewhere/else", thread_id=None))
        assert sess._notify_count() == 0

    def test_non_turn_complete_events_do_not_count(self, notify_log):
        sess = codex_server.CodexSession(name="unit")
        ev = _notify_event(cwd=str(sess.cwd))
        ev["type"] = "agent-turn-started"
        _write_events(notify_log, ev)
        assert sess._notify_count() == 0


# ===========================================================================
# _notify_count is INCREMENTAL: the shared log grows for the whole server run
# (payloads embed full input messages), so a glance may only read NEW bytes
# ===========================================================================

class TestNotifyCountIncremental:
    def test_appends_are_counted_without_rereading_consumed_bytes(self, notify_log):
        sess = codex_server.CodexSession(name="unit")
        _write_events(notify_log, _notify_event(cwd=str(sess.cwd)))
        assert sess._notify_count() == 1
        # Corrupt the already-consumed prefix ON DISK, same byte length: any
        # implementation that re-parses from offset 0 now sees garbage fused
        # to the next line and counts 0. Only the appended bytes may be read.
        notify_log.write_bytes(b"x" * notify_log.stat().st_size)
        with open(notify_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(_notify_event(cwd=str(sess.cwd))) + "\n")
        assert sess._notify_count() == 2

    def test_truncation_resets_and_rescans_from_the_top(self, notify_log):
        sess = codex_server.CodexSession(name="unit")
        ev = _notify_event(cwd=str(sess.cwd))
        _write_events(notify_log, ev, ev, ev)
        assert sess._notify_count() == 3
        _write_events(notify_log, ev)  # rotated: smaller file, one event
        assert sess._notify_count() == 1

    def test_partial_trailing_line_left_for_the_next_glance(self, notify_log):
        sess = codex_server.CodexSession(name="unit")
        full = json.dumps(_notify_event(cwd=str(sess.cwd)))
        notify_log.write_text(full + "\n" + full[:20])  # append still in flight
        assert sess._notify_count() == 1
        with open(notify_log, "a", encoding="utf-8") as f:
            f.write(full[20:] + "\n")  # the hook's write completes
        assert sess._notify_count() == 2

    def test_missing_file_resets_so_a_recreated_log_is_rescanned(self, notify_log):
        sess = codex_server.CodexSession(name="unit")
        ev = _notify_event(cwd=str(sess.cwd))
        _write_events(notify_log, ev, ev)
        assert sess._notify_count() == 2
        notify_log.unlink()
        assert sess._notify_count() == 0
        _write_events(notify_log, ev)  # fresh file, shorter than the old cursor
        assert sess._notify_count() == 1


# ===========================================================================
# _build_command: the -c notify=[...] override, and its shell quoting
# ===========================================================================

class TestBuildCommand:
    def test_includes_quoted_notify_override_when_enabled(self, monkeypatch, tmp_path):
        hook = tmp_path / "notify-hook.sh"
        monkeypatch.setattr(codex_server, "CODEX_NOTIFY", True)
        monkeypatch.setattr(codex_server, "NOTIFY_HOOK", hook)
        monkeypatch.setattr(codex_server, "CODEX_EXTRA_ARGS", "")
        cmd = codex_server.CodexSession(name="unit")._build_command()
        # the TOML inline array survives the shell as ONE argv part...
        parts = shlex.split(cmd)
        assert parts[parts.index("-c") + 1] == f'notify=["{hook}"]'
        # ...because shlex.join single-quoted it (the eyeball check, pinned)
        assert f"""-c 'notify=["{hook}"]'""" in cmd

    def test_omitted_when_disabled(self, monkeypatch):
        monkeypatch.setattr(codex_server, "CODEX_NOTIFY", False)
        monkeypatch.setattr(codex_server, "CODEX_EXTRA_ARGS", "")
        cmd = codex_server.CodexSession(name="unit")._build_command()
        parts = shlex.split(cmd)
        assert "-c" not in parts
        assert "notify=" not in cmd


# ===========================================================================
# Hook-path safety: the path lands in TOML with NO escaping — an unsafe char
# must degrade to pure polling, never a codex that rejects its config (503s)
# ===========================================================================

class TestHookPathSafety:
    UNSAFE = Path('/tmp/codex "quoted"/notify-hook.sh')

    def test_unsafe_path_omits_the_notify_flag(self, monkeypatch):
        monkeypatch.setattr(codex_server, "CODEX_NOTIFY", True)
        monkeypatch.setattr(codex_server, "CODEX_EXTRA_ARGS", "")
        monkeypatch.setattr(codex_server, "NOTIFY_HOOK", self.UNSAFE)
        cmd = codex_server.CodexSession(name="unit")._build_command()
        assert "-c" not in shlex.split(cmd)
        assert "notify=" not in cmd

    def test_unsafe_path_declines_the_hook_install(self, monkeypatch, tmp_path):
        bad = tmp_path / 'evil"dir'
        monkeypatch.setattr(codex_server, "NOTIFY_DIR", bad)
        monkeypatch.setattr(codex_server, "NOTIFY_LOG", bad / "events.jsonl")
        monkeypatch.setattr(codex_server, "NOTIFY_HOOK", bad / "notify-hook.sh")
        codex_server._install_notify_hook()
        assert not bad.exists(), "nothing may be created for an unwireable hook"

    def test_safe_path_installs_an_executable_hook(self, monkeypatch, tmp_path):
        d = tmp_path / "notify"
        monkeypatch.setattr(codex_server, "NOTIFY_DIR", d)
        monkeypatch.setattr(codex_server, "NOTIFY_LOG", d / "events.jsonl")
        monkeypatch.setattr(codex_server, "NOTIFY_HOOK", d / "notify-hook.sh")
        codex_server._install_notify_hook()
        hook = d / "notify-hook.sh"
        assert hook.is_file() and os.access(hook, os.X_OK)
        assert str(d / "events.jsonl") in hook.read_text()


# ===========================================================================
# send() captures the notify baseline at submit — on BOTH branches. Without
# it, an earlier turn's event would fast-path the next turn's collection.
# ===========================================================================

class TestSendBaselines:
    def _wire(self, sess, monkeypatch, captured):
        async def is_alive():
            return True

        async def collect(baseline_completes, baseline_starts):
            captured["baselines"] = (baseline_completes, baseline_starts)
            return "ok"

        monkeypatch.setattr(sess, "is_alive", is_alive)
        monkeypatch.setattr(sess, "_collect_response", collect)

    def test_first_turn_send_baselines_preseeded_events(self, notify_log, tmp_path,
                                                        monkeypatch):
        # 3 matching events already on disk when the FIRST turn is submitted:
        # send() must record them as the baseline (before the rollout binds),
        # or any of them could fast-path this turn.
        monkeypatch.setattr(codex_server, "CODEX_SESSIONS_DIR", tmp_path / "none")
        rollout = tmp_path / "rollout-unit.jsonl"
        _write_events(rollout, _meta(session_id="thread-1", cwd="/x"), _start())
        sess = codex_server.CodexSession(name="unit")
        captured = {}
        self._wire(sess, monkeypatch, captured)

        async def submit(prompt):
            pass

        async def detect(before):
            return rollout

        monkeypatch.setattr(sess, "_submit", submit)
        monkeypatch.setattr(sess, "_detect_new_session", detect)
        _write_events(notify_log, *[_notify_event(cwd=str(sess.cwd))] * 3)

        assert _run(sess.send("q")) == "ok"
        assert sess._last_notify_baseline == 3
        assert sess._session_id == "thread-1"
        assert captured["baselines"] == (0, 1)

    def test_next_turn_send_baselines_previous_turns_events(self, notify_log, tmp_path,
                                                            monkeypatch):
        # Turn 2: one event from turn 1 (matched by thread-id) plus a cwd match
        # are on disk at submit — both belong to the past and set the baseline.
        rollout = tmp_path / "rollout-unit.jsonl"
        _write_events(rollout, _meta(), _start("t1"), _complete("t1", last="one"))
        sess = codex_server.CodexSession(name="unit")
        sess._rollout_path = rollout
        sess._session_id = "thread-1"
        sess._turn_count = 1
        captured = {}
        self._wire(sess, monkeypatch, captured)

        async def submit_confirmed(prompt, baseline_starts):
            pass

        monkeypatch.setattr(sess, "_submit_confirmed", submit_confirmed)
        _write_events(notify_log,
                      _notify_event(cwd="/elsewhere", thread_id="thread-1"),
                      _notify_event(cwd=str(sess.cwd)))

        assert _run(sess.send("q")) == "ok"
        assert sess._last_notify_baseline == 2
        assert sess._last_baseline_completes == 1
        assert captured["baselines"] == (1, 1)


# ===========================================================================
# /chat observability: `via` is optional, so old clients see no change
# ===========================================================================

def test_chat_response_via_defaults_to_null():
    r = codex_server.ChatResponse(response="x", session="s", turn=1, elapsed_ms=0)
    assert r.via is None
    assert codex_server.ChatResponse(
        response="x", session="s", turn=1, elapsed_ms=0, via="notify"
    ).via == "notify"
