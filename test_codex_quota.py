"""
Unit tests for the codex bridge's usage-limit handling and model re-pin.

A turn that dies on the ChatGPT usage limit ends with a task_complete whose
answer is null and whose `error` names the cause; the bridge used to return a
generic empty 504 and /last reported "done, 0 chars". Worse, codex then
silently switches the thread to a fallback model, defeating the CODEX_MODEL
pin. These pin the new behaviour:

  * classification — the quota task_complete is `usage_limit` (never an empty
    answer); any other terminal error is `error`; a turn served by a model
    other than the pin is `model_drift`
  * the "try again at …" reset time: time-only (today, rolling to tomorrow),
    dated, unparseable -> null
  * drift read from the rollout: a thread_settings_applied after the last
    completed turn naming a different model
  * `codex resume <thread>` carries every launch flag, and the re-pin runs
    BEFORE the next prompt (kill confirmed, resume in the same tmux session,
    baselines re-read from the same rollout)
  * /chat answers 429 (usage_limit) / 409 (model_drift) with a JSON body, and
    /last reports the same verdicts with done=true, response="" plus the
    additive error / resets_at fields

No server / codex / tmux / Docker — the rollout, the screen and tmux are mocked.
Run:  ./.venv/bin/python -m pytest test_codex_quota.py -v
"""

import asyncio
import json
import os
import shlex
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import codex_server


def _run(coro):
    return asyncio.run(coro)


PIN = "gpt-6-astra"
FALLBACK = "gpt-5.6-luna"

# A REAL usage-limit task_complete from a container rollout (2026-09-04, ids
# redacted), verbatim in shape: null answer + error naming the cause.
QUOTA_LINE = (
    '{"timestamp":"2026-09-04T08:13:42.297Z","ordinal":48,"type":"event_msg","payload":'
    '{"type":"task_complete","turn_id":"01a06b7a-0000-0000-0000-000000000001",'
    '"last_agent_message":null,"error":{"message":"You\'ve hit your usage limit. '
    'Upgrade to Pro (https://chatgpt.com/explore/pro), visit '
    'https://chatgpt.com/codex/settings/usage to purchase more credits or try again '
    'at 12:32 PM.","codex_error_info":"usage_limit_exceeded"},"started_at":1788509575,'
    '"completed_at":1788509622,"duration_ms":46459,"time_to_first_token_ms":3759}}'
)
QUOTA_EVENT = json.loads(QUOTA_LINE)
QUOTA_MSG = QUOTA_EVENT["payload"]["error"]["message"]


# --- event builders ---------------------------------------------------------

def _meta(session_id="thread-1", cwd="/x"):
    return {"type": "session_meta", "payload": {"id": session_id, "cwd": cwd}}


def _start(turn_id="t1"):
    return {"type": "event_msg", "payload": {"type": "task_started", "turn_id": turn_id},
            "timestamp": "t"}


def _complete(turn_id="t1", last="the answer", error=None):
    p = {"type": "task_complete", "turn_id": turn_id, "last_agent_message": last}
    if error is not None:
        p["error"] = error
    return {"type": "event_msg", "payload": p, "timestamp": "t"}


def _context(model=PIN, effort="xhigh"):
    return {"type": "turn_context", "payload": {"model": model, "effort": effort},
            "timestamp": "t"}


def _settings(model):
    return {"type": "event_msg", "timestamp": "t",
            "payload": {"type": "thread_settings_applied", "thread_id": "thread-1",
                        "thread_settings": {"model": model, "reasoning_effort": "medium"}}}


def _assistant(text):
    return {"type": "response_item", "timestamp": "t",
            "payload": {"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": text}]}}


# The shape seen in the wild: a good turn, then the quota turn, then the silent
# fallback settings written while idle.
GOOD_TURN = [_meta(), _start("t1"), _context(PIN), _complete("t1", last="answer one")]
QUOTA_TURN = [_start("t2"), _context(PIN), QUOTA_EVENT]
DRIFTED = GOOD_TURN + QUOTA_TURN + [_settings(FALLBACK), _settings(FALLBACK)]


# ===========================================================================
# Classification
# ===========================================================================

class TestVerdict:
    def test_real_quota_line_is_usage_limit_not_an_empty_answer(self):
        v = codex_server._turn_verdict(GOOD_TURN + QUOTA_TURN, baseline_completes=1, pin=PIN)
        assert v["status"] == "usage_limit"
        assert v["error"] == QUOTA_MSG
        assert v["model"] == PIN
        assert v["resets_at"]  # parsed (the exact instant is pinned in TestResetTime)

    def test_message_fallback_when_error_info_missing(self):
        err = {"message": "You've hit your usage limit. Try again later."}
        assert codex_server._is_usage_limit(err) is True
        assert codex_server._is_usage_limit({"message": "boom"}) is False
        assert codex_server._is_usage_limit(None) is False

    def test_clean_turn_is_done(self):
        v = codex_server._turn_verdict(GOOD_TURN, baseline_completes=0, pin=PIN)
        assert (v["status"], v["error"], v["resets_at"], v["model"]) == ("done", None, None, PIN)

    def test_other_terminal_error_with_no_text_is_error(self):
        events = [_meta(), _start(), _context(),
                  _complete(last=None, error={"message": "stream disconnected"})]
        v = codex_server._turn_verdict(events, 0, PIN)
        assert (v["status"], v["error"]) == ("error", "stream disconnected")

    def test_error_alongside_an_answer_keeps_the_answer(self):
        events = [_meta(), _start(), _context(), _assistant("partial but real"),
                  _complete(last=None, error={"message": "late hiccup"})]
        assert codex_server._turn_verdict(events, 0, PIN)["status"] == "done"

    def test_answer_from_a_non_pinned_model_is_model_drift(self):
        events = [_meta(), _start(), _context(FALLBACK, "medium"),
                  _complete(last="an answer from the wrong model")]
        v = codex_server._turn_verdict(events, 0, PIN)
        assert v["status"] == "model_drift"
        assert v["model"] == FALLBACK
        assert FALLBACK in v["error"] and PIN in v["error"]

    def test_no_pin_means_no_drift_verdict(self):
        events = [_meta(), _start(), _context(FALLBACK), _complete(last="fine")]
        assert codex_server._turn_verdict(events, 0, "")["status"] == "done"

    def test_quota_wins_over_drift(self):
        events = [_meta(), _start(), _context(FALLBACK), QUOTA_EVENT]
        assert codex_server._turn_verdict(events, 0, PIN)["status"] == "usage_limit"

    def test_drift_wins_over_a_generic_error(self):
        # An errored, answerless turn on the wrong model is still a drift: the
        # session must re-pin (409), not shrug it off as a 502.
        events = [_meta(), _start(), _context(FALLBACK),
                  _complete(last=None, error={"message": "stream disconnected"})]
        assert codex_server._turn_verdict(events, 0, PIN)["status"] == "model_drift"

    def test_reset_is_anchored_to_the_completion_timestamp(self):
        # Re-reading the same turn later (/last polls) must not roll the reset
        # over to tomorrow: "12:32 PM" is relative to when the turn ended.
        anchor = datetime.fromtimestamp(codex_server._parse_iso_ts(QUOTA_EVENT["timestamp"]))
        expected = anchor.replace(hour=12, minute=32, second=0, microsecond=0)
        if expected <= anchor:
            expected += timedelta(days=1)
        v = codex_server._turn_verdict(GOOD_TURN + QUOTA_TURN, 1, PIN)
        assert v["resets_at"] == _iso(expected)
        assert codex_server._turn_verdict(GOOD_TURN + QUOTA_TURN, 1, PIN)["resets_at"] == v["resets_at"]

    def test_reset_falls_back_to_completed_at_then_null(self):
        no_ts = json.loads(QUOTA_LINE)
        del no_ts["timestamp"]  # payload.completed_at (epoch) still anchors it
        anchor = datetime.fromtimestamp(no_ts["payload"]["completed_at"])
        expected = anchor.replace(hour=12, minute=32, second=0, microsecond=0)
        if expected <= anchor:
            expected += timedelta(days=1)
        v = codex_server._turn_verdict([_meta(), _start(), no_ts], 0, PIN)
        assert (v["status"], v["resets_at"]) == ("usage_limit", _iso(expected))
        del no_ts["payload"]["completed_at"]  # nothing to anchor to -> null, still 429
        v = codex_server._turn_verdict([_meta(), _start(), no_ts], 0, PIN)
        assert (v["status"], v["resets_at"]) == ("usage_limit", None)

    def test_scopes_to_the_new_turn(self):
        # baseline=1: turn 1's clean answer must not mask turn 2's quota error,
        # and turn 2's quota must not leak into a verdict on turn 1 (baseline=0).
        events = GOOD_TURN + QUOTA_TURN
        assert codex_server._turn_verdict(events, 1, PIN)["status"] == "usage_limit"
        assert codex_server._turn_verdict(events, 0, PIN)["status"] == "done"
        assert codex_server._turn_model(events, 1) == PIN
        assert codex_server._turn_error(events, 0) is None


# ===========================================================================
# Reset time
# ===========================================================================

def _iso(dt: datetime) -> str:
    return dt.astimezone().isoformat(timespec="seconds")


class TestResetTime:
    NOW = datetime(2026, 9, 6, 10, 0, 0)  # naive = the container's local clock

    def test_time_only_later_today(self):
        got = codex_server._parse_reset_time("... or try again at 2:12 PM.", now=self.NOW)
        assert got == _iso(datetime(2026, 9, 6, 14, 12))

    def test_time_only_already_past_rolls_to_tomorrow(self):
        got = codex_server._parse_reset_time("... or try again at 9:50 AM.", now=self.NOW)
        assert got == _iso(datetime(2026, 9, 7, 9, 50))

    def test_time_equal_to_now_rolls_to_tomorrow(self):
        got = codex_server._parse_reset_time("try again at 10:00 AM.", now=self.NOW)
        assert got == _iso(datetime(2026, 9, 7, 10, 0))

    def test_dated_form(self):
        got = codex_server._parse_reset_time(
            "... or try again at Aug 20th, 2026 7:21 AM.", now=self.NOW)
        assert got == _iso(datetime(2026, 8, 20, 7, 21))

    def test_real_message_parses(self):
        assert codex_server._parse_reset_time(QUOTA_MSG, now=self.NOW) == \
            _iso(datetime(2026, 9, 6, 12, 32))

    def test_unparseable_or_absent_is_none(self):
        assert codex_server._parse_reset_time("try again at .", now=self.NOW) is None
        assert codex_server._parse_reset_time("try again at soon-ish.", now=self.NOW) is None
        assert codex_server._parse_reset_time("You've hit your usage limit.", now=self.NOW) is None
        assert codex_server._parse_reset_time(None) is None

    def test_carries_the_local_offset(self):
        got = codex_server._parse_reset_time("try again at 2:12 PM.", now=self.NOW)
        assert datetime.fromisoformat(got).utcoffset() is not None

    def test_time_only_without_an_anchor_is_unknowable(self):
        # No anchor = no day to pin "2:12 PM" to (never the wall clock, which
        # would make the answer depend on when it is re-read): null.
        assert codex_server._parse_reset_time("try again at 2:12 PM.") is None
        # the dated form needs no anchor
        assert codex_server._parse_reset_time("try again at Aug 20th, 2026 7:21 AM.") == \
            _iso(datetime(2026, 8, 20, 7, 21))


# ===========================================================================
# Drift read from the rollout (thread_settings_applied)
# ===========================================================================

class TestDriftDetector:
    def test_fallback_after_the_quota_turn_is_drift(self):
        assert codex_server._drifted_model(DRIFTED, PIN) == FALLBACK

    def test_settings_matching_the_pin_are_not_drift(self):
        events = GOOD_TURN + [_settings(PIN)]
        assert codex_server._drifted_model(events, PIN) is None

    def test_settings_before_the_last_completed_turn_are_history(self):
        # The fallback was applied, then a turn completed (and was judged on its
        # own turn_context): those settings no longer say anything about NOW.
        events = DRIFTED + [_start("t3"), _context(FALLBACK), _complete("t3", last="x")]
        assert codex_server._drifted_model(events, PIN) is None

    def test_scan_start_skips_settings_a_repin_already_undid(self):
        # After the re-pin the old fallback lines still sit in the same rollout;
        # scanning from the re-pin point must not read them as a fresh drift...
        assert codex_server._drifted_model(DRIFTED, PIN, start=len(DRIFTED)) is None
        # ...while a fallback applied AFTER it is.
        again = DRIFTED + [_settings(PIN), _settings(FALLBACK)]
        assert codex_server._drifted_model(again, PIN, start=len(DRIFTED)) == FALLBACK

    def test_no_pin_disables_detection(self):
        assert codex_server._drifted_model(DRIFTED, "") is None

    def test_startup_settings_on_a_fresh_thread(self):
        # codex writes the current settings at startup too — only a different
        # model counts.
        assert codex_server._drifted_model([_meta(), _settings(PIN)], PIN) is None
        assert codex_server._drifted_model([_meta(), _settings(FALLBACK)], PIN) == FALLBACK


# ===========================================================================
# _build_command(resume_id=...)
# ===========================================================================

class TestResumeCommand:
    @pytest.fixture(autouse=True)
    def _pins(self, monkeypatch, tmp_path):
        monkeypatch.setattr(codex_server, "CODEX_MODEL", PIN)
        monkeypatch.setattr(codex_server, "CODEX_EFFORT", "xhigh")
        monkeypatch.setattr(codex_server, "CODEX_SERVICE_TIER", "default")
        monkeypatch.setattr(codex_server, "CODEX_EXTRA_ARGS", "--add-dir /repos")
        monkeypatch.setattr(codex_server, "CODEX_NOTIFY", True)
        monkeypatch.setattr(codex_server, "NOTIFY_HOOK", tmp_path / "notify-hook.sh")

    def test_resume_carries_every_launch_flag(self, tmp_path):
        sess = codex_server.CodexSession(name="unit@dev")
        fresh = shlex.split(sess._build_command())
        resumed = shlex.split(sess._build_command(resume_id="thread-abc"))
        assert resumed[:3] == [codex_server.CODEX_CMD, "resume", "thread-abc"]
        # everything a fresh launch gets, in the same order, follows the id
        assert resumed[3:] == fresh[1:]
        assert "--dangerously-bypass-approvals-and-sandbox" in resumed
        assert resumed[resumed.index("-m") + 1] == PIN
        c_values = [resumed[k + 1] for k, p in enumerate(resumed) if p == "-c"]
        assert 'model_reasoning_effort="xhigh"' in c_values
        assert 'service_tier="default"' in c_values
        hook = str(tmp_path / "notify-hook.sh")
        assert any(v.startswith("hooks.Stop=") and f'command="{hook}"' in v for v in c_values)
        assert "--dangerously-bypass-hook-trust" in resumed
        assert resumed.count("--add-dir") == 2  # CODEX_EXTRA_ARGS grant + the worktree
        assert str(sess.cwd) in resumed

    def test_fresh_launch_has_no_resume(self):
        parts = shlex.split(codex_server.CodexSession(name="unit@dev")._build_command())
        assert "resume" not in parts


# ===========================================================================
# The re-pin: kill confirmed, resume in the same tmux session, state kept
# ===========================================================================

def _free_pid() -> int:
    pid = 4_000_000
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return pid
        except PermissionError:
            pass
        pid -= 1


@pytest.fixture()
def tmux_rec(monkeypatch):
    """Record tmux calls; display-message answers with a pid that does not exist
    (so the exit wait returns at once)."""
    calls: list[tuple] = []
    pid = _free_pid()

    async def fake_tmux(*args, **kwargs):
        calls.append(args)
        if args and args[0] == "display-message":
            return (0, f"{pid}\n")
        return (0, "")

    monkeypatch.setattr(codex_server, "_tmux", fake_tmux)
    return calls


def _bound_session(monkeypatch, events, *, name="unit@dev"):
    sess = codex_server.CodexSession(name=name)
    sess._rollout_path = Path("/tmp/does-not-matter/rollout-x.jsonl")
    sess._session_id = "thread-1"
    sess._turn_count = 1
    sess._last_baseline_completes = 1
    monkeypatch.setattr(codex_server, "_read_rollout", lambda path: list(events()))

    async def capture():
        return f"› earlier prompt echoed\n• reply\n\n› Ask Codex to do anything\n\n  {PIN} xhigh · {sess.cwd}"

    monkeypatch.setattr(sess, "_capture", capture)
    return sess


class TestComposerIdle:
    """A resumed thread's ready signal: the LAST caret line with the status line
    (which names this session's cwd) beneath it, and no busy marker."""

    def test_idle_composer_with_status_line_is_ready(self, monkeypatch):
        sess = codex_server.CodexSession(name="unit@dev")
        screen = f"› old prompt\n• old reply\n\n› Ask Codex to do anything\n\n  {PIN} xhigh · {sess.cwd}"
        assert sess._composer_idle(screen) is True

    def test_history_caret_alone_is_not_ready(self):
        # The transcript is still being redrawn: prompts echo with the same
        # glyph but no status line follows yet.
        sess = codex_server.CodexSession(name="unit@dev")
        assert sess._composer_idle("› old prompt\n• old reply\n") is False

    def test_busy_screen_is_not_ready(self):
        sess = codex_server.CodexSession(name="unit@dev")
        screen = f"› Ask Codex\n• Working (3s • esc to interrupt)\n  {PIN} xhigh · {sess.cwd}"
        assert sess._composer_idle(screen) is False

    def test_another_sessions_cwd_is_not_ours(self):
        sess = codex_server.CodexSession(name="unit@dev")
        assert sess._composer_idle(f"› Ask Codex\n\n  {PIN} xhigh · /tmp/other/c1") is False


class TestRepin:
    @pytest.fixture(autouse=True)
    def _env(self, monkeypatch):
        monkeypatch.setattr(codex_server, "CODEX_MODEL", PIN)
        monkeypatch.setattr(codex_server, "CODEX_EFFORT", "")
        monkeypatch.setattr(codex_server, "CODEX_SERVICE_TIER", "")
        monkeypatch.setattr(codex_server, "CODEX_EXTRA_ARGS", "")
        monkeypatch.setattr(codex_server, "CODEX_NOTIFY", False)
        monkeypatch.setattr(codex_server, "SUBMIT_GRACE", 0.0)
        monkeypatch.setattr(codex_server, "RESPONSE_POLL_INTERVAL", 0.02)

    def test_kills_waits_then_resumes_in_the_same_tmux_session(self, tmux_rec, monkeypatch):
        sess = _bound_session(monkeypatch, lambda: DRIFTED)
        sess._needs_repin = True
        _run(sess._repin(FALLBACK))

        names = [c[0] for c in tmux_rec]
        assert names.index("kill-session") < names.index("new-session")
        new = next(c for c in tmux_rec if c[0] == "new-session")
        assert new[new.index("-s") + 1] == sess.tmux_session  # same session name
        assert f"resume thread-1" in new[-1] and f"-m {PIN}" in new[-1]
        # the ready wait ran against the resumed screen (caret line = ready)
        assert sess._ready_at > 0
        # state kept: same rollout/thread, flag cleared, drift scan moved past
        # the fallback lines the resume just undid
        assert sess._rollout_path == Path("/tmp/does-not-matter/rollout-x.jsonl")
        assert sess._session_id == "thread-1"
        assert sess._needs_repin is False
        assert sess._drift_scan_from == len(DRIFTED)

    def test_failed_resume_raises_and_drops_the_worktree(self, tmux_rec, monkeypatch):
        removed = []

        async def remove(cwd):
            removed.append(cwd)

        async def boom(self, resumed=False):
            raise RuntimeError("codex startup timed out after 60s")

        monkeypatch.setattr(codex_server.worktree, "remove", remove)
        monkeypatch.setattr(codex_server.CodexSession, "_wait_ready", boom)
        sess = _bound_session(monkeypatch, lambda: DRIFTED)
        with pytest.raises(RuntimeError, match="re-pin of thread thread-1 failed"):
            _run(sess._repin(FALLBACK))
        assert removed == [sess.cwd]

    def test_exit_wait_sigkills_only_the_lingering_process(self, monkeypatch):
        # pid 1 goes away at once (and must never be signalled again — its
        # number could be reused); pid 2 lingers and gets the SIGKILL.
        monkeypatch.setattr(codex_server, "REPIN_EXIT_TIMEOUT", 0.05)
        polls = {2: 0}
        sent = []

        def fake_alive(pid):
            if pid == 1:
                return False
            polls[2] += 1
            return not sent  # dies once SIGKILLed

        def fake_kill(pid, sig):
            sent.append((pid, sig))

        monkeypatch.setattr(codex_server, "_pid_alive", fake_alive)
        monkeypatch.setattr(codex_server.os, "kill", fake_kill)
        _run(codex_server.CodexSession(name="unit@dev")._await_process_exit({1, 2}))
        assert sent == [(2, 9)]

    def test_zombie_counts_as_exited(self, tmp_path, monkeypatch):
        # A zombie has released the thread lock and cannot be killed: waiting on
        # it would only end in a false "did not exit after SIGKILL".
        proc = tmp_path / "proc"
        (proc / "777").mkdir(parents=True)
        (proc / "777" / "stat").write_text("777 (codex (tui)) Z 1 777 777 0 -1 4194560\n")
        (proc / "778").mkdir()
        (proc / "778" / "stat").write_text("778 (codex) S 1 778 778 0 -1 4194560\n")
        import builtins
        real_open = open
        monkeypatch.setattr(builtins, "open", lambda p, *a, **k: real_open(
            str(p).replace("/proc/", f"{proc}/", 1) if str(p).startswith("/proc/") else p, *a, **k))
        assert codex_server._pid_alive(777) is False
        assert codex_server._pid_alive(778) is True
        # unreadable /proc entry: falls back to the signal probe
        assert codex_server._pid_alive(_free_pid()) is False

    def test_unknown_process_aborts_before_killing_anything(self, monkeypatch):
        # Neither the pane pid nor a /proc match: the thread's lock holder
        # cannot be confirmed gone, so nothing is killed and the session (flag
        # included) is left exactly as it was.
        calls = []

        async def fake_tmux(*args, **kwargs):
            calls.append(args[0])
            return (0, "") if args[0] != "display-message" else (1, "no pane")

        monkeypatch.setattr(codex_server, "_tmux", fake_tmux)
        monkeypatch.setattr(codex_server.CodexSession, "_proc_pids", lambda self: set())
        sess = _bound_session(monkeypatch, lambda: DRIFTED)
        sess._needs_repin = True
        with pytest.raises(RuntimeError, match="cannot identify the codex process"):
            _run(sess._repin(FALLBACK))
        assert "kill-session" not in calls and "new-session" not in calls
        assert sess._needs_repin is True and sess._session_id == "thread-1"

    def test_proc_scan_finds_a_process_granted_our_cwd(self, monkeypatch, tmp_path):
        sess = codex_server.CodexSession(name="unit@dev")
        # our own process would match if it carried the grant; it does not
        assert sess._proc_pids() == set()
        # a fake /proc with one matching and one foreign cmdline
        proc = tmp_path / "proc"
        (proc / "4242").mkdir(parents=True)
        (proc / "4242" / "cmdline").write_bytes(
            f"codex\0--dangerously-bypass-approvals-and-sandbox\0--add-dir\0{sess.cwd}\0".encode())
        (proc / "4243").mkdir()
        (proc / "4243" / "cmdline").write_bytes(b"codex\0--add-dir\0/tmp/other/c1\0")
        (proc / "self").mkdir()
        real_listdir, real_open = os.listdir, open

        monkeypatch.setattr(codex_server.os, "listdir",
                            lambda p: real_listdir(proc) if p == "/proc" else real_listdir(p))
        import builtins
        monkeypatch.setattr(builtins, "open", lambda p, *a, **k: real_open(
            str(p).replace("/proc/", f"{proc}/", 1) if str(p).startswith("/proc/") else p, *a, **k))
        assert sess._proc_pids() == {4242}

    def test_send_repins_when_the_last_turn_completed_late_on_the_limit(self, monkeypatch):
        # /chat timed out on turn 2; it then ended on the usage limit and no
        # fallback settings were written yet: neither the flag nor the settings
        # scan knows, so the completion itself must trigger the re-pin — and the
        # baselines are published BEFORE it so a concurrent /last reads pending.
        sess = _bound_session(monkeypatch, lambda: GOOD_TURN + QUOTA_TURN)
        sess._turn_count = 2
        sess._last_baseline_completes = 1  # turn 2's submit baseline
        seen = {}

        async def repin(drifted):
            seen["drifted"] = drifted
            seen["baseline_at_repin"] = sess._last_baseline_completes

        async def submit(prompt, baseline_starts):
            pass

        async def collect(baseline_completes, baseline_starts):
            sess._last_verdict = {"status": "done"}
            return "ok"

        async def alive():
            return True

        monkeypatch.setattr(sess, "_repin", repin)
        monkeypatch.setattr(sess, "_submit_confirmed", submit)
        monkeypatch.setattr(sess, "_collect_response", collect)
        monkeypatch.setattr(sess, "is_alive", alive)
        assert _run(sess.send("next")) == "ok"
        assert seen == {"drifted": None, "baseline_at_repin": 2}

    def test_send_repins_before_the_prompt_when_flagged(self, monkeypatch):
        sess = _bound_session(monkeypatch, lambda: GOOD_TURN + [_settings(PIN)])
        sess._needs_repin = True
        order = []

        async def repin(drifted):
            order.append(("repin", drifted))

        async def submit(prompt, baseline_starts):
            order.append(("submit", baseline_starts))

        async def collect(baseline_completes, baseline_starts):
            order.append(("collect", baseline_completes))
            sess._last_verdict = {"status": "done"}
            return "ok"

        async def alive():
            return True

        monkeypatch.setattr(sess, "_repin", repin)
        monkeypatch.setattr(sess, "_submit_confirmed", submit)
        monkeypatch.setattr(sess, "_collect_response", collect)
        monkeypatch.setattr(sess, "is_alive", alive)
        assert _run(sess.send("next")) == "ok"
        assert order == [("repin", None), ("submit", 1), ("collect", 1)]

    def test_send_repins_on_rollout_drift_even_without_the_flag(self, monkeypatch):
        # A bridge restart loses the flag; the fallback settings in the rollout
        # still trigger the re-pin, and the baselines come from the file AFTER it.
        state = {"events": DRIFTED}
        sess = _bound_session(monkeypatch, lambda: state["events"])
        seen = {}

        async def repin(drifted):
            seen["drifted"] = drifted
            state["events"] = DRIFTED + [_settings(PIN)]  # what resume writes

        async def submit(prompt, baseline_starts):
            seen["starts"] = baseline_starts

        async def collect(baseline_completes, baseline_starts):
            seen["completes"] = baseline_completes
            sess._last_verdict = {"status": "done"}
            return "ok"

        async def alive():
            return True

        monkeypatch.setattr(sess, "_repin", repin)
        monkeypatch.setattr(sess, "_submit_confirmed", submit)
        monkeypatch.setattr(sess, "_collect_response", collect)
        monkeypatch.setattr(sess, "is_alive", alive)
        assert _run(sess.send("next")) == "ok"
        assert seen == {"drifted": FALLBACK, "starts": 2, "completes": 2}

    def test_no_pin_never_repins(self, monkeypatch):
        monkeypatch.setattr(codex_server, "CODEX_MODEL", "")
        sess = _bound_session(monkeypatch, lambda: DRIFTED)
        sess._needs_repin = True

        async def repin(drifted):
            raise AssertionError("must not re-pin without a pin")

        async def submit(prompt, baseline_starts):
            pass

        async def collect(baseline_completes, baseline_starts):
            sess._last_verdict = {"status": "done"}
            return "ok"

        async def alive():
            return True

        monkeypatch.setattr(sess, "_repin", repin)
        monkeypatch.setattr(sess, "_submit_confirmed", submit)
        monkeypatch.setattr(sess, "_collect_response", collect)
        monkeypatch.setattr(sess, "is_alive", alive)
        assert _run(sess.send("next")) == "ok"


# ===========================================================================
# _collect_response: the quota turn ends collection at once, withholds text,
# flags the re-pin, dumps a diagnostic; send() raises the TurnFailure
# ===========================================================================

@pytest.fixture()
def collect_env(monkeypatch, tmp_path):
    monkeypatch.setattr(codex_server, "CODEX_MODEL", PIN)
    monkeypatch.setattr(codex_server, "CODEX_NOTIFY", False)
    monkeypatch.setattr(codex_server, "RESPONSE_POLL_INTERVAL", 0.02)
    monkeypatch.setattr(codex_server, "RESPONSE_MIN_WAIT", 0.0)
    monkeypatch.setattr(codex_server, "RESPONSE_STALL_TIMEOUT", 5.0)
    monkeypatch.setattr(codex_server, "RESPONSE_HARD_TIMEOUT", 10.0)
    monkeypatch.setattr(codex_server, "TIMEOUT_LOG_DIR", tmp_path / "timeouts")
    return tmp_path / "timeouts"


class TestCollectVerdict:
    def test_quota_turn_is_withheld_flagged_and_dumped(self, collect_env, monkeypatch):
        sess = _bound_session(monkeypatch, lambda: GOOD_TURN + QUOTA_TURN)
        assert _run(sess._collect_response(baseline_completes=1, baseline_starts=1)) == ""
        assert sess._last_verdict["status"] == "usage_limit"
        assert sess._last_verdict["error"] == QUOTA_MSG
        assert sess._needs_repin is True
        dumps = list(collect_env.glob("*.log"))
        assert len(dumps) == 1 and "reason=usage_limit" in dumps[0].read_text()

    def test_drifted_answer_is_withheld_and_flagged(self, collect_env, monkeypatch):
        events = GOOD_TURN + [_start("t2"), _context(FALLBACK, "medium"),
                              _complete("t2", last="wrong-model answer")]
        sess = _bound_session(monkeypatch, lambda: events)
        assert _run(sess._collect_response(1, 1)) == ""
        assert sess._last_verdict["status"] == "model_drift"
        assert sess._needs_repin is True

    def test_clean_turn_is_unchanged(self, collect_env, monkeypatch):
        sess = _bound_session(monkeypatch, lambda: GOOD_TURN)
        assert _run(sess._collect_response(0, 0)) == "answer one"
        assert sess._last_verdict["status"] == "done"
        assert sess._needs_repin is False

    def test_send_raises_turn_failure(self, collect_env, monkeypatch):
        state = {"events": GOOD_TURN}
        sess = _bound_session(monkeypatch, lambda: state["events"])

        async def submit(prompt, baseline_starts):
            state["events"] = GOOD_TURN + QUOTA_TURN  # the turn lands after the paste

        async def alive():
            return True

        monkeypatch.setattr(sess, "_submit_confirmed", submit)
        monkeypatch.setattr(sess, "is_alive", alive)
        with pytest.raises(codex_server.TurnFailure) as exc:
            _run(sess.send("more"))
        assert exc.value.verdict["status"] == "usage_limit"


# ===========================================================================
# Endpoints: /chat 429 + 409 bodies, /last 200 usage_limit / model_drift shape
# ===========================================================================

@pytest.fixture()
def app_env(monkeypatch, collect_env):
    """TestClient without tmux/git: lifespan hooks stubbed, and the session the
    endpoints reach is a real CodexSession whose rollout/screen are mocked."""
    async def noop():
        return None

    monkeypatch.setattr(codex_server, "_ensure_tmux_server", noop)
    monkeypatch.setattr(codex_server.worktree, "prune_stale", noop)
    monkeypatch.setattr(codex_server, "CODEX_NOTIFY", False)


def _serve(monkeypatch, sess, on_submit=None):
    async def get_or_create(name):
        return sess

    async def get(name):
        return sess

    async def submit(prompt, baseline_starts):
        if on_submit is not None:
            on_submit()

    async def alive():
        return True

    monkeypatch.setattr(codex_server.manager, "get_or_create", get_or_create)
    monkeypatch.setattr(codex_server.manager, "get", get)
    monkeypatch.setattr(sess, "_submit_confirmed", submit)
    monkeypatch.setattr(sess, "is_alive", alive)
    return TestClient(codex_server.app)


class TestEndpoints:
    def test_chat_answers_429_with_the_reset_and_keeps_the_session(self, app_env, monkeypatch):
        state = {"events": GOOD_TURN}
        sess = _bound_session(monkeypatch, lambda: state["events"])
        stopped = []

        async def stop():
            stopped.append(True)

        monkeypatch.setattr(sess, "stop", stop)
        with _serve(monkeypatch, sess,
                    on_submit=lambda: state.update(events=GOOD_TURN + QUOTA_TURN)) as client:
            r = client.post("/chat/unit@dev", json={"prompt": "more"})
        assert r.status_code == 429, r.text
        d = r.json()["detail"]
        assert d["error"] == "usage_limit"
        assert d["message"] == QUOTA_MSG
        assert d["resets_at"] and datetime.fromisoformat(d["resets_at"])
        assert d["model"] == PIN and d["pin"] == PIN
        assert d["session"] == "unit@dev"
        assert stopped == [] and sess._needs_repin is True  # alive, re-pin pending

    def test_chat_answers_409_on_model_drift(self, app_env, monkeypatch):
        state = {"events": GOOD_TURN}
        drifted = GOOD_TURN + [_start("t2"), _context(FALLBACK, "medium"),
                               _complete("t2", last="wrong-model answer")]
        sess = _bound_session(monkeypatch, lambda: state["events"])
        with _serve(monkeypatch, sess, on_submit=lambda: state.update(events=drifted)) as client:
            r = client.post("/chat/unit@dev", json={"prompt": "more"})
        assert r.status_code == 409, r.text
        d = r.json()["detail"]
        assert d["error"] == "model_drift"
        assert d["model"] == FALLBACK and d["pin"] == PIN
        assert "wrong-model answer" not in r.text
        assert sess._needs_repin is True

    def test_chat_answers_502_on_another_terminal_error(self, app_env, monkeypatch):
        state = {"events": GOOD_TURN}
        errored = GOOD_TURN + [_start("t2"), _context(PIN),
                               _complete("t2", last=None, error={"message": "stream disconnected"})]
        sess = _bound_session(monkeypatch, lambda: state["events"])
        with _serve(monkeypatch, sess, on_submit=lambda: state.update(events=errored)) as client:
            r = client.post("/chat/unit@dev", json={"prompt": "more"})
        assert r.status_code == 502, r.text
        assert r.json()["detail"]["error"] == "error"
        assert r.json()["detail"]["message"] == "stream disconnected"
        assert sess._needs_repin is False

    def test_chat_clean_turn_still_200(self, app_env, monkeypatch):
        # Baselines come from the file at submit (turn 1 done); the answer
        # lands after the paste, on the pinned model -> the usual 200.
        state = {"events": GOOD_TURN}
        sess = _bound_session(monkeypatch, lambda: state["events"])
        answered = GOOD_TURN + [_start("t2"), _context(PIN), _complete("t2", last="answer two")]
        with _serve(monkeypatch, sess, on_submit=lambda: state.update(events=answered)) as client:
            r = client.post("/chat/unit@dev", json={"prompt": "more"})
        assert r.status_code == 200, r.text
        assert r.json()["response"] == "answer two"
        assert sess._needs_repin is False

    def test_last_reports_usage_limit_done_with_empty_response(self, app_env, monkeypatch):
        sess = _bound_session(monkeypatch, lambda: GOOD_TURN + QUOTA_TURN)
        sess._turn_count = 2
        with _serve(monkeypatch, sess) as client:
            r = client.get("/last/unit@dev", params={"wait": 0})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["done"] is True
        assert body["status"] == "usage_limit"
        assert body["response"] == ""
        assert body["error"] == QUOTA_MSG
        assert body["resets_at"] and datetime.fromisoformat(body["resets_at"])
        assert body["turn"] == 2 and body["session"] == "unit@dev"

    def test_last_reports_model_drift(self, app_env, monkeypatch):
        events = GOOD_TURN + [_start("t2"), _context(FALLBACK, "medium"),
                              _complete("t2", last="wrong-model answer")]
        sess = _bound_session(monkeypatch, lambda: events)
        with _serve(monkeypatch, sess) as client:
            r = client.get("/last/unit@dev", params={"wait": 0})
        body = r.json()
        assert (body["done"], body["status"], body["response"]) == (True, "model_drift", "")
        assert FALLBACK in body["error"] and body["resets_at"] is None

    def test_last_clean_turn_keeps_the_old_shape(self, app_env, monkeypatch):
        sess = _bound_session(monkeypatch, lambda: GOOD_TURN)
        sess._last_baseline_completes = 0
        with _serve(monkeypatch, sess) as client:
            r = client.get("/last/unit@dev", params={"wait": 0})
        body = r.json()
        assert (body["done"], body["status"], body["response"]) == (True, "done", "answer one")
        assert body["error"] is None and body["resets_at"] is None

    def test_last_pending_has_null_extras(self, app_env, monkeypatch):
        sess = _bound_session(monkeypatch, lambda: GOOD_TURN + [_start("t2")])
        with _serve(monkeypatch, sess) as client:
            r = client.get("/last/unit@dev", params={"wait": 0})
        body = r.json()
        assert (body["done"], body["status"]) == (False, "pending")
        assert body["error"] is None and body["resets_at"] is None


def test_last_response_extras_default_to_null():
    r = codex_server.LastResponse(done=True, response="x", turn=1, session="s", elapsed_ms=0)
    assert r.error is None and r.resets_at is None


# ===========================================================================
# Capacity ("server_overloaded"): classified, retried with a backoff on the
# same session, then 503 — never for the usage limit or a drift
# ===========================================================================

# A REAL capacity task_complete from a container rollout (2026-09-06 12:03:52,
# TestChatMemory turn 2): codex gave the turn up in 2.9s and did not retry.
CAPACITY_LINE = (
    '{"timestamp":"2026-09-06T12:03:52.263Z","ordinal":21,"type":"event_msg","payload":'
    '{"type":"task_complete","turn_id":"01a0769a-7615-7541-8792-b4f1245d5e3b",'
    '"last_agent_message":null,"error":{"message":"Selected model is at capacity. '
    'Please try a different model.","codex_error_info":"server_overloaded"},'
    '"started_at":1788696229,"completed_at":1788696232,"duration_ms":2854}}'
)
CAPACITY_EVENT = json.loads(CAPACITY_LINE)
CAPACITY_MSG = CAPACITY_EVENT["payload"]["error"]["message"]
CAPACITY_TURN = [_start("t2"), _context(PIN), CAPACITY_EVENT]


class TestOverloadVerdict:
    def test_real_capacity_line_is_a_retryable_error(self):
        v = codex_server._turn_verdict(GOOD_TURN + CAPACITY_TURN, baseline_completes=1, pin=PIN)
        assert v["status"] == "error"
        assert v["error"] == CAPACITY_MSG
        assert v["error_info"] == "server_overloaded"
        assert v["model"] == PIN
        assert codex_server._is_overloaded(v) is True

    def test_message_fallback_when_error_info_missing(self):
        assert codex_server._is_overloaded(
            {"status": "error", "error": "The model is overloaded right now", "error_info": None}) is True
        assert codex_server._is_overloaded(
            {"status": "error", "error": "Selected model is at capacity."}) is True
        assert codex_server._is_overloaded(
            {"status": "error", "error": "stream disconnected", "error_info": "stream_error"}) is False

    def test_structured_error_info_variants_do_not_crash(self):
        # codex_error_info is a tagged enum: struct variants serialize as an
        # OBJECT (e.g. http_connection_failed); that must classify, not raise.
        v = codex_server._turn_verdict(
            GOOD_TURN + [_start("t2"), _context(PIN), _complete("t2", last=None, error={
                "message": "connection failed",
                "codex_error_info": {"http_connection_failed": {"http_status_code": 502}}})],
            1, PIN)
        assert v["status"] == "error"
        assert codex_server._is_overloaded(v) is False
        assert codex_server._is_overloaded(
            {"status": "error", "error": "at capacity", "error_info": {"x": 1}}) is True

    def test_usage_limit_and_drift_are_never_retryable(self):
        quota = codex_server._turn_verdict(GOOD_TURN + QUOTA_TURN, 1, PIN)
        assert quota["status"] == "usage_limit"
        assert codex_server._is_overloaded(quota) is False
        # Even wording that mentions capacity does not make a quota/drift retryable.
        assert codex_server._is_overloaded(
            {"status": "usage_limit", "error": "at capacity", "error_info": "server_overloaded"}) is False
        assert codex_server._is_overloaded(
            {"status": "model_drift", "error": "overloaded"}) is False
        assert codex_server._is_overloaded({"status": "done", "error": None}) is False
        assert codex_server._is_overloaded(None) is False


@pytest.fixture()
def overload_env(monkeypatch):
    """Retry knobs pinned, and asyncio.sleep recorded instead of slept — but
    the module's monotonic clock advances by the slept amount, so the budget
    arithmetic is exercised as if the backoff had really elapsed."""
    import types
    import time as real_time
    monkeypatch.setattr(codex_server, "CODEX_MODEL", PIN)
    monkeypatch.setattr(codex_server, "CODEX_OVERLOAD_RETRIES", 2)
    monkeypatch.setattr(codex_server, "CODEX_OVERLOAD_BACKOFF", 5.0)
    monkeypatch.setattr(codex_server, "CODEX_OVERLOAD_MIN_BUDGET", 45.0)
    monkeypatch.setattr(codex_server, "RESPONSE_HARD_TIMEOUT", 180.0)
    slept: list[float] = []
    clock = types.SimpleNamespace(**{k: getattr(real_time, k) for k in dir(real_time)
                                     if not k.startswith("_")})
    clock.monotonic = lambda: real_time.monotonic() + sum(slept)
    monkeypatch.setattr(codex_server, "time", clock)

    async def fake_sleep(secs):
        slept.append(secs)

    monkeypatch.setattr(codex_server.asyncio, "sleep", fake_sleep)
    return slept


def _overload_session(monkeypatch, verdicts: list[dict]):
    """A bound session whose _submit_confirmed / _collect_response are mocked:
    each submit starts a turn in the (fake) rollout, each collect pops the next
    verdict and completes that turn — so send()'s baseline re-reads move
    exactly as they would against the real file."""
    state = {"events": list(GOOD_TURN)}
    sess = _bound_session(monkeypatch, lambda: state["events"])
    submits: list[tuple[str, int]] = []
    collects: list[tuple[int, int, float | None]] = []

    async def submit(prompt, baseline_starts):
        submits.append((prompt, baseline_starts))
        state["events"] = state["events"] + [_start(f"t{len(submits) + 1}"), _context(PIN)]

    async def collect(baseline_completes, baseline_starts, hard_timeout=None):
        collects.append((baseline_completes, baseline_starts, hard_timeout))
        v = verdicts.pop(0)
        sess._last_verdict = v
        done = _complete(f"t{len(submits) + 1}", last="answer two") if v["status"] == "done" \
            else CAPACITY_EVENT if v["status"] == "error" else QUOTA_EVENT
        state["events"] = state["events"] + [done]
        return "answer two" if v["status"] == "done" else ""

    async def alive():
        return True

    monkeypatch.setattr(sess, "_submit_confirmed", submit)
    monkeypatch.setattr(sess, "_collect_response", collect)
    monkeypatch.setattr(sess, "is_alive", alive)
    return sess, submits, collects


OVERLOADED = {"status": "error", "error": CAPACITY_MSG, "resets_at": None,
              "model": PIN, "error_info": "server_overloaded"}
DONE = {"status": "done", "error": None, "resets_at": None, "model": PIN}
QUOTA = {"status": "usage_limit", "error": QUOTA_MSG, "resets_at": None, "model": PIN}


class TestOverloadRetry:
    def test_retries_once_and_returns_the_second_attempts_answer(self, overload_env, monkeypatch):
        sess, submits, collects = _overload_session(monkeypatch, [dict(OVERLOADED), dict(DONE)])
        assert _run(sess.send("what did I say my name was?")) == "answer two"
        # The SAME prompt, re-submitted verbatim as a new turn on the same session.
        assert [p for p, _ in submits] == ["what did I say my name was?"] * 2
        # Baselines were re-read from the rollout after the failed turn (1 -> 2).
        assert [b for _, b in submits] == [1, 2]
        assert [(c, s) for c, s, _ in collects] == [(1, 1), (2, 2)]
        assert sess._last_baseline_completes == 2 and sess._last_baseline_starts == 2
        # One backoff of CODEX_OVERLOAD_BACKOFF before the retry.
        assert overload_env == [5.0]
        # The retry's own collection is capped at what is left of the budget
        # AFTER the backoff (5s) and the re-submit have been spent.
        assert collects[0][2] is None
        assert 170.0 <= collects[1][2] <= 175.0  # 180 - 5 - elapsed(~0)
        assert sess.turn_count == 2  # one turn from the caller's point of view

    def test_backoff_doubles_and_gives_up_after_the_retries(self, overload_env, monkeypatch):
        sess, submits, collects = _overload_session(
            monkeypatch, [dict(OVERLOADED), dict(OVERLOADED), dict(OVERLOADED)])
        with pytest.raises(codex_server.TurnFailure) as exc:
            _run(sess.send("more"))
        assert len(submits) == 3 and len(collects) == 3  # 1 + CODEX_OVERLOAD_RETRIES
        assert overload_env == [5.0, 10.0]
        assert exc.value.overloaded is True and exc.value.attempts == 3
        assert exc.value.verdict["status"] == "error"  # /last keeps "error"
        assert exc.value.verdict["error"] == CAPACITY_MSG
        # Each retry's cap shrinks by every backoff spent so far (5, then 5+10).
        assert 170.0 <= collects[1][2] <= 175.0
        assert 160.0 <= collects[2][2] <= 165.0

    def test_usage_limit_is_never_retried(self, overload_env, monkeypatch):
        sess, submits, collects = _overload_session(monkeypatch, [dict(QUOTA), dict(DONE)])
        with pytest.raises(codex_server.TurnFailure) as exc:
            _run(sess.send("more"))
        assert exc.value.verdict["status"] == "usage_limit"
        assert exc.value.overloaded is False and exc.value.attempts == 1
        assert len(submits) == 1 and overload_env == []

    def test_other_terminal_errors_are_not_retried(self, overload_env, monkeypatch):
        broken = {"status": "error", "error": "stream disconnected", "resets_at": None,
                  "model": PIN, "error_info": "stream_error"}
        sess, submits, _ = _overload_session(monkeypatch, [broken, dict(DONE)])
        with pytest.raises(codex_server.TurnFailure) as exc:
            _run(sess.send("more"))
        assert exc.value.overloaded is False and len(submits) == 1 and overload_env == []

    def test_no_retry_when_the_remaining_budget_is_below_the_minimum(self, overload_env, monkeypatch):
        # 40s hard timeout - 5s backoff = 35s < CODEX_OVERLOAD_MIN_BUDGET (45s):
        # the retry could not finish inside the call's budget, so fail now.
        monkeypatch.setattr(codex_server, "RESPONSE_HARD_TIMEOUT", 40.0)
        sess, submits, _ = _overload_session(monkeypatch, [dict(OVERLOADED), dict(DONE)])
        with pytest.raises(codex_server.TurnFailure) as exc:
            _run(sess.send("more"))
        assert exc.value.overloaded is True and exc.value.attempts == 1
        assert len(submits) == 1 and overload_env == []

    def test_retries_are_configurable(self, overload_env, monkeypatch):
        monkeypatch.setattr(codex_server, "CODEX_OVERLOAD_RETRIES", 0)
        sess, submits, _ = _overload_session(monkeypatch, [dict(OVERLOADED), dict(DONE)])
        with pytest.raises(codex_server.TurnFailure) as exc:
            _run(sess.send("more"))
        assert exc.value.attempts == 1 and len(submits) == 1 and overload_env == []


class TestOverloadEndpoints:
    def test_chat_answers_503_after_the_retries_are_exhausted(self, app_env, overload_env, monkeypatch):
        # Real _collect_response against a fake rollout: every submit lands
        # another capacity turn, so the bridge retries once (CODEX_OVERLOAD_
        # RETRIES=1) and then answers 503 with the structured body.
        monkeypatch.setattr(codex_server, "CODEX_OVERLOAD_RETRIES", 1)
        state = {"events": list(GOOD_TURN)}
        sess = _bound_session(monkeypatch, lambda: state["events"])

        def another_capacity_turn():
            n = len(state["events"])
            state["events"] = state["events"] + [_start(f"t{n}"), _context(PIN), CAPACITY_EVENT]

        with _serve(monkeypatch, sess, on_submit=another_capacity_turn) as client:
            r = client.post("/chat/unit@dev", json={"prompt": "more"})
        assert r.status_code == 503, r.text
        d = r.json()["detail"]
        assert d == {"error": "server_overloaded", "message": CAPACITY_MSG, "attempts": 2,
                     "session": "unit@dev", "model": PIN, "pin": PIN}
        assert [s for s in overload_env if s >= 1] == [5.0]  # one backoff, then 503
        assert sess._needs_repin is False  # not a quota event: no re-pin

    def test_chat_answers_503_at_once_when_no_budget_is_left(self, app_env, overload_env, monkeypatch):
        monkeypatch.setattr(codex_server, "RESPONSE_HARD_TIMEOUT", 30.0)
        state = {"events": list(GOOD_TURN)}
        sess = _bound_session(monkeypatch, lambda: state["events"])
        with _serve(monkeypatch, sess,
                    on_submit=lambda: state.update(events=GOOD_TURN + CAPACITY_TURN)) as client:
            r = client.post("/chat/unit@dev", json={"prompt": "more"})
        assert r.status_code == 503, r.text
        assert r.json()["detail"]["attempts"] == 1
        assert [s for s in overload_env if s >= 1] == []

    def test_other_terminal_errors_keep_502(self, app_env, overload_env, monkeypatch):
        state = {"events": list(GOOD_TURN)}
        errored = GOOD_TURN + [_start("t2"), _context(PIN),
                               _complete("t2", last=None, error={"message": "stream disconnected"})]
        sess = _bound_session(monkeypatch, lambda: state["events"])
        with _serve(monkeypatch, sess, on_submit=lambda: state.update(events=errored)) as client:
            r = client.post("/chat/unit@dev", json={"prompt": "more"})
        assert r.status_code == 502, r.text
        assert r.json()["detail"]["error"] == "error"

    def test_last_reports_the_capacity_turn_as_error(self, app_env, monkeypatch):
        sess = _bound_session(monkeypatch, lambda: GOOD_TURN + CAPACITY_TURN)
        sess._turn_count = 2
        with _serve(monkeypatch, sess) as client:
            r = client.get("/last/unit@dev", params={"wait": 0})
        body = r.json()
        assert (body["done"], body["status"], body["response"]) == (True, "error", "")
        assert body["error"] == CAPACITY_MSG and body["resets_at"] is None
