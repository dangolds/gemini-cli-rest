"""
Unit tests for the agy end-of-turn bell fast path.

agy (settings "notifications": true) rings the terminal BEL exactly once per
turn, only AFTER the final answer is flushed to the transcript; a tmux
alert-bell hook records each ring as one line in a per-session file under
BELL_DIR. These pin the fast path's contract:

  * a recorded bell PLUS a new DONE answer past the submit baseline, confirmed
    by a single not-busy glance at the screen, completes the turn immediately —
    no idle-marker match, no debounce;
  * the bell NEVER completes a turn by itself (the answer must be durable),
    and never against a busy screen (tmux records ANY pane bell, so a stray
    mid-turn ring plus an intermediate DONE step must not end the turn early);
  * a stray ring credits progress ONCE (edge-triggered), so it can never
    disable the stall detector for the rest of the turn;
  * bell baselines are per-turn, so a previous turn's ring can't complete the
    next turn;
  * with AGY_BELL off, detection is the legacy polling path exactly.

No server / agy / Docker — transcript and screen are mocked; the bell file is
real (tmp_path) unless a test needs a time-phased count.
Run:  ./.venv/bin/python -m pytest test_bell.py -v
"""

import asyncio
import json
import time

import pytest

import server


def _run(coro):
    return asyncio.run(coro)


def _answer(idx, content="the answer"):
    return {
        "step_index": idx, "source": "MODEL", "type": "PLANNER_RESPONSE",
        "status": "DONE", "content": content, "created_at": "t",
    }


@pytest.fixture()
def bellenv(monkeypatch, tmp_path):
    """Bell fast path armed, all timing constants shrunk for the tests."""
    monkeypatch.setattr(server, "AGY_BELL", True)
    monkeypatch.setattr(server, "BELL_DIR", tmp_path / "bells")
    (tmp_path / "bells").mkdir()
    monkeypatch.setattr(server, "RESPONSE_POLL_INTERVAL", 0.05)
    monkeypatch.setattr(server, "RESPONSE_FAST_POLL", 0.01)
    monkeypatch.setattr(server, "RESPONSE_MIN_WAIT", 0.0)
    monkeypatch.setattr(server, "RESPONSE_STALL_TIMEOUT", 5.0)
    monkeypatch.setattr(server, "RESPONSE_HARD_TIMEOUT", 10.0)
    monkeypatch.setattr(server, "TIMEOUT_LOG_DIR", tmp_path / "timeouts")


def _session(monkeypatch, *, transcript, screen, alive=True):
    """Build a session whose transcript/screen are driven by callables."""
    sess = server.AgySession(name="unit")
    sess._conversation_id = "conv"
    sess._turn_count = 1
    monkeypatch.setattr(server, "_read_transcript", lambda cid: transcript())
    monkeypatch.setattr(sess, "_transcript_mtime", lambda: 0.0)  # isolate: progress via busy/steps only

    async def capture():
        return screen() if callable(screen) else screen

    async def is_alive():
        return alive

    monkeypatch.setattr(sess, "_capture", capture)
    monkeypatch.setattr(sess, "is_alive", is_alive)
    return sess


def _ring(sess, n=1):
    """Record n rings the way the tmux hook does (one timestamp line each)."""
    with open(sess.bell_file, "ab") as f:
        f.write(b"1751600000.000000000\n" * n)


BUSY = "Generating...  esc to cancel"
# Not busy, but no READY_MARKER either (a long answer scrolled the status bar
# away): only the bell path may complete against this screen.
QUIET_NO_MARKER = "tail of a long answer, status bar scrolled away"


# --- (a) bell + durable answer completes with no idle marker ----------------

def test_bell_completes_without_idle_marker(bellenv, monkeypatch):
    # The READY_MARKER never shows, so the legacy path can't finish this turn:
    # completion can only come from bell + answer-past-baseline, confirmed by
    # the single not-busy glance — no idle-marker match, no debounce.
    sess = _session(
        monkeypatch,
        transcript=lambda: [_answer(0, "belled answer")],
        screen=QUIET_NO_MARKER,
    )
    _ring(sess)
    t0 = time.monotonic()
    assert _run(sess._collect_response(baseline=-1)) == "belled answer"
    assert sess._last_exit_reason == "bell"
    # well under the 2-full-poll debounce the legacy path would have needed
    assert time.monotonic() - t0 < 2 * server.RESPONSE_POLL_INTERVAL


# --- (b) a bell alone never completes; legacy path still finishes the turn --

def test_stray_bell_without_answer_does_not_complete(bellenv, monkeypatch):
    # Phase 1 (<0.15s): a stray ring but NO answer past the baseline — the
    # bell alone must not end the turn. The ring is gone by the time the
    # answer lands (0.3s), so the turn must finish via the legacy idle-screen
    # path, proving the fallback still runs underneath the fast wakeups.
    t0 = time.monotonic()
    sess = _session(
        monkeypatch,
        transcript=lambda: ([_answer(0, "real answer")]
                            if time.monotonic() - t0 > 0.3 else []),
        screen=lambda: ("? for shortcuts" if time.monotonic() - t0 > 0.3 else BUSY),
    )
    monkeypatch.setattr(
        sess, "_bell_count", lambda: 1 if time.monotonic() - t0 < 0.15 else 0
    )
    assert _run(sess._collect_response(baseline=-1)) == "real answer"
    assert sess._last_exit_reason == "transcript_done"


# --- (b') the busy-glance: a mid-turn stray ring must not end the turn ------

def test_bell_with_answer_but_busy_screen_does_not_complete(bellenv, monkeypatch):
    # A stray BEL (tool output bytes) landed while agy is mid-turn, and an
    # intermediate DONE step is already in the transcript. Without the
    # busy-glance the bell path would return that partial; instead the turn
    # must keep running until the hard cap ends this perpetually-busy screen.
    monkeypatch.setattr(server, "RESPONSE_HARD_TIMEOUT", 0.4)
    sess = _session(
        monkeypatch,
        transcript=lambda: [_answer(0, "intermediate step")],
        screen=BUSY,
    )
    _ring(sess)
    _run(sess._collect_response(baseline=-1))
    assert sess._last_exit_reason == "hard_timeout"  # never "bell"


def test_bell_with_answer_and_not_busy_screen_completes(bellenv, monkeypatch):
    # Same setup with the busy marker gone: the single glance passes and the
    # bell path ends the turn (still no READY_MARKER on screen).
    sess = _session(
        monkeypatch,
        transcript=lambda: [_answer(0, "confirmed answer")],
        screen=QUIET_NO_MARKER,
    )
    _ring(sess)
    assert _run(sess._collect_response(baseline=-1)) == "confirmed answer"
    assert sess._last_exit_reason == "bell"


# --- (b'') stall detector survives a stray ring (edge-triggered progress) ---

def test_stray_bell_does_not_disable_stall_detector(bellenv, monkeypatch):
    # One stray ring past the baseline, but no answer EVER: the ring buys one
    # stall window, not immunity. Level-triggered progress crediting would
    # keep the turn alive to the hard cap; it must end at the stall timeout.
    monkeypatch.setattr(server, "RESPONSE_STALL_TIMEOUT", 0.3)
    monkeypatch.setattr(server, "RESPONSE_HARD_TIMEOUT", 5.0)
    sess = _session(
        monkeypatch,
        transcript=lambda: [],           # static, never an answer
        screen="idle  ? for shortcuts",  # idle the whole time
    )
    _ring(sess)
    t0 = time.monotonic()
    assert _run(sess._collect_response(baseline=-1)) == ""
    assert sess._last_exit_reason == "stalled"  # NOT "hard_timeout"
    assert time.monotonic() - t0 < 2.0  # gave up at the stall window, not the cap


# --- (c) kill-switch: bells on disk are ignored entirely --------------------

def test_disabled_completes_via_legacy_and_never_reads_bells(bellenv, monkeypatch):
    monkeypatch.setattr(server, "AGY_BELL", False)
    sess = _session(
        monkeypatch,
        transcript=lambda: [_answer(0, "legacy answer")],
        screen="all done  ? for shortcuts",
    )
    _ring(sess, n=3)  # rings exist on disk...

    def boom():
        raise AssertionError("bell file consulted while AGY_BELL is off")

    monkeypatch.setattr(sess, "_bell_count", boom)  # ...but must never be read
    assert _run(sess._collect_response(baseline=-1)) == "legacy answer"
    assert sess._last_exit_reason == "transcript_done"


# --- (d) bell baselines are per-turn ----------------------------------------

def test_turn1_bell_does_not_complete_turn2(bellenv, monkeypatch):
    # Turn 2's submit recorded baseline=1 (turn 1 rang once). The idle marker
    # never shows, so only a bell BEYOND that baseline may end the turn —
    # turn 1's ring must not, even though turn 2's answer is already in the
    # transcript. The second ring lands at 0.3s; completion must wait for it.
    t0 = time.monotonic()
    sess = _session(
        monkeypatch,
        transcript=lambda: [_answer(6, "turn two answer")],
        screen=QUIET_NO_MARKER,
    )
    sess._last_bell_baseline = 1  # what send() records at turn-2 submit
    monkeypatch.setattr(
        sess, "_bell_count", lambda: 1 if time.monotonic() - t0 < 0.3 else 2
    )
    assert _run(sess._collect_response(baseline=5)) == "turn two answer"
    assert sess._last_exit_reason == "bell"
    assert time.monotonic() - t0 >= 0.29  # waited for the beyond-baseline ring


# --- (d') send() captures the bell baseline at submit time -------------------

class TestSendCapturesBellBaseline:
    """Both send() branches must snapshot _bell_count() into
    _last_bell_baseline BEFORE submitting — it is what keeps a previous turn's
    ring from completing the new turn, and what /last measures against."""

    def _stub_common(self, monkeypatch, sess, seen):
        async def is_alive():
            return True

        async def collect(baseline):
            return "canned"

        monkeypatch.setattr(sess, "is_alive", is_alive)
        monkeypatch.setattr(sess, "_collect_response", collect)
        return seen

    def test_first_turn_branch(self, bellenv, monkeypatch):
        sess = server.AgySession(name="unit")
        _ring(sess, n=3)  # pre-existing rings at submit time
        seen = {}
        self._stub_common(monkeypatch, sess, seen)

        async def submit(prompt):
            # the baseline must already be published when the prompt goes in
            seen["baseline_at_submit"] = sess._last_bell_baseline

        async def detect(before):
            return "conv"

        monkeypatch.setattr(sess, "_submit", submit)
        monkeypatch.setattr(sess, "_detect_new_conversation", detect)
        monkeypatch.setattr(server, "_brain_dirs", lambda: {})
        assert _run(sess.send("hi")) == "canned"
        assert sess._last_bell_baseline == 3
        assert seen["baseline_at_submit"] == 3

    def test_follow_up_turn_branch(self, bellenv, monkeypatch):
        sess = server.AgySession(name="unit")
        sess._conversation_id = "conv"
        _ring(sess, n=2)
        seen = {}
        self._stub_common(monkeypatch, sess, seen)

        async def submit_confirmed(prompt, baseline):
            seen["baseline_at_submit"] = sess._last_bell_baseline

        monkeypatch.setattr(sess, "_submit_confirmed", submit_confirmed)
        monkeypatch.setattr(server, "_read_transcript", lambda cid: [_answer(4)])
        assert _run(sess.send("hi")) == "canned"
        assert sess._last_bell_baseline == 2
        assert seen["baseline_at_submit"] == 2
        assert sess._last_baseline_step == 4  # step baseline published alongside


# --- (e) /last: a recorded bell substitutes for the idle marker -------------

class TestLastWithBell:
    def test_bell_with_quiet_screen_is_done_without_idle_marker(self, bellenv, monkeypatch):
        # The bell is durable in the file, so /last accepts it in place of the
        # READY_MARKER — the status bar may have scrolled away entirely. The
        # single glance only has to confirm the pane isn't busy.
        sess = _session(
            monkeypatch,
            transcript=lambda: [_answer(0, "final")],
            screen=QUIET_NO_MARKER,
        )
        _ring(sess)
        assert _run(sess.last(0)) == (True, "final", 1)

    def test_busy_screen_with_bell_is_not_done(self, bellenv, monkeypatch):
        # A recorded ring plus a DONE step, but the pane is still generating:
        # the ring was stray/mid-turn, so /last must not hand out the partial.
        sess = _session(
            monkeypatch,
            transcript=lambda: [_answer(0, "partial")],
            screen=BUSY,
        )
        _ring(sess)
        assert _run(sess.last(0)) == (False, "", 1)

    def test_bell_at_baseline_does_not_substitute_idle(self, bellenv, monkeypatch):
        # Only turn 1's ring is on disk and the baseline already covers it:
        # no NEW end-of-turn evidence, no idle marker -> not done.
        sess = _session(
            monkeypatch,
            transcript=lambda: [_answer(6, "partial")],
            screen=QUIET_NO_MARKER,
        )
        sess._last_baseline_step = 5
        sess._last_bell_baseline = 1
        _ring(sess, n=1)
        assert _run(sess.last(0)) == (False, "", 1)

    def test_bell_without_answer_is_not_done(self, bellenv, monkeypatch):
        # A ring with nothing past the baseline in the transcript must not
        # produce an answer out of thin air (nor a previous turn's) — even
        # though the not-busy glance accepts the bell as evidence.
        sess = _session(monkeypatch, transcript=lambda: [], screen=QUIET_NO_MARKER)
        _ring(sess)
        assert _run(sess.last(0)) == (False, "", 1)

    def test_disabled_still_requires_idle_marker(self, bellenv, monkeypatch):
        monkeypatch.setattr(server, "AGY_BELL", False)
        sess = _session(
            monkeypatch,
            transcript=lambda: [_answer(0, "final")],
            screen=QUIET_NO_MARKER,  # not busy, but no READY_MARKER either
        )
        _ring(sess)
        assert _run(sess.last(0)) == (False, "", 1)

    def test_evidence_is_checked_before_the_answer_is_read(self, bellenv, monkeypatch):
        # The final flush + ring can land between /last's two looks. Reading
        # the transcript FIRST and validating it with evidence SECOND would
        # bless that stale partial read as done — the transcript here gains
        # its final step only once the evidence glance has happened, and the
        # returned answer must include it.
        state = {"evidence_checked": False}

        def screen():
            state["evidence_checked"] = True
            return QUIET_NO_MARKER

        def transcript():
            if state["evidence_checked"]:
                return [_answer(0, "step one"), _answer(1, "final step")]
            return [_answer(0, "step one")]

        sess = _session(monkeypatch, transcript=transcript, screen=screen)
        _ring(sess)
        assert _run(sess.last(0)) == (True, "step one\n\nfinal step", 1)


# --- (f) _bell_count file semantics ------------------------------------------

class TestBellCount:
    def test_missing_file_is_zero(self, bellenv):
        assert server.AgySession(name="unit")._bell_count() == 0

    def test_empty_file_is_zero(self, bellenv):
        sess = server.AgySession(name="unit")
        sess.bell_file.write_bytes(b"")
        assert sess._bell_count() == 0

    def test_counts_only_complete_lines(self, bellenv):
        # The third ring is still being written (no newline yet) — it must not
        # count until the hook's append completes.
        sess = server.AgySession(name="unit")
        sess.bell_file.write_bytes(b"1751600000.1\n1751600042.4\n1751600099.7")
        assert sess._bell_count() == 2


# --- (g) arming refuses an injection-prone BELL_DIR --------------------------

class TestBellArming:
    """BELL_DIR reaches the hook unquoted through tmux run-shell + sh: a space
    silently breaks the append (arming 'succeeds', no line ever written) and
    $()/backticks would execute on every ring — so arming must refuse any
    path outside the conservative charset and fall back to pure polling."""

    @pytest.fixture()
    def tmux_calls(self, monkeypatch):
        calls = []

        async def fake_tmux(*args, **kwargs):
            calls.append(args)
            return (0, "")

        async def fake_spawn(*argv, **kwargs):
            class Proc:
                async def wait(self):
                    return 0

            return Proc()

        monkeypatch.setattr(server, "_tmux", fake_tmux)
        # the daemon fork bypasses _tmux; stub it so no real tmux ever starts
        monkeypatch.setattr(server.asyncio, "create_subprocess_exec", fake_spawn)
        return calls

    def test_safe_dir_arms_the_hook(self, bellenv, tmux_calls):
        _run(server._ensure_tmux_server())
        assert any(a[0] == "set-hook" for a in tmux_calls)

    @pytest.mark.parametrize("bad", ["bells with space", "bells$(reboot)", "bells`x`"])
    def test_unsafe_dir_skips_arming_entirely(self, bellenv, tmux_calls, monkeypatch, tmp_path, bad):
        monkeypatch.setattr(server, "BELL_DIR", tmp_path / bad)
        _run(server._ensure_tmux_server())
        assert tmux_calls == []  # neither options nor hook were set


# --- lifecycle hygiene: a fresh/killed process carries no stale bells --------

def test_kill_unlinks_bell_file_and_resets_baseline(bellenv, monkeypatch):
    async def fake_tmux(*args, **kwargs):
        return (0, "")

    monkeypatch.setattr(server, "_tmux", fake_tmux)
    sess = server.AgySession(name="unit")
    _ring(sess, n=2)
    sess._last_bell_baseline = 2
    _run(sess._kill())
    assert not sess.bell_file.exists()
    assert sess._bell_count() == 0
    assert sess._last_bell_baseline == 0


# --- settings: lifespan ensures agy actually rings ---------------------------

class TestEnsureNotifications:
    def test_flips_disabled_to_enabled_preserving_other_keys(self, monkeypatch, tmp_path):
        monkeypatch.setattr(server, "AGY_STATE_DIR", tmp_path)
        path = tmp_path / "settings.json"
        path.write_text(json.dumps(
            {"toolPermission": "always-proceed", "notifications": False}
        ))
        server._ensure_agy_notifications()
        data = json.loads(path.read_text())
        assert data["notifications"] is True
        assert data["toolPermission"] == "always-proceed"
        assert not path.with_name(path.name + ".tmp").exists()  # atomic replace

    def test_missing_file_is_not_created(self, monkeypatch, tmp_path):
        # Seeding belongs to the entrypoint; a partial file written here would
        # suppress its heredoc seed.
        monkeypatch.setattr(server, "AGY_STATE_DIR", tmp_path)
        server._ensure_agy_notifications()
        assert not (tmp_path / "settings.json").exists()

    def test_unparseable_file_is_left_untouched(self, monkeypatch, tmp_path):
        monkeypatch.setattr(server, "AGY_STATE_DIR", tmp_path)
        path = tmp_path / "settings.json"
        path.write_text("{not json")
        server._ensure_agy_notifications()
        assert path.read_text() == "{not json"
