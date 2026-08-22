"""
Unit tests for AgySession._submit_confirmed / _await_ingest.

These pin the behavior that actually bit the review harness: agy sometimes
swallows the submit Enter (the pasted prompt sits unsubmitted while the screen
looks idle), so the bridge polled an unchanging transcript until the 140s max
timeout and logged "0 message(s)" — even though agy answered instantly the
moment the input finally registered.

agy can also swallow the PASTE ITSELF (a prompt sent while it is still
rendering a long previous answer is consumed whole): the input box comes back
EMPTY, the transcript never moves, and re-pressing Enter has nothing to send —
only re-pasting the prompt recovers that turn.

The safety property under test is NO DUPLICATE MESSAGES: we only re-press Enter
when agy is idle AND has ingested nothing for a full confirm window, and we only
re-paste when the input box is on top of that demonstrably EMPTY — re-checked in
the instant before the paste. If agy took the input (a new transcript step, or a
busy screen) we must never re-send.

No server / agy / Docker — everything agy-facing is mocked.
Run:  ./.venv/bin/python -m pytest test_submit_confirm.py -v
"""

import asyncio
import time

import pytest

import server


# How agy renders its input box: two horizontal rules with the caret line
# between them — bare ">" when empty, "> ..." when text is waiting.
RULE = "─" * 60
EMPTY_BOX = f"⢿ Prioritizing Tool Usage...\n{RULE}\n>\n{RULE}\n? for shortcuts"
FULL_BOX = f"{RULE}\n> please review the diff I just sent\n{RULE}\n? for shortcuts"


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def fast(monkeypatch):
    """Shrink the timing constants so windows elapse in fractions of a second."""
    monkeypatch.setattr(server, "SUBMIT_CONFIRM_WAIT", 0.25)
    monkeypatch.setattr(server, "RESPONSE_POLL_INTERVAL", 0.02)
    monkeypatch.setattr(server, "SUBMIT_REPASTE_DELAY", 0.0)


@pytest.fixture()
def enters(monkeypatch):
    """Replace _tmux with a recorder; returns the list of Enter keystrokes."""
    sent = []

    async def fake_tmux(*args, **kwargs):
        if args and args[0] == "send-keys" and args[-1] == "Enter":
            sent.append(args)
        return (0, "")

    monkeypatch.setattr(server, "_tmux", fake_tmux)
    return sent


@pytest.fixture()
def tmux(monkeypatch):
    """Recorder for BOTH corrective actions: .enters and .pastes.

    A paste too many is a duplicated turn, an Enter too many is free — so the
    re-paste tests assert on both counts, not just the keystrokes.
    """
    class _Rec:
        def __init__(self):
            self.enters: list = []
            self.pastes: list = []

    rec = _Rec()

    async def fake_tmux(*args, **kwargs):
        if args and args[0] == "send-keys" and args[-1] == "Enter":
            rec.enters.append(args)
        elif args and args[0] == "paste-buffer":
            rec.pastes.append(args)
        return (0, "")

    monkeypatch.setattr(server, "_tmux", fake_tmux)
    return rec


def _session(monkeypatch, *, screen):
    sess = server.AgySession(name="unit")
    sess._conversation_id = "conv"

    async def capture():
        return screen() if callable(screen) else screen

    monkeypatch.setattr(sess, "_capture", capture)
    return sess


# --- the input was accepted: never re-send -> no duplicate ------------------

def test_accepted_via_new_step_does_not_resend(fast, enters, monkeypatch):
    # transcript already shows a step beyond baseline on the first read
    monkeypatch.setattr(server, "_read_transcript", lambda cid: [{"step_index": 6}])
    sess = _session(monkeypatch, screen="all idle  ? for shortcuts")
    _run(sess._submit_confirmed("hi", baseline=5))
    assert len(enters) == 1, "accepted submit must not be re-sent (would duplicate)"


def test_busy_screen_counts_as_accepted_does_not_resend(fast, enters, monkeypatch):
    # no new transcript step, but agy is visibly generating -> it took the input
    monkeypatch.setattr(server, "_read_transcript", lambda cid: [{"step_index": 5}])
    sess = _session(monkeypatch, screen="Generating...  esc to cancel")
    _run(sess._submit_confirmed("hi", baseline=5))
    assert len(enters) == 1, "busy means agy accepted the turn; re-sending would duplicate"


# --- the Enter was dropped: re-press exactly once ---------------------------

def test_dropped_submit_is_resent_once_then_ingested(fast, enters, monkeypatch):
    # the transcript only advances once a *second* Enter (the resend) lands —
    # exactly how a dropped submit behaves: nothing is logged until it's resent
    def transcript(cid):
        return [{"step_index": 6 if len(enters) >= 2 else 5}]

    monkeypatch.setattr(server, "_read_transcript", transcript)
    sess = _session(monkeypatch, screen="idle, no markers  ? for shortcuts")
    _run(sess._submit_confirmed("hi", baseline=5))
    assert len(enters) == 2, "exactly one resend: the original Enter plus one retry"


def test_gives_up_after_max_retries_without_spamming(fast, enters, monkeypatch):
    # agy never ingests (transcript frozen, screen idle): bounded resends only
    monkeypatch.setattr(server, "_read_transcript", lambda cid: [{"step_index": 5}])
    sess = _session(monkeypatch, screen="idle  ? for shortcuts")
    _run(sess._submit_confirmed("hi", baseline=5))
    # 1 original + SUBMIT_MAX_RETRIES resends, then it stops (no infinite loop)
    assert len(enters) == 1 + server.SUBMIT_MAX_RETRIES


# --- the PASTE was consumed: re-paste, never just Enter ---------------------

def test_empty_box_is_repasted_once_then_ingested(fast, tmux, monkeypatch):
    # The input box is empty and the transcript frozen: an Enter re-press would
    # send nothing, so the whole prompt goes back in. It lands on the re-paste.
    def transcript(cid):
        return [{"step_index": 6 if len(tmux.pastes) >= 2 else 5}]

    monkeypatch.setattr(server, "_read_transcript", transcript)
    sess = _session(monkeypatch, screen=EMPTY_BOX)
    _run(sess._submit_confirmed("hi", baseline=5))
    assert len(tmux.pastes) == 2, "exactly one re-paste: the original plus one retry"
    assert len(tmux.enters) == 2, "each paste carries its own Enter; no extra re-press"


def test_empty_box_repastes_are_bounded(fast, tmux, monkeypatch):
    # agy never takes it: bounded re-pastes, then the (useless but harmless)
    # Enter retries, then it stops — no infinite loop, no paste storm.
    monkeypatch.setattr(server, "_read_transcript", lambda cid: [{"step_index": 5}])
    sess = _session(monkeypatch, screen=EMPTY_BOX)
    _run(sess._submit_confirmed("hi", baseline=5))
    assert len(tmux.pastes) == 1 + server.SUBMIT_REPASTE_MAX
    assert len(tmux.enters) == len(tmux.pastes) + server.SUBMIT_MAX_RETRIES


def test_repaste_is_aborted_when_the_turn_shows_up_first(tmux, monkeypatch):
    """The duplicate guard: agy logs the turn during the settle delay, so the
    re-paste decision is stale by the time it would fire and must be dropped."""
    monkeypatch.setattr(server, "SUBMIT_CONFIRM_WAIT", 0.25)
    monkeypatch.setattr(server, "RESPONSE_POLL_INTERVAL", 0.02)
    monkeypatch.setattr(server, "SUBMIT_REPASTE_DELAY", 0.5)
    t0 = time.monotonic()
    monkeypatch.setattr(
        server, "_read_transcript",
        lambda cid: [{"step_index": 6 if time.monotonic() - t0 > 0.4 else 5}],
    )
    sess = _session(monkeypatch, screen=EMPTY_BOX)
    _run(sess._submit_confirmed("hi", baseline=5))
    assert len(tmux.pastes) == 1, "the turn was ingested after all — re-pasting duplicates it"


def test_busy_screen_aborts_the_repaste(fast, tmux, monkeypatch):
    # Empty box (agy already consumed the text into a turn) but the screen goes
    # busy during the settle delay -> agy IS working; the paste must not fire.
    monkeypatch.setattr(server, "SUBMIT_REPASTE_DELAY", 0.2)
    monkeypatch.setattr(server, "_read_transcript", lambda cid: [{"step_index": 5}])
    t0 = time.monotonic()
    sess = _session(
        monkeypatch,
        screen=lambda: (EMPTY_BOX + "\nGenerating...  esc to cancel"
                        if time.monotonic() - t0 > 0.3 else EMPTY_BOX),
    )
    _run(sess._submit_confirmed("hi", baseline=5))
    assert len(tmux.pastes) == 1


def test_text_still_in_the_box_is_never_repasted(fast, tmux, monkeypatch):
    """The pre-existing invariant: with the prompt visibly waiting in the box a
    dropped Enter is the only possible fault, so Enter is the only correction."""
    monkeypatch.setattr(server, "_read_transcript", lambda cid: [{"step_index": 5}])
    sess = _session(monkeypatch, screen=FULL_BOX)
    _run(sess._submit_confirmed("hi", baseline=5))
    assert len(tmux.pastes) == 1, "re-pasting a prompt that is still in the box duplicates it"
    assert len(tmux.enters) == 1 + server.SUBMIT_MAX_RETRIES


def test_repaste_max_zero_restores_the_enter_only_behavior(fast, tmux, monkeypatch):
    monkeypatch.setattr(server, "SUBMIT_REPASTE_MAX", 0)
    monkeypatch.setattr(server, "_read_transcript", lambda cid: [{"step_index": 5}])
    sess = _session(monkeypatch, screen=EMPTY_BOX)
    _run(sess._submit_confirmed("hi", baseline=5))
    assert len(tmux.pastes) == 1
    assert len(tmux.enters) == 1 + server.SUBMIT_MAX_RETRIES


# --- reading the input box off the rendered screen --------------------------

class TestInputBoxEmpty:
    """Conservative by contract: True only when the box is located AND blank,
    because a True becomes a re-paste while a False becomes a harmless Enter."""

    def test_bare_caret_between_rules_is_empty(self):
        assert server.AgySession._input_box_empty(EMPTY_BOX) is True

    def test_text_in_the_box_is_not_empty(self):
        assert server.AgySession._input_box_empty(FULL_BOX) is False

    def test_history_echo_of_a_prompt_does_not_answer_for_the_box(self):
        # agy echoes submitted prompts with the same "> " prefix ABOVE the box;
        # only the caret line inside the last rule pair counts.
        screen = f"> an earlier prompt\n  its wrapped tail\n{RULE}\n>\n{RULE}\n? for shortcuts"
        assert server.AgySession._input_box_empty(screen) is True

    def test_wrapped_paste_inside_the_box_is_not_empty(self):
        screen = f"{RULE}\n> first line of the paste\n  and its continuation\n{RULE}\n? for shortcuts"
        assert server.AgySession._input_box_empty(screen) is False

    def test_no_box_on_screen_is_not_empty(self):
        # A survey/interstitial drawn over the input box: unreadable, so the
        # answer must be the safe one.
        screen = "How's the CLI experience so far?\n [1] Good  [2] Fine\n? for shortcuts"
        assert server.AgySession._input_box_empty(screen) is False
