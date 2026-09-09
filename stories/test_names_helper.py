"""Group A (helper): the one name helper behaves as TestPRD Rule 2 requires."""
from __future__ import annotations

import re

import httpx
import pytest

from bridgetests import live, names

pytestmark = [pytest.mark.hermetic, pytest.mark.group("A")]

ROUTE = re.compile(r"^[A-Za-z0-9_-]+(@[A-Za-z0-9._/-]+)?$")


def test_key_has_name_stamp_and_base():
    k = names.key("chain")
    assert k == f"{names.REPO_PREFIX}chain-{names.STAMP}@main"
    assert ROUTE.fullmatch(k.removeprefix(names.REPO_PREFIX)), \
        "a key must pass the bridges' route regex"


def test_stamp_is_short_and_route_safe():
    # any length: a pinned BRIDGE_RUN_STAMP=r1 is valid; route safety is what matters
    assert re.fullmatch(r"[A-Za-z0-9_-]+", names.STAMP)


def test_pinned_stamp_gets_a_nonce(monkeypatch):
    """BRIDGE_RUN_STAMP=r1 -> "r1-<4 hex>": two processes pinned alike never
    share a stamp, and the operator still finds them by the "-r1" prefix."""
    import importlib
    original = names.STAMP
    monkeypatch.setenv("BRIDGE_RUN_STAMP", "r1")
    fresh = importlib.reload(names)
    try:
        assert re.fullmatch(r"r1-[0-9a-f]{4}", fresh.STAMP), fresh.STAMP
        k = fresh.key("x")
        assert fresh.is_ours(k) and fresh.matches_stamp(k, "r1") and fresh.matches_stamp(k, fresh.STAMP)
        assert not fresh.matches_stamp(k, "r")            # a prefix of the pinned part is not it
        assert not fresh.matches_stamp(k, fresh.STAMP[:-1])
    finally:
        # the process keeps ONE stamp: a live story in this session must find
        # its sessions under the stamp they were created with
        monkeypatch.delenv("BRIDGE_RUN_STAMP")
        importlib.reload(names)
        names.STAMP = original


def test_bare_equals_the_name_part_of_key():
    assert names.bare("chain") == names.name_part(names.key("chain"))
    assert names.bare("chain") == names.key("chain").split("@", 1)[0]
    assert "@" not in names.bare("chain")


def test_key_accepts_a_base_with_a_slash():
    k = names.key("q", "origin/main")
    assert k.endswith("@origin/main")
    assert names.name_part(k) == names.bare("q")


def test_key_with_empty_base_is_branchless():
    assert names.key("q", "") == names.bare("q")
    assert names.key("q", None) == names.bare("q")


def test_raw_returns_text_unchanged():
    for probe in ["", "@", " x ", "a/b", "n@@m", "x" * 200]:
        assert names.raw(probe) == probe


def test_prefix_switch_changes_key_and_bare_and_nothing_else(monkeypatch):
    old = names.REPO_PREFIX
    new = "other/" if old != "other/" else "another/"
    before_key, before_bare, before_stamp = names.key("s"), names.bare("s"), names.STAMP
    monkeypatch.setattr(names, "REPO_PREFIX", new)
    # key/bare change by exactly the prefix swap
    assert names.key("s") == new + before_key.removeprefix(old)
    assert names.bare("s") == new + before_bare.removeprefix(old)
    assert names.key("s", "origin/dev") == new + before_bare.removeprefix(old) + "@origin/dev"
    # raw, STAMP and ownership do not
    assert names.STAMP == before_stamp
    assert names.raw("@@") == "@@"
    assert names.is_ours(names.key("s"))
    assert names.is_ours(names.bare("s"))
    assert names.is_ours(before_key) and names.is_ours(before_bare)


def test_is_ours_recognizes_only_this_runs_stamp():
    assert names.is_ours(names.key("a"))
    assert names.is_ours(names.bare("a"))
    assert names.is_ours(names.key("a", "origin/dev"))
    assert not names.is_ours("a@main")
    assert not names.is_ours(f"a-{names.STAMP}x@main")      # stamp not at the end of the name
    assert not names.is_ours(f"a@main-{names.STAMP}")       # stamp in the base does not count
    assert not names.is_ours("")


def test_matches_stamp_with_another_stamp():
    assert names.matches_stamp("a-r123456@main", "r123456")
    assert not names.matches_stamp("a-r123456@main", "r654321")
    assert not names.matches_stamp("a-r123456@main", "")
    # a nonced (pinned) stamp matches its pinned prefix and its full form only
    assert names.matches_stamp("a-r1-ab12@main", "r1") and names.matches_stamp("a-r1-ab12", "r1-ab12")
    assert not names.matches_stamp("a-r1-ab12@main", "r1-ab1")
    assert not names.matches_stamp("a-r10-ab12@main", "r1")
    assert not names.matches_stamp("a-r1-xyz@main", "r1")     # not a hex nonce: another stamp


# --- live ownership is enforced in the client, before any HTTP ---------------------


def _no_network(monkeypatch):
    def boom(*a, **kw):
        pytest.fail(f"HTTP attempted for a session this run does not own: {a} {kw}")
    monkeypatch.setattr(httpx.Client, "request", boom)


@pytest.mark.parametrize("agent", list(live.PORTS))
def test_bridge_refuses_a_foreign_session_before_any_http(monkeypatch, agent):
    _no_network(monkeypatch)
    b = live.Bridge.for_agent(agent)
    foreign = names.raw("operator@main")
    with pytest.raises(ValueError):
        b.delete(foreign)
    with pytest.raises(ValueError):
        b.chat(foreign, "hi")
    with pytest.raises(ValueError):
        b.chat_raw(foreign, {"prompt": "hi"})
    with pytest.raises(ValueError):
        b.last(foreign)
    with pytest.raises(ValueError):
        b.clear(foreign)
    with pytest.raises(ValueError):
        b.reset(foreign)
    with pytest.raises(ValueError):
        b.teardown_session(foreign)


def test_registry_refuses_a_key_already_live_on_the_bridge(monkeypatch):
    _no_network(monkeypatch)
    b = live.Bridge.for_agent("agy")
    taken = names.key("taken")
    monkeypatch.setattr(live.Bridge, "health", lambda self, timeout=10.0: live.Reply(
        200, {"sessions": [{"name": taken, "alive": True, "turn_count": 1}]}, "{}", 0.1))
    monkeypatch.setattr(live.Bridge, "pane_pids", lambda self, k: [])
    reg = live.SessionRegistry(b)
    with pytest.raises(ValueError, match="already live on the bridge"):
        reg.register(taken)
    assert reg.keys == []
    assert reg("free") == names.key("free")  # a key not on the bridge is adopted


def test_kill_verified_pane_goes_through_sh_and_checks_rc(monkeypatch):
    _no_network(monkeypatch)
    b = live.Bridge.for_agent("agy")
    key = names.key("victim")
    want = b.tmux_session_name(key)
    calls: list[tuple[str, ...]] = []
    state = {"rc": 1, "pane": True}

    def fake_exec(self, *cmd, timeout=30.0):
        calls.append(cmd)
        if cmd[:2] == ("sh", "-c") and cmd[2].startswith("tmux "):
            panes = f"{want} 777\nagy-somebody-else 778\n" if state["pane"] else "agy-somebody-else 778\n"
            return 0, panes + "__tmux_rc=0\n"
        if cmd[:2] == ("sh", "-c") and cmd[2].startswith("kill "):
            if state["rc"] == 0:
                state["pane"] = False
            return state["rc"], "" if state["rc"] == 0 else "kill: 777: No such process"
        raise AssertionError(cmd)

    monkeypatch.setattr(live.Bridge, "docker_exec", fake_exec)
    # `kill` is a shell builtin in the container: it must run through sh -c
    assert b.kill_verified_pane(key) is None       # rc != 0 -> failed kill -> None
    assert ("sh", "-c", "kill -9 777") in calls
    assert not any(c[:1] == ("kill",) for c in calls)
    state["rc"] = 0
    assert b.kill_verified_pane(key) == 777        # rc 0 and the pane re-listed as gone
    # a foreign key is never killed, and never even listed
    calls.clear()
    assert b.kill_verified_pane(names.raw("operator@main")) is None
    assert calls == []


def test_reply_supports_in_and_get():
    rep = live.Reply(200, {"via": "bell", "turn": 1}, "{}", 0.1)
    assert "via" in rep and "nope" not in rep
    assert rep.get("via") == "bell" and rep.get("nope", "d") == "d"
    empty = live.Reply(None, None, "", 0.1, "timeout")
    assert "via" not in empty and empty.get("via") is None
    with pytest.raises(KeyError):
        empty["via"]


def test_own_refuses_path_tricks_in_a_stamped_key(monkeypatch):
    _no_network(monkeypatch)
    b = live.Bridge.for_agent("agy")
    for bad in [names.key("probe", "main/../../stop"), names.key("probe", "./main"),
                names.key("probe", "main//x"), names.key("probe", "main/"),
                names.key("probe", "main x"), names.key("pro be")]:
        with pytest.raises(ValueError):
            b.delete(bad)
    # the honest forms still pass the gate (no HTTP: _own returns first, then the
    # monkeypatched request would fail the test - so probe _own directly)
    assert live._own(names.key("probe")) == names.key("probe")
    assert live._own(names.key("probe", "origin/dev")) == names.key("probe", "origin/dev")
    assert live._own(names.bare("probe")) == names.bare("probe")


def test_own_strips_the_repo_prefix_but_keeps_the_traversal_checks(monkeypatch):
    _no_network(monkeypatch)
    b = live.Bridge.for_agent("agy")
    monkeypatch.setattr(names, "REPO_PREFIX", "other/")
    good = names.key("probe")
    assert good.startswith("other/") and live._own(good) == good
    assert live._own(names.key("probe", "origin/dev")) == names.key("probe", "origin/dev")
    assert live._own(names.bare("probe")) == names.bare("probe")
    for bad in [f"other/../x-{names.STAMP}@main", f"other/./x-{names.STAMP}@main",
                f"other//x-{names.STAMP}@main", names.key("probe", "main/../../stop"),
                f"other/x/y-{names.STAMP}@main"]:
        assert names.is_ours(bad)          # the stamp alone would let it through
        with pytest.raises(ValueError):
            b.delete(bad)


def _panes_exec(marker_present: bool, tmux_rc: int, body: str = ""):
    def fake_exec(self, *cmd, timeout=30.0):
        assert cmd[:2] == ("sh", "-c") and "list-panes" in cmd[2], cmd
        if not marker_present:
            return 1, "Error response from daemon: No such container: gemini-cli-rest-bridges-1\n"
        return tmux_rc, body + f"__tmux_rc={tmux_rc}\n"
    return fake_exec


def test_pane_listing_tells_no_panes_from_no_container(monkeypatch):
    _no_network(monkeypatch)
    b = live.Bridge.for_agent("codex")
    key = names.key("p")
    monkeypatch.setattr(live.Bridge, "docker_exec", _panes_exec(False, 1))
    assert b._list_panes_checked() is None and b.pane_pids(key) is None
    assert b.list_panes() == []
    # a confirmed "nothing to list" from tmux (3.5a wording, and the older ones) -> []
    for text in ["error connecting to /tmp/tmux-0/codex-rest (No such file or directory)\n",
                 "no current target\n", "no server running on /tmp/tmux-0/codex-rest\n",
                 "no sessions\n"]:
        monkeypatch.setattr(live.Bridge, "docker_exec", _panes_exec(True, 1, text))
        assert b._list_panes_checked() == [] and b.pane_pids(key) == [], text
    # any other tmux failure is "unknown", never "none"
    monkeypatch.setattr(live.Bridge, "docker_exec", _panes_exec(True, 1, "usage: list-panes [-as]\n"))
    assert b._list_panes_checked() is None and b.pane_pids(key) is None
    monkeypatch.setattr(live.Bridge, "docker_exec", _panes_exec(True, 1, "error connecting to /tmp/x (Permission denied)\n"))
    assert b._list_panes_checked() is None
    monkeypatch.setattr(live.Bridge, "docker_exec",
                        _panes_exec(True, 0, f"{b.tmux_session_name(key)} 900\nother 901\n"))
    assert b._list_panes_checked() == [(b.tmux_session_name(key), 900), ("other", 901)]
    assert b.pane_pids(key) == [900]


def test_registry_requires_verified_health_and_no_orphan_pane(monkeypatch):
    _no_network(monkeypatch)
    b = live.Bridge.for_agent("agy")
    key = names.key("orphan")
    reg = live.SessionRegistry(b)
    # health unavailable -> refuse
    monkeypatch.setattr(live.Bridge, "health",
                        lambda self, timeout=10.0: live.Reply(None, None, "", 0.1, "ConnectError"))
    monkeypatch.setattr(live.Bridge, "pane_pids", lambda self, k: [])
    with pytest.raises(ValueError, match="health unavailable"):
        reg.register(key)
    # health fine, but the container cannot be asked for panes -> refuse
    monkeypatch.setattr(live.Bridge, "health",
                        lambda self, timeout=10.0: live.Reply(200, {"sessions": []}, "{}", 0.1))
    monkeypatch.setattr(live.Bridge, "pane_pids", lambda self, k: None)
    with pytest.raises(ValueError, match="cannot be asked"):
        reg.register(key)
    # health fine, entry absent, but an orphan pane named for the key -> refuse
    monkeypatch.setattr(live.Bridge, "pane_pids", lambda self, k: [4242])
    with pytest.raises(ValueError, match="orphan pane"):
        reg.register(key)
    assert reg.keys == []
    monkeypatch.setattr(live.Bridge, "pane_pids", lambda self, k: [])
    assert reg.register(key) == key and reg.key("other") == names.key("other")
    # the same checks are one public call for the retained suites' _adopt()
    monkeypatch.setattr(live.Bridge, "pane_pids", lambda self, k: [4242])
    with pytest.raises(ValueError, match="orphan pane"):
        b.assert_not_live(key)
    monkeypatch.setattr(live.Bridge, "pane_pids", lambda self, k: [])
    assert b.assert_not_live(key) is None


def test_teardown_does_not_take_a_dead_health_for_absence(monkeypatch):
    _no_network(monkeypatch)
    b = live.Bridge.for_agent("agy")
    key = names.key("gone")
    monkeypatch.setattr(live.Bridge, "delete",
                        lambda self, k, timeout=0: live.Reply(500, None, "boom", 0.1))
    monkeypatch.setattr(live.Bridge, "pane_pids", lambda self, k: [])
    monkeypatch.setattr(live.Bridge, "worktree_dirs", lambda self, k: [])
    monkeypatch.setattr(live.Bridge, "health",
                        lambda self, timeout=10.0: live.Reply(None, None, "", 0.1, "ConnectError"))
    assert b.teardown_session(key) == "failed:health-unavailable"
    monkeypatch.setattr(live.Bridge, "health",
                        lambda self, timeout=10.0: live.Reply(200, {"sessions": []}, "{}", 0.1))
    assert b.teardown_session(key) == "absent"


def test_for_url_parses_host_and_port():
    b = live.Bridge.for_url("codex", "http://127.0.0.1:8001")
    assert (b.agent, b.host, b.port) == ("codex", "127.0.0.1", 8001)
    assert live.Bridge.for_url("agy", "http://bridge.local").port == live.PORTS["agy"]
    with pytest.raises(ValueError):
        live.Bridge.for_url("agy", "not a url")


def test_request_has_an_absolute_deadline(monkeypatch):
    import threading
    release = threading.Event()

    def trickle(*a, **kw):
        release.wait(30)   # a response that never finishes within the deadline
        raise RuntimeError("late")

    monkeypatch.setattr(httpx.Client, "request", trickle)
    monkeypatch.setattr(live.Bridge, "DEADLINE_SLACK", 0.2)
    b = live.Bridge.for_agent("agy")
    rep = b.health(timeout=0.2)
    release.set()  # let the abandoned worker finish
    assert rep.status is None and "deadline" in (rep.error or "")
    assert rep.elapsed < 5.0


def test_fake_git_empty_refs_stay_empty():
    from bridgetests import fakes
    assert "origin/main" in fakes.FakeGit().refs
    assert "origin/main" in fakes.FakeGit(None).refs
    assert fakes.FakeGit({}).refs == {}
    assert fakes.FakeGit({"origin/dev": "abc"}).refs == {"origin/dev": "abc"}


def test_dotted_base_is_matched_under_the_name_tmux_lists(monkeypatch):
    _no_network(monkeypatch)
    b = live.Bridge.for_agent("agy")
    key = names.key("x", "release/1.2")
    derived = b.tmux_session_name(key)            # what the server passes to new-session
    listed = live.tmux_display_name(derived)      # what tmux shows in list-panes
    # the servers derive through worktree.tmux_safe_name: no dot ever reaches
    # tmux, so the listed name IS the derived one (the path keeps the dot)
    assert "release-1_2-" in derived and listed == derived
    assert derived == f"agy-{live.worktree.tmux_safe_name(key)}"
    assert "release-1.2-" in live.worktree.safe_name(key)
    assert live.tmux_display_name("agy-x.y:z_w") == "agy-x_y_z_w"
    monkeypatch.setattr(live.Bridge, "docker_exec",
                        _panes_exec(True, 0, f"{listed} 900\nother 901\n"))
    assert b.pane_pids(key) == [900]
    # a pane listed under the OLD dotted derivation (a pre-fix leftover) is not ours
    old = f"agy-{live.worktree.safe_name(key)}"
    monkeypatch.setattr(live.Bridge, "docker_exec",
                        _panes_exec(True, 0, f"{old} 900\nother 901\n"))
    assert b.pane_pids(key) == []


def test_fake_cli_lists_sessions_the_way_tmux_does(monkeypatch, tmp_path):
    """The fake CLI keeps tmux 3.5a's behaviour (probed in the container):
    new-session lists a dotted name normalized, and a later `-t ={dotted}:`
    (the servers' _target form) fails with "can't find session: <dotted>".
    The servers no longer derive a dotted name (worktree.tmux_safe_name), so
    the same key now resolves; a hermetic story in test_hermetic_smoke.py
    drives the full turn."""
    import asyncio
    import types
    from bridgetests import fakes
    fb = fakes.install(monkeypatch, tmp_path, "agy")
    key = names.key("x", "release/1.2")
    dotted = f"agy-{live.worktree.safe_name(key)}"          # the pre-fix derivation
    derived = fb.tmux_session_name(key)                     # what the server derives now
    assert "." in dotted and "." not in derived and derived == live.tmux_display_name(dotted)
    session_cls = getattr(fb.module, "AgySession", None) or fb.module.CodexSession
    assert session_cls._target.fget(types.SimpleNamespace(tmux_session=derived)) == f"={derived}:"
    run = lambda *a: asyncio.run(fb.cli._tmux(*a))
    assert run("new-session", "-d", "-s", dotted, "-c", str(tmp_path), "true") == (0, "")
    assert fb.cli.live_sessions() == [derived]              # created, listed normalized
    assert run("has-session", "-t", f"={dotted}:") == (1, f"can't find session: {dotted}")
    assert run("has-session", "-t", f"={derived}:") == (0, "")   # the normalized form resolves
    # a bare dotted target is a pane lookup, as in tmux
    assert run("has-session", "-t", dotted) == (1, f"can't find pane: {dotted.rsplit('.', 1)[1]}")


def test_teardown_waits_for_an_abandoned_request(monkeypatch):
    import threading
    release = threading.Event()

    def slow_post(self, method, url, **kw):
        release.wait(30)
        return httpx.Response(200, json={"session": "x", "turn": 1, "response": "late"})

    monkeypatch.setattr(httpx.Client, "request", slow_post)
    monkeypatch.setattr(live.Bridge, "DEADLINE_SLACK", 0.1)
    monkeypatch.setattr(live, "DELETE_DEADLINE", 0.5)
    deletes: list[str] = []
    monkeypatch.setattr(live.Bridge, "delete",
                        lambda self, k, timeout=0: (deletes.append(k), live.Reply(404, None, "", 0.0))[1])
    monkeypatch.setattr(live.Bridge, "pane_pids", lambda self, k: [])
    monkeypatch.setattr(live.Bridge, "worktree_dirs", lambda self, k: [])
    b = live.Bridge.for_agent("agy")
    key = names.key("late")
    rep = b.chat(key, "hi", timeout=0.1)
    assert rep.status is None and "deadline" in rep.error
    assert len(b.pending_requests(key)) == 1
    # still in flight past the (patched) delete deadline: unresolved, nothing deleted
    assert b.teardown_session(key) == "failed:request-still-pending"
    assert deletes == []
    # the worker finishes late, before the next teardown's deadline: proceed
    threading.Timer(0.2, release.set).start()
    assert b.teardown_session(key) == "absent"
    assert deletes == [key] and b.pending_requests(key) == [] and key not in b._pending


def test_request_is_pending_while_its_worker_runs(monkeypatch):
    """A request is registered as pending BEFORE its worker starts, so
    pending_requests/wait_pending see it while it is in flight, not only once
    its caller gave up at the deadline."""
    import threading
    started, release = threading.Event(), threading.Event()

    def blocked(self, method, url, **kw):
        started.set()
        release.wait(30)
        return httpx.Response(200, json={"session": "x", "turn": 1, "response": "ok"})

    monkeypatch.setattr(httpx.Client, "request", blocked)
    b = live.Bridge.for_agent("agy")
    key = names.key("inflight")
    out: list = []
    t = threading.Thread(target=lambda: out.append(b._request("POST", "/chat/x", timeout=5.0, session=key)))
    t.start()
    assert started.wait(5)
    assert len(b.pending_requests(key)) == 1
    assert not b.wait_pending(key, 0.05)
    release.set()
    t.join(5)
    assert out and out[0].status == 200
    assert b.wait_pending(key, 2.0)
    assert b.pending_requests(key) == [] and key not in b._pending


def _health_with(*keys):
    return lambda self, timeout=10.0: live.Reply(
        200, {"sessions": [{"name": k, "alive": True, "turn_count": 1} for k in keys]}, "{}", 0.1)


def test_uncertain_key_is_rechecked_and_marked(monkeypatch):
    key = names.key("unsure")

    def read_timeout(self, method, url, **kw):
        raise httpx.ReadTimeout("slow")

    monkeypatch.setattr(httpx.Client, "request", read_timeout)
    b = live.Bridge.for_agent("agy")
    rep = b.chat(key, "hi", timeout=0.2)
    assert rep.status is None and "timeout" in rep.error and key in b._uncertain
    assert b.health().status is None and "health" not in b._uncertain   # only session requests count
    # nothing visible: the outcome carries a "?" - reported once, so the key
    # is no longer uncertain and the next teardown says a plain word
    monkeypatch.setattr(live.Bridge, "delete", lambda self, k, timeout=0: live.Reply(404, None, "", 0.0))
    monkeypatch.setattr(live.Bridge, "pane_pids", lambda self, k: [])
    monkeypatch.setattr(live.Bridge, "worktree_dirs", lambda self, k: [])
    monkeypatch.setattr(live.Bridge, "health", _health_with())
    assert b.teardown_session(key) == "absent?" and key not in b._uncertain
    assert b.teardown_session(key) == "absent"
    # the late request landed: the re-check lists it, it is deleted again -> confirmed
    b._uncertain.add(key)
    seen = {"n": 0}

    def health_then_listed(self, timeout=10.0):
        seen["n"] += 1
        return _health_with(key)(self) if seen["n"] == 1 else _health_with()(self)

    deletes: list[str] = []
    monkeypatch.setattr(live.Bridge, "health", health_then_listed)
    monkeypatch.setattr(live.Bridge, "delete",
                        lambda self, k, timeout=0: (deletes.append(k), live.Reply(200, None, "", 0.0))[1])
    assert b.teardown_session(key) == "deleted" and key not in b._uncertain
    assert deletes == [key, key]
    # a never-uncertain key is untouched by all this
    other = names.key("sure")
    assert b.teardown_session(other) == "deleted" and other not in b._uncertain


def test_teardowns_own_abandoned_delete_is_not_a_success(monkeypatch):
    import threading
    release = threading.Event()

    def slow_delete(self, method, url, **kw):
        release.wait(30)
        return httpx.Response(200, json={"status": "ok"})

    monkeypatch.setattr(httpx.Client, "request", slow_delete)
    monkeypatch.setattr(live.Bridge, "DEADLINE_SLACK", 0.1)
    monkeypatch.setattr(live, "DELETE_DEADLINE", 0.2)
    monkeypatch.setattr(live.Bridge, "pane_pids", lambda self, k: [])
    monkeypatch.setattr(live.Bridge, "worktree_dirs", lambda self, k: [])
    monkeypatch.setattr(live.Bridge, "health", _health_with())
    b = live.Bridge.for_agent("codex")
    key = names.key("slowdel")
    assert b.teardown_session(key) == "failed:request-still-pending"
    assert len(b.pending_requests(key)) == 1
    release.set()


def test_worktree_left_on_disk_is_swept_or_unresolved(monkeypatch):
    _no_network(monkeypatch)
    b = live.Bridge.for_agent("agy")
    key = names.key("wt")
    safe = live.worktree.safe_name(key)
    root = b.sessions_root
    disk = {"dirs": [f"{root}/abc123/{safe}/c1", f"{root}/abc123/{safe}/c2"], "sweep_clears": True,
            "listable": True}
    sweeps: list[str] = []

    def fake_exec(self, *cmd, timeout=30.0):
        if cmd[:2] == ("sh", "-c") and "list-panes" in cmd[2]:
            return 1, "no current target\n__tmux_rc=1\n"
        if cmd[:2] == ("sh", "-c") and cmd[2].startswith("for d in"):
            assert f"{root}/*/{safe}/c*" in cmd[2] and cmd[2].endswith("echo __wt_listed__")
            if not disk["listable"]:
                return 127, "docker: no such container\n"
            return 0, "".join(d + "\n" for d in disk["dirs"]) + "__wt_listed__\n"
        if cmd[:2] == ("sh", "-c") and "worktree remove --force" in cmd[2]:
            sweeps.append(cmd[2])
            if disk["sweep_clears"]:
                disk["dirs"] = []
            return 0, ""
        raise AssertionError(cmd)

    monkeypatch.setattr(live.Bridge, "docker_exec", fake_exec)
    monkeypatch.setattr(live.Bridge, "delete", lambda self, k, timeout=0: live.Reply(200, None, "", 0.0))
    # leftovers after a delete are swept: remove --force (rm -rf fallback) per dir, then prune
    assert b.teardown_session(key) == "deleted+swept"
    [script] = sweeps
    for d in (f"{root}/abc123/{safe}/c1", f"{root}/abc123/{safe}/c2"):
        assert f"git -C {live.REPO_IN_CONTAINER} worktree remove --force {d} || rm -rf {d}" in script
    assert script.endswith(f"git -C {live.REPO_IN_CONTAINER} worktree prune")
    assert "deleted+swept" in live.Bridge.CONFIRMED and "killed+deleted+swept" in live.Bridge.CONFIRMED
    # a sweep that leaves dirs behind stays unresolved
    disk["dirs"], disk["sweep_clears"] = [f"{root}/abc123/{safe}/c2"], False
    out = b.teardown_session(key)
    assert out.startswith("failed:worktree-left(") and f"{root}/abc123/{safe}/c2" in out
    # a path outside this bridge's sessions root is never removed
    sweeps.clear()
    disk["dirs"] = ["/app/slitled-platform"]
    out = b.teardown_session(key)
    assert out.startswith("failed:worktree-left(") and sweeps == []
    # "absent" consults the disk too (a generation can outlive the manager entry)
    disk["dirs"], disk["sweep_clears"] = [f"{root}/abc123/{safe}/c2"], True
    monkeypatch.setattr(live.Bridge, "delete", lambda self, k, timeout=0: live.Reply(404, None, "", 0.0))
    assert b.teardown_session(key) == "absent+swept" and "absent+swept" in live.Bridge.CONFIRMED
    assert b.worktree_dirs(key) == [] and b.teardown_session(key) == "absent"
    disk["dirs"] = []
    monkeypatch.setattr(live.Bridge, "delete", lambda self, k, timeout=0: live.Reply(200, None, "", 0.0))
    assert b.teardown_session(key) == "deleted"
    # a disk that cannot be inspected is not "nothing on disk"
    disk["listable"] = False
    assert b.worktree_dirs(key) is None
    assert b.teardown_session(key) == "failed:cannot-list-worktrees"
    # ... also when it is the re-listing after a sweep
    disk["dirs"], disk["listable"], sweeps[:] = [f"{root}/abc123/{safe}/c1"], True, []
    monkeypatch.setattr(live.Bridge, "docker_exec",
                        lambda self, *cmd, timeout=30.0: (disk.update(listable=False), fake_exec(self, *cmd))[1]
                        if "worktree remove --force" in cmd[-1] else fake_exec(self, *cmd))
    assert b.teardown_session(key) == "failed:cannot-list-worktrees" and len(sweeps) == 1


def test_sweep_refuses_foreign_keys_and_paths_not_shaped_as_the_keys_generation(monkeypatch):
    _no_network(monkeypatch)
    execs: list = []
    monkeypatch.setattr(live.Bridge, "docker_exec",
                        lambda self, *cmd, timeout=30.0: (execs.append(cmd), (0, ""))[1])
    b = live.Bridge.for_agent("codex")
    key = names.key("wt")
    safe = live.worktree.safe_name(key)
    root = b.sessions_root
    with pytest.raises(ValueError):
        b.sweep_worktrees(names.raw("wt-other@main"), [f"{root}/abc123/{safe}/c1"])
    other = live.worktree.safe_name(names.key("other"))
    for bad in ([f"{root}/abc123/{other}/c1"],            # another key's generation
                [f"{root}/abc123/{safe}"],                # the slug dir, not a generation
                [f"{root}/abc123/{safe}/c1/deeper"],      # below a generation
                [f"{root}/abc123/{safe}/x1"],             # not a c<generation>
                [f"{root}/abc123/{safe}/cache"],          # starts with c, not c<int>
                [f"{root}/../abc123/{safe}/c1"],          # traversal
                [f"{root}/./{safe}/c1"],                  # `.` run-id segment
                [f"{root}//{safe}/c1"],                   # empty run-id segment
                [f"{safe}/c1"],                           # outside the root
                [f"{root}/abc123/{safe}/c1", f"{root}/abc123/{safe}"]):   # one bad dir spoils the set
        assert b.sweep_worktrees(key, bad) == bad
    assert b.sweep_worktrees(key, []) == []          # nothing to sweep, no script run
    assert execs == []


def test_abandoned_worker_failure_marks_the_key_uncertain(monkeypatch):
    """An abandoned request (deadline passed, caller already answered) that
    later fails inside httpx may still have been run by the server: the
    WORKER records the uncertainty, not the caller that is no longer there."""
    import threading
    release = threading.Event()

    def late_timeout(self, method, url, **kw):
        release.wait(30)
        raise httpx.ReadTimeout("late")

    monkeypatch.setattr(httpx.Client, "request", late_timeout)
    monkeypatch.setattr(live.Bridge, "DEADLINE_SLACK", 0.1)
    b = live.Bridge.for_agent("agy")
    key = names.key("late-err")
    rep = b.chat(key, "hi", timeout=0.1)
    assert rep.status is None and "deadline" in rep.error and key not in b._uncertain
    assert len(b.pending_requests(key)) == 1
    release.set()
    assert b.wait_pending(key, 2.0)
    assert key in b._uncertain and key not in b._pending
    # a non-httpx failure of an abandoned worker is not an uncertainty
    monkeypatch.setattr(httpx.Client, "request", lambda self, m, u, **kw: (_ for _ in ()).throw(RuntimeError("x")))
    other = names.key("plain-err")
    with pytest.raises(RuntimeError):
        b.chat(other, "hi", timeout=0.1)
    assert other not in b._uncertain


def test_group_session_default_name_comes_from_module_and_class(monkeypatch):
    _no_network(monkeypatch)
    monkeypatch.setattr(live.Bridge, "assert_not_live", lambda self, k: None)
    reg = live.SessionRegistry(live.Bridge.for_agent("agy"), label="group:test_common:TestHealthShape")
    name = live.GroupSession.default_name_for("test_common", "TestHealthShape")
    assert name == "group-common-testhealthshape"
    g = live.GroupSession(reg, default_name=name)
    assert g() == names.key(name) and g() == g.key()
    # same class name in two modules -> two keys
    assert live.GroupSession.default_name_for("stories.test_agy", "TestHealthShape") \
        == "group-agy-testhealthshape" != name
    assert live.GroupSession.default_name_for(None, None) == "group-module-class"
    assert live._own(names.key(live.GroupSession.default_name_for("test_x.y", "Test_Odd$Name")))


def test_close_ends_the_client_and_its_abandoned_worker(monkeypatch):
    """close() is idempotent and rebuilds on demand; a worker abandoned past
    the deadline that is still inside the client ends once the client closes."""
    import threading
    started, release = threading.Event(), threading.Event()

    def trickle(self, method, url, **kw):
        started.set()
        release.wait(30)
        if self.is_closed:               # what a real transport does after close(): the request fails
            raise httpx.CloseError("client closed")
        return httpx.Response(200, json={"status": "ok"})

    monkeypatch.setattr(httpx.Client, "request", trickle)
    monkeypatch.setattr(live.Bridge, "DEADLINE_SLACK", 0.1)
    b = live.Bridge.for_agent("agy")
    first = b.client()
    assert b.client() is first
    rep = b.health(timeout=0.1)
    assert rep.status is None and "deadline" in rep.error and started.wait(1)
    worker = [t for t in threading.enumerate() if t.name == "bridge-http"]
    assert worker and worker[0].daemon
    b.close()
    b.close()                            # idempotent
    assert b._client is None and first.is_closed
    release.set()
    worker[0].join(2)
    assert not worker[0].is_alive()
    assert b.client() is not first and not b.client().is_closed
    live.close_all()
    assert b._client is None


def test_fake_clock_wait_for_times_out_on_fake_time():
    import asyncio
    import time as real_time
    from bridgetests import fakes
    clock = fakes.FakeClock()
    proxy = fakes._AsyncioProxy(clock)

    async def never():
        await asyncio.get_running_loop().create_future()

    async def scenario():
        t0 = real_time.perf_counter()
        with pytest.raises(asyncio.TimeoutError):
            await proxy.wait_for(never(), 30)
        assert real_time.perf_counter() - t0 < 1.0
    asyncio.run(scenario())
    assert clock.elapsed == 30


def test_fake_clock_wait_for_returns_when_the_task_finishes_in_time():
    import asyncio
    from bridgetests import fakes
    clock = fakes.FakeClock()
    proxy = fakes._AsyncioProxy(clock)

    async def soon():
        await proxy.sleep(2)
        return "ok"

    async def scenario():
        assert await proxy.wait_for(soon(), 5) == "ok"
        assert await proxy.wait_for(soon(), None) == "ok"   # no timeout: a plain await
    asyncio.run(scenario())
    assert clock.elapsed == 4


def test_baseline_compare_and_approve(tmp_path, monkeypatch):
    import json
    from bridgetests import baseline
    story = lambda n, o, ran=True, hermetic=False: {"nodeid": n, "outcome": o, "ran": ran, "hermetic": hermetic}
    ref = {"stories": [story("a", "passed"), story("b", "passed"), story("gone", "passed")]}
    run = {"stories": [story("a", "passed"), story("b", "failed"), story("c", "passed"),
                       story("d", "skipped", ran=False)]}
    assert baseline.compare(run, ref) == {
        "changed": [("b", "passed", "failed")], "missing": ["gone"], "new": ["c"]}
    # a story skipped in the run is neither changed nor new
    run2 = {"stories": [story("a", "skipped", ran=False), story("d", "skipped", ran=False)]}
    assert baseline.compare(run2, ref) == {"changed": [], "missing": ["b", "gone"], "new": []}
    # approve refuses a run with a failure, accepts a green one
    monkeypatch.setenv("BRIDGE_BASELINE_REFERENCE", str(tmp_path / "ref" / "reference.json"))
    red = tmp_path / "red.json"
    red.write_text(json.dumps(run))
    assert baseline.approve(red) is None and not baseline.reference_path().exists()
    green = tmp_path / "green.json"
    green.write_text(json.dumps({"stories": [story("a", "passed"), story("d", "skipped", ran=False)]}))
    assert baseline.approve(green) == baseline.reference_path()
    assert json.loads(baseline.reference_path().read_text()) == json.loads(green.read_text())
    # a hermetic-only run is not a reference
    herm = tmp_path / "herm.json"
    herm.write_text(json.dumps({"stories": [story("h", "passed", hermetic=True)]}))
    assert baseline.approve(herm) is None
    # a failed teardown refuses the run even when every story passed
    torn = tmp_path / "torn.json"
    torn.write_text(json.dumps({"stories": [dict(story("a", "passed"), teardown_error=True)]}))
    assert baseline.approve(torn) is None
    # an unresolved teardown refuses the run even when every story passed
    unres = tmp_path / "unres.json"
    unres.write_text(json.dumps({"stories": [story("a", "passed")],
                                 "unresolved_teardowns": [["agy", "k-x@main", "deleted?"]]}))
    assert baseline.approve(unres) is None
    # so does a pytest run that did not finish cleanly
    assert baseline.approve(green, exit_ok=False) is None
    assert json.loads(baseline.reference_path().read_text()) == json.loads(green.read_text())


def test_baseline_reads_unresolved_teardowns_at_build_time(monkeypatch):
    from bridgetests import baseline
    monkeypatch.setattr(live, "UNRESOLVED", [])
    rec = baseline.BaselineRecorder({"agy": 8000}, "c")
    assert rec.build()["unresolved_teardowns"] == []
    live.UNRESOLVED.append(("agy", "k@main", "deleted?"))
    assert rec.build()["unresolved_teardowns"] == [["agy", "k@main", "deleted?"]]


def test_baseline_teardown_failure_marks_the_story_error():
    import types
    from bridgetests import baseline
    item = types.SimpleNamespace(get_closest_marker=lambda n: None, module=types.SimpleNamespace(GROUP="A"))
    report = lambda nodeid, when, outcome: types.SimpleNamespace(
        nodeid=nodeid, when=when, outcome=outcome, duration=0.1)
    rec = baseline.BaselineRecorder({"agy": 8000}, "c")
    for nodeid, call in (("skipped", "skipped"), ("passed", "passed"), ("failed", "failed")):
        rec.record(item, report(nodeid, "setup", "passed"))
        rec.record(item, report(nodeid, "call", call))
        rec.record(item, report(nodeid, "teardown", "failed"))
    rec.record(item, report("setup-skipped", "setup", "skipped"))
    rec.record(item, report("setup-skipped", "teardown", "failed"))
    rec.record(item, report("clean", "setup", "passed"))
    rec.record(item, report("clean", "call", "passed"))
    rec.record(item, report("clean", "teardown", "passed"))
    got = {n: (r["outcome"], r.get("teardown_error")) for n, r in rec.records.items()}
    assert got == {"skipped": ("error", True), "passed": ("error", True), "failed": ("failed", True),
                   "setup-skipped": ("error", True), "clean": ("passed", None)}


def test_fake_clock_wait_for_cancels_the_child_when_the_caller_is_cancelled():
    import asyncio
    from bridgetests import fakes
    clock = fakes.FakeClock()
    proxy = fakes._AsyncioProxy(clock)
    state = {"child": "running"}

    async def child():
        try:
            await asyncio.get_running_loop().create_future()
        except asyncio.CancelledError:
            state["child"] = "cancelled"
            raise

    async def scenario():
        waiter = asyncio.ensure_future(proxy.wait_for(child(), 30))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert state["child"] == "cancelled"   # not left running behind the cancelled caller
    asyncio.run(scenario())
    assert clock.elapsed == 0                  # no timeout was reached
