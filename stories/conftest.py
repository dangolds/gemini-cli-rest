"""Fixtures and hooks for the behavior stories (see bridgetests/README.md).

Live fixtures (skip when no bridge answers /health):
  bridge          class-scoped, parametrized over ports (ids "agy", "codex")
  own_session     function-scoped registry: k = own_session("name") -> key,
                  deleted at story teardown (deadline + verified kill)
  group_session   class-scoped shared key: group_session() -> key, built on
                  first use, deleted at class teardown

Hermetic fixtures (no network, no container):
  fake_bridge     parametrized over servers (ids "agy", "codex"): FakeBridge
                  with .cli / .git / .clock / .client() (see bridgetests/fakes.py)

Hooks: collection fails (UsageError) when a hermetic story requests a live
fixture, so nothing touches the network on its behalf; the baseline recorder
(bridgetests/baseline.py) records every story and writes
logs/baseline/<date>-<stamp>.json when at least one live story ran, compares
it to baseline/reference.json when that exists (BRIDGE_BASELINE_APPROVE=1
makes a green run the new reference); keys whose teardown could not be
verified (live.UNRESOLVED) are printed at session finish, and every Bridge's
HTTP client is closed.
"""
from __future__ import annotations

import json

import pytest

from bridgetests import baseline, fakes, live, names

# --- baseline hooks --------------------------------------------------------------

_recorder: baseline.BaselineRecorder | None = None


def _get_recorder() -> baseline.BaselineRecorder:
    # Lazy: this conftest is only loaded at session start when stories/ is an
    # initial argument, so the first report creates the recorder if needed.
    global _recorder
    if _recorder is None:
        _recorder = baseline.BaselineRecorder(
            live.PORTS, live.CONTAINER,
            image_id_fn=lambda: live.Bridge.for_agent("agy").image_id(),
        )
    return _recorder


LIVE_FIXTURES = ("bridge", "own_session", "group_session")


def pytest_sessionstart(session):
    _get_recorder()


def pytest_collection_modifyitems(session, config, items):
    """A hermetic story must never request a live fixture: refuse at collection,
    before any fixture (and any HTTP) runs."""
    bad = [
        f"{item.nodeid} requests {sorted(set(item.fixturenames) & set(LIVE_FIXTURES))}"
        for item in items
        if item.get_closest_marker("hermetic") and set(item.fixturenames) & set(LIVE_FIXTURES)
    ]
    if bad:
        raise pytest.UsageError(
            "hermetic stories must not use the live fixtures "
            f"{LIVE_FIXTURES}:\n  " + "\n  ".join(bad))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    _get_recorder().record(item, outcome.get_result())


@pytest.hookimpl(trylast=True)  # after pytest's own fixture finalizers
def pytest_sessionfinish(session, exitstatus):
    if _recorder is None:
        return
    path = _recorder.write()
    if path is not None:
        print(f"\n[baseline] run stamp {names.STAMP}: wrote {path}", flush=True)
        _report_against_reference(path, exit_ok=(exitstatus == 0))
    else:
        print(f"\n[baseline] run stamp {names.STAMP}: no live story ran, no baseline written",
              flush=True)
    if live.UNRESOLVED:
        print(f"\n[teardown] {len(live.UNRESOLVED)} session(s) of run stamp {names.STAMP} "
              f"could NOT be verified gone - check the container:", flush=True)
        for agent, key, outcome in live.UNRESOLVED:
            print(f"  {agent} {key}: {outcome}", flush=True)
    live.close_all()


def _report_against_reference(run_path, *, exit_ok: bool = True) -> None:
    ref = baseline.reference_path()
    if ref.exists():
        try:
            diff = baseline.compare(json.loads(run_path.read_text(encoding="utf-8")),
                                    json.loads(ref.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            print(f"[baseline] cannot compare against {ref}: {exc}", flush=True)
        else:
            print(f"[baseline] vs reference {ref}: {len(diff['changed'])} changed, "
                  f"{len(diff['missing'])} missing, {len(diff['new'])} new", flush=True)
            for nodeid, before, after in diff["changed"][:20]:
                print(f"  changed {nodeid}: {before} -> {after}", flush=True)
    else:
        print(f"[baseline] no reference at {ref} (BRIDGE_BASELINE_APPROVE=1 on a green run creates it)",
              flush=True)
    if baseline.approve_requested():
        dest = baseline.approve(run_path, exit_ok=exit_ok)
        if dest is not None:
            print(f"[baseline] approved: {run_path} is now the reference {dest}", flush=True)


# --- live fixtures ------------------------------------------------------------------


@pytest.fixture(scope="session")
def live_health_report():
    """Printed (never asserted) /health of both ports before the first live
    story and after the run. Lazy: no HTTP happens here; the `bridge` fixture
    takes the "before" snapshot on its first use, and "after" is only printed
    when "before" was."""
    state = {"before": False}
    yield state
    if state["before"]:
        live.health_snapshot("after")


@pytest.fixture(scope="class", params=list(live.PORTS), ids=list(live.PORTS))
def bridge(request, live_health_report) -> live.Bridge:
    """The live bridge under test; every common story is parametrized over it."""
    if request.node.get_closest_marker("hermetic"):
        pytest.fail("a hermetic story must not use the live `bridge` fixture")
    if not live_health_report["before"]:
        live_health_report["before"] = True
        live.health_snapshot("before")
    b = live.Bridge.for_agent(request.param)
    if not b.is_up():
        pytest.skip(f"no bridge at {b.base_url} (docker compose up -d); live story skipped")
    return b


@pytest.fixture(autouse=True)
def _hermetic_never_live(request):
    """Backstop for a function-level hermetic marker inside a live class."""
    if request.node.get_closest_marker("hermetic") and "bridge" in request.fixturenames:
        pytest.fail("a hermetic story must not use the live `bridge` fixture")
    yield


@pytest.fixture()
def own_session(bridge) -> live.SessionRegistry:
    """Per-story sessions: `k = own_session("chain")` builds and registers the
    key; every registered key is deleted at teardown, also on failure."""
    reg = live.SessionRegistry(bridge, label="story")
    yield reg
    reg.teardown()


@pytest.fixture(scope="class")
def group_session(request, bridge) -> live.GroupSession:
    """One session shared by a class of read-only stories: `k = group_session()`
    (built on first use, named "group-<module>-<classname>"), deleted at class
    teardown."""
    classname = request.cls.__name__ if request.cls is not None else None
    module = getattr(request.node, "module", None) or request.module
    stem = module.__name__.rpartition(".")[2] if module is not None else None
    reg = live.SessionRegistry(bridge, label=f"group:{stem}:{classname or 'module'}")
    yield live.GroupSession(reg, default_name=live.GroupSession.default_name_for(stem, classname))
    reg.teardown()


# --- hermetic fixtures --------------------------------------------------------------


@pytest.fixture(params=list(live.PORTS), ids=list(live.PORTS))
def fake_bridge(request, monkeypatch, tmp_path) -> fakes.FakeBridge:
    """The server module under test wired to the fake CLI, fake git and fake
    clock, one per agent (ids "agy", "codex")."""
    return fakes.install(monkeypatch, tmp_path, request.param)
