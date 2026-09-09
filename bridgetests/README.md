# bridgetests — support package for the behavior stories

Stories live in `stories/` (one file per surface; `test_common.py` is the
parametrized common file). This package is what they import. Read `TestPRD.md`
for the rules; this file says how to use the pieces.

## Naming a session (`bridgetests.names`)

Never spell a key by hand. Three forms:

| form | gives | use for |
| --- | --- | --- |
| `names.key("chain")` | `chain-<STAMP>@main` | every live session (default base `main`; `names.key("q", "origin/dev")` for a slash base) |
| `names.bare("chain")` | `chain-<STAMP>` | the no-base management stories (equals the name part of `key`) |
| `names.raw("@@")` | `@@` unchanged | hermetic grammar probes ONLY — a raw name has no stamp and must never reach a live bridge |

`names.STAMP` is one token per process: random, or `BRIDGE_RUN_STAMP=r1` plus
a 4-hex nonce (`r1-ab12`), so two processes pinned alike never share a stamp;
`names.is_ours(k)` says whether a key carries this process's exact stamp,
`names.matches_stamp(k, "r1")` also accepts the pinned prefix (find your
sessions by `-r1-`). A registry still adopts a key only after `/health`
answered 200 without listing it and no tmux pane named for it exists
("pinned stamp collision or leftover"; health unavailable is also a refusal). `names.REPO_PREFIX` (`""` today) is
the single line the repository change flips to `"slitled/"`.

## Fixtures (`stories/conftest.py`)

Live (skipped, not failed, when `GET /health` on that port is not 200):

- `bridge` — the port under test, class-scoped, parametrized with ids `agy` and
  `codex`. Helpers take the FULL key and never raise on an HTTP outcome; every
  helper that names a session raises `ValueError` before any HTTP when the key
  does not carry this run's stamp or does not fit the servers' route grammar
  (no empty, `.` or `..` base segment; `health`/`sessions` are open):
  `bridge.chat(key, prompt, timeout=185)`, `bridge.last(key, wait=None)`,
  `bridge.clear(key)`, `bridge.reset(key)`, `bridge.delete(key)`,
  `bridge.health()`, `bridge.sessions()`, `bridge.assert_not_live(key)` (raises
  `ValueError` unless health is 200, the key unlisted and no pane named for it
  exists; the registry and the retained suites adopt keys through it); each
  HTTP helper returns a `Reply` with
  `.status` (None on a client timeout/connection error, then `.error`), `.body`
  (parsed JSON or None), `.text`, `.elapsed`; `rep["turn"]`, `"via" in rep`
  and `rep.get("via")` read the body.
  Disk/container: `bridge.docker_exec(*cmd)`, `bridge.worktree_dirs(key)`,
  `bridge.list_panes()`, `bridge.tmux_session_name(key)`, `bridge.sessions_root`.
- `own_session` — per-story ownership: `k = own_session("chain")` (alias
  `own_session.key("chain")`) builds `names.key("chain")` and registers it;
  `own_session.register(k)` adopts a key built with the helper. Every registered key is deleted at story teardown,
  also on failure. Use it for anything that mutates (clear, reset, delete,
  concurrency) or asserts on turn counts.
- `group_session` — one shared key per class: `k = group_session()` (built on
  first use as `group-<module>-<classname>`, e.g. `group-common-testhealthshape`,
  same key on every call), deleted at class
  teardown. Use it for
  read-only groups (health shape, last-answer readback, prompt content); never
  assert on its turn count.

Teardown rule: `DELETE` with a 200 s deadline (`live.DELETE_DEADLINE`; a running
turn may hold the lock for the 180 s cap). The status is not trusted on its
own (the servers drop the manager entry before stopping the session, so a
failed delete can leave a live pane that later answers 404): after the delete,
or once it has passed the deadline, teardown lists the container's tmux panes;
a surviving pane whose tmux session name equals
`<agent>-<worktree.tmux_safe_name(key)>` for a key this run owns is killed
(verified gone), the delete repeated and the panes listed again. A worktree
generation still on disk after a delete (the servers only prune at startup)
is swept: `git worktree remove --force` (`rm -rf` as the fallback) per dir
under this bridge's sessions root, then `git worktree prune`. Outcomes are
`deleted`, `absent`, `killed+deleted`, `deleted+swept`, `killed+deleted+swept`
(resolved, `live.Bridge.CONFIRMED`), `deleted?`/`absent?` (an earlier request
to the key ended in an ordinary timeout or connection loss and might still
land; reported once, a second teardown of the key says the plain word) or
`failed:<why>` (including `request-still-pending` when an abandoned request
is still in flight and `worktree-left(...)` when the sweep could not clear
the disk). Every outcome not in CONFIRMED is unresolved and listed at session
finish (`live.UNRESOLVED`). Note that httpx timeouts are per phase
(connect/read/write), not an absolute deadline, so every request also runs
under an absolute deadline of `timeout + 5` s in a worker thread (the worker
is abandoned on expiry, `Reply.error` says "deadline ... exceeded"), and
teardown treats `Reply.elapsed >= DELETE_DEADLINE` as a timeout. Each Bridge
keeps one `httpx.Client`; `live.close_all()` at session finish closes them,
which ends any abandoned worker still inside. Health of both ports is
printed before the first live story and after the run, never asserted.

Hermetic (no network, no container):

- `fake_bridge` — parametrized with ids `agy`/`codex`; a `FakeBridge` with the
  server module (`.module`), `.cli`, `.git`, `.clock`, `.client()` (a FastAPI
  TestClient; enter it with `with fb.client() as c:` to run the lifespan),
  `.expected_via` (`bell`/`notify`), `.worktree_dir(key, generation)`.
  - `fb.cli.answer("text", after=7.0)` — the next submitted turn answers that
    after 7 fake seconds (`fb.cli.default` for unscripted turns); `fb.cli.turns`
    records every submitted prompt; `fb.cli.live_sessions()` the fake panes.
  - `fb.git.refs` is `{ref: sha}`; `fb.git.advance("origin/main")` moves a ref;
    `fb.git.add_error = "..."` fails the next worktree add; `fb.git.added`,
    `.removed`, `.fetches` record what git was asked; `FakeGit.commit_of(dir)`
    reads the commit a fake worktree holds.
  - `fb.clock.now`/`.elapsed`; every `asyncio.sleep` inside the server modules
    (and `worktree`) advances it and yields once; `asyncio.wait_for` there
    times out on fake seconds (an idle awaited task jumps the clock to the
    deadline). Limits: lock waits, real subprocesses and the TestClient thread
    keep real time, so a story must not depend on real time passing.

Mark hermetic stories `@pytest.mark.hermetic` (on the class or module); a
hermetic story that requests `bridge`, `own_session` or `group_session` fails
at collection, before anything runs.

## Markers

- `@pytest.mark.hermetic` — no live bridge, never any live teardown.
- `@pytest.mark.units(n)` — live spawn-and-turn units the story spends (every
  turn counts: setup, follow-up, retry, respawn after clear/reset). Counted into
  the baseline per port.
- `@pytest.mark.group("B")` — the TestPRD section 4 group, on the class (or
  test). Fallback: a module-level `GROUP = "B"`.

## Running

```
.venv/bin/pytest stories -m hermetic -q            # host only, under 2 min
.venv/bin/pytest stories -q                        # live on both ports (+ hermetic)
.venv/bin/pytest stories -k "agy" -q               # one port
BRIDGE_RUN_STAMP=r1 .venv/bin/pytest stories -q    # pinned stamp (e.g. to find your sessions)
```

Nothing under `stories/` or in this package ever calls `POST /stop`. The
retained suites' live stop tests (`test_server.py`, `test_codex_server.py`)
are skipped unless `BRIDGE_LIVE_STOP=1` is set — only when the bridge is known
to be yours alone.

## Baseline file

Every run in which at least one live story ran writes
`logs/baseline/<YYYYMMDD-HHMMSS>-<stamp>.json` (`BRIDGE_BASELINE_DIR` to move
it). Hermetic-only runs, and runs where every live story was skipped, write
nothing. The reference every run is compared to (by story nodeid: changed
outcomes, missing and new stories, printed at session finish) is
`baseline/reference.json`, tracked in git (`BRIDGE_BASELINE_REFERENCE` to
move it); `BRIDGE_BASELINE_APPROVE=1` makes a fully green live run the new
reference, a run with any failure is refused. Schema:

- top level: `date`, `time`, `run_stamp`, `repo_prefix`, `container`,
  `image_id`, `wall_time_s`, `totals`, `stories`
- per story: `nodeid`, `agent` (`agy`/`codex`/null), `port`, `group`, `units`
  (declared), `ran` (the call phase executed), `units_spent` (the declared
  units, counted only when the call phase ran, whatever its outcome; zero when
  setup failed or skipped), `hermetic`, `outcome` (`passed`/`failed`/`skipped`/
  `error`), `duration`, `teardown_error` (only when present)
- `totals`: `stories`, `stories_per_group`, `live_stories_ran`,
  `live_units_per_agent` (sum of `units_spent` of the live stories, keyed by
  agent name), `outcomes`

## Environment knobs

`BRIDGE_HOST` (127.0.0.1), `BRIDGE_CONTAINER` (gemini-cli-rest-bridges-1),
`BRIDGE_RUN_STAMP` (pinned stamp, nonced), `BRIDGE_BASELINE_DIR`,
`BRIDGE_BASELINE_REFERENCE`, `BRIDGE_BASELINE_APPROVE`, `BRIDGE_LIVE_STOP`.
