# Review Summary — TestPRD Stage 0 (test-suite foundation) vs main

Stage 0 of `TestPRD-tasks.md` (2026-09-08), working tree against `main`: `bridgetests/` (names, live client, baseline, fakes), `stories/` (conftest, helper tests, hermetic smoke, live smoke), minimal edits to `test_server.py`, `test_codex_server.py`, `pytest.ini`. Reviewed in 7 rounds by Codex (gpt-6-astra, xhigh, gate) and Gemini (agy, advisor, Ready on round 4), main session (fable) as final authority. 31 findings raised across the rounds, all fixed on the branch; Codex never issued a formal Ready, the last three low points were fixed without a further round by the operator's rule. Hermetic 72 marked stories plus 301 existing green; live smoke and retained live suites green on both ports. Round 8 (2026-09-09) fixed every item below; each heading carries its fix. Gemini: Ready (two low notes, both fixed). Codex: no verdict, the ChatGPT usage limit (429) blocked it; rerun after the reset. Hermetic 301 + 43 + 38 green; live stories 6/6 on both ports (incl. the two new teardown stories); retained live suite agy 49/49, codex blocked by the same 429. Item #1 is fixed in the server code but the container still runs the old code until the operator rebuilds.

## High

### 1. A base with a dot in it cannot get a working session on either bridge

**Fixed:** `worktree.tmux_safe_name` (dots/colons → `_`) used by both servers' `tmux_session`; hermetic dotted-base story in `stories/test_hermetic_smoke.py`; needs a rebuild to deploy.

**By:** codex (bridge), fable/xhigh (main, verified in the container)
**Where:** `server.py` (`AgySession.tmux_session`, `_target`), `codex_server.py` (`CodexSession.tmux_session`, `_target`), `worktree.py` (`safe_name` keeps `.`)
**Why:** tmux 3.5a stores a session name with `.`/`:` replaced by `_` (`agy-x.y` is listed as `agy-x_y`) and refuses the dotted name as a target (`has-session -t "=agy-x.y:"` answers "can't find session: agy-x.y"). Every tmux call after `new-session` misses for a key like `x@release/1.2`, so the turn fails and the pane is orphaned. Out of the suite's diff; filed as `features.md` #16. The suite's teardown and fake CLI already mirror the real tmux behaviour.
**Suggestion:** Normalise `.`/`:` to `_` in the tmux name on the server side (or exclude them in `safe_name`), then add the Stage 1 name-grammar story for a dotted base.

## Medium

### 2. The emergency pane-kill path has never run live

**Fixed:** `stories/test_common.py::TestTeardownPaths::test_verified_pane_kill_then_delete` proves the pane listing and verified kill live on both ports (the 200 s deadline itself cannot fire on a healthy bridge; that branch stays hermetic).

**By:** fable/xhigh (main)
**Where:** `bridgetests/live.py` (`Bridge.kill_verified_pane`, `Bridge.teardown_session` deadline branch)
**Why:** The branch that fires when a delete outlives the 200 s deadline (list panes, verify the run-owned name, `sh -c kill -9`, re-list, delete again) is covered by hermetic tests with a faked `docker exec` only. It is the one path that touches processes in the shared container, and no Stage 0 live turn outlived the deadline, so its real behaviour is unproven.
**Suggestion:** In Stage 1 group C (CLI death smoke) drive one run-stamped session past the deadline on purpose and assert the teardown outcome word `killed+deleted` and an empty pane list.

### 3. Leftover worktrees are reported, never removed

**Fixed:** teardown sweeps run-owned leftovers (`git worktree remove --force`, `rm -rf` fallback, `worktree prune`) under the sessions root only; outcomes `deleted+swept` / `killed+deleted+swept`.

**By:** codex (bridge), fable/xhigh (main)
**Where:** `bridgetests/live.py` (`Bridge._finish`: `failed:worktree-left(...)`), `server.py` / `codex_server.py` (prune only in the lifespan startup)
**Why:** A delete that fails between the manager pop and the worktree removal leaves a generation on disk. The suite flags it as unresolved but only a bridge restart prunes it, and the container is never restarted by the suite. On a long-lived container these accumulate under `/tmp/*-rest-sessions` until the next deploy.
**Suggestion:** Decide in Stage 3: either an operator-run `docker exec git worktree prune` step in the baseline runner, or a bridge-side prune endpoint (out of the suite's scope).

### 4. Uncertain and pending request branches of teardown are hermetic only

**Fixed:** `test_client_timeout_is_uncertain_until_rechecked` drives a client timeout live, waits through `/last`, asserts `deleted?` then `absent`; the `?` is reported once.

**By:** codex (bridge), fable/xhigh (main)
**Where:** `bridgetests/live.py` (`Bridge._request` deadline path, `_pending`, `_uncertain`, `Bridge.teardown_session` outcomes `absent?`/`deleted?`/`failed:request-still-pending`)
**Why:** The tracking of abandoned and client-timed-out requests exists to stop a late-landing chat from outliving cleanup. Neither branch has fired against the real container, so the interplay with the server's 180 s cap (client waits 185 s plus 5 s slack) is asserted only with fakes.
**Suggestion:** Same live story as #2, second half: a chat that hits the cap, then teardown; assert the outcome word and that the key is not listed afterwards.

### 5. Two client stacks: the retained suites keep their own helpers beside `bridgetests.live`

**Fixed:** retained `chat`/`last`/`cleanup` are thin wrappers over `live.Bridge` (`_bridge()`), test bodies untouched.

**By:** fable/xhigh (main)
**Where:** `test_server.py` (`chat`, `last`, `cleanup`, `_adopt`, `live_server`), `test_codex_server.py` (same names), `bridgetests/live.py` (`Bridge`)
**Why:** The retained helpers are gated and bounded now, but every ownership or deadline rule has to be applied twice, and the two already diverged once (the missing route check found in round 4). N5 of the TestPRD forbade rewriting the retained suites, so this is by design for now.
**Suggestion:** After Stage 3, move the retained suites onto `Bridge` in one mechanical pass (`chat` → `bridge.chat`, `cleanup` → `own_session`), then delete the local helpers.

### 6. Double cleanup in the retained suites

**Fixed:** `cleanup` tears down with verification and drops the key from `_created`; the module teardown only handles leftovers.

**By:** fable/xhigh (main)
**Where:** `test_server.py` (`cleanup` fixture, then the autouse `live_server` teardown over `_created`), `test_codex_server.py` (same)
**Why:** The per-test `cleanup` deletes first; the autouse registry teardown then re-deletes and reports `absent`, costing one health call and one docker exec per session. Correct, but noisy, and the extra calls count against the live budget.
**Suggestion:** Fold into #5; until then, accept.

### 7. No reference baseline is kept in git

**Fixed:** `baseline/reference.json` (tracked home, `BRIDGE_BASELINE_REFERENCE`), `BRIDGE_BASELINE_APPROVE=1` copies a fully green run there, session finish prints the by-nodeid comparison.

**By:** fable/xhigh (main)
**Where:** `bridgetests/baseline.py` (`BaselineRecorder.write` to `logs/baseline/`), `.gitignore` (`logs/`)
**Why:** FR-8 wants the first green run to be the reference every later run is compared to by story name. Baseline files stay local and untracked, so the comparison has no fixed anchor and would be lost on a fresh checkout.
**Suggestion:** Stage 3 decides the home: `baseline/` tracked in git (small JSON, one per release gate) or a pinned file path referenced from `features.md`.

### 8. Pinned-stamp uniqueness is enforced only at adoption time

**Fixed:** a pinned stamp gets a 4-hex nonce (`r1-ab12`); `matches_stamp` accepts the pinned prefix, `is_ours` stays exact.

**By:** codex (bridge)
**Where:** `bridgetests/names.py` (`STAMP` from `BRIDGE_RUN_STAMP`), `bridgetests/live.py` (`Bridge.assert_not_live`, `SessionRegistry.register`)
**Why:** Two concurrent runs sharing one pinned stamp are caught only if the other run's key already exists when this run adopts it; a key the other run creates later under the same name is owned by both and either teardown may delete it. Documented in the README as an operator rule.
**Suggestion:** Append a per-process nonce to a pinned stamp (`r1-<4 hex>`) and keep `matches_stamp` prefix-based so the operator can still find their sessions.

### 9. Abandoned daemon threads after a deadline

**Fixed:** one `httpx.Client` per `Bridge`, `close()`/`close_all()` at session finish end abandoned workers.

**By:** codex (bridge)
**Where:** `bridgetests/live.py` (`Bridge._request`)
**Why:** A request past its absolute deadline keeps running in a daemon thread until httpx gives up on its own; its late result is dropped and only the pending/uncertain bookkeeping tracks the side effect. Harmless for process exit, but a long test session can accumulate threads.
**Suggestion:** Use an `httpx.Client` per Bridge and close it on session finish so abandoned requests are torn down with the transport.

## Design

### D1. The fake clock does not reach `asyncio.wait_for`, lock waits or the TestClient thread

**Fixed:** `_AsyncioProxy.wait_for` runs on the fake clock (idle → jump to the deadline → cancel + TimeoutError); Lock waits and the TestClient thread still keep real time.

**By:** fable/xhigh (main)
**Where:** `bridgetests/fakes.py` (`FakeClock`, `_AsyncioProxy.sleep`), `bridgetests/README.md` (limits paragraph)
**Why:** Stage 1 group C (the cap, the conversation wait) and group G (startup timeout, overload retries) need timeouts to fire under fake time. Only code inside the patched modules sees the fake clock; a story that depends on `wait_for` or a lock timeout will either run in real time or hang.
**Suggestion:** Before Stage 1 T1.3/T1.7 start, add a `FakeClock.wait_for` proxy (advance the clock to the timeout when the awaited task is not done) and document which server waits are covered.

### D2. Group-session key is per class name, not per module

**Fixed:** `group-<module stem>-<class>`.

**By:** codex (bridge)
**Where:** `stories/conftest.py` (`group_session`), `bridgetests/live.py` (`GroupSession.default_name_for`)
**Why:** Two story modules with the same class name (`TestHealthShape` in a common and an agent file) under one stamp share one group key; the second class would adopt a live key and fail at `assert_not_live`.
**Suggestion:** Derive the default from `request.node.nodeid`'s module stem plus class name.

### D3. Per-session overhead is not counted in the FR-7 budget

**Fixed:** counted in `TestPRD-tasks.md` Stage 1 as a fixed per-session cost (about four docker execs per session).

**By:** fable/xhigh (main)
**Where:** `bridgetests/live.py` (`Bridge.assert_not_live`: one health and one pane list per adoption; `Bridge._finish`: one pane list and one worktree list per teardown), `TestPRD.md` FR-7
**Why:** The 32-unit budget counts spawns and turns; each session now also costs three or four `docker exec` round trips (about a second each). Over 32 units per port that is a minute or two inside the thirty.
**Suggestion:** Count it in Stage 1's inventory (T1.x live-unit column) as a fixed per-session cost, or batch the pane and worktree listings into one exec.
