# TestPRD — Stories and Tasks

> Breakdown of `TestPRD.md` into stages and tasks. Date: 2026-09-08. Status: Stage 0 DONE 2026-09-09 (11 rounds; debt of `TestPRD-stage0-review.md` fixed in round 8, codex's follow-ups in rounds 9-11, codex AND gemini Ready on round 11); waiting for the operator's go on Stage 1.

## Progress

- [x] Go for Stage 0 (2026-09-08)
- [x] Stage 0 written (2026-09-08): hermetic 11 + 41 green, smoke green on both ports, baseline file written, retained live suites green on both ports (47/49 agy, 28/30 codex: one stop test skipped behind the flag each, one name-shadowing bug found by the run and fixed)
- [x] Stage 0 reviewed (2026-09-08): 7 codex rounds (10+7+6+2+1+2+3 points, all fixed; the last 3 low ones without a further round), agy Ready on round 4; bridge bug found → features.md #16; tech debt listed in `TestPRD-stage0-review.md`
- [x] Stage 0 debt fixed (2026-09-09): all 12 items of `TestPRD-stage0-review.md` (round 8) plus 13 codex follow-ups (rounds 9-11, listed at the end of that file); round 11: codex Ready, gemini Ready; server fix (features.md #16) waits for a rebuild; verification run after round 11: agy retained 49/49, stories agy-side all green, codex side rerun after the quota reset: stories 6/6, retained 30 passed / 1 skipped; all green
- [ ] Stage 1 (waits for the operator's go)
- [ ] Stage 2
- [ ] Stage 3
> Rules: fable subagents write, one opus runner runs live suites, both bridges review every task's diff, main session reviews last. No container rebuild or restart at any stage.

## Stage 0 — Foundation (serial, one agent, blocks everything)

| Task | What | Done when |
| ---- | ---- | --------- |
| T0.1 | Name helper with three forms: full key, bare name, raw (hermetic only); run stamp built in | unit tests on the helper pass |
| T0.2 | Fixtures: port parametrization, group-shared session, per-story session, teardown with deadline and verified run-owned kill | one smoke story passes on 8000 and 8001 |
| T0.3 | Baseline writer: date, image id, per-story outcome and duration, unit count | a baseline file is written by the smoke run |
| T0.4 | Fake CLI, fake git, fake clock for hermetic stories | one hermetic story using all three passes |
| T0.5 | Retained suites: adopt the helper, stop tests behind the opt-in flag, stop-based cleanup replaced by own-session delete | retained suites pass hermetic; live pass on both ports |

## Stage 1 — Write the stories (parallel, one agent per task)

Each task delivers its hermetic stories green on the host, its live stories run once by the writer on both ports, and the count of live units it uses. Writers use their own run stamp, so they never collide with each other or with the operator. Only T1.5 and the full run are reserved for the runner.

| Task | Group | Live stories | Hermetic stories | Live units |
| ---- | ----- | ------------ | ---------------- | ---------- |
| T1.1 | A names | accepted shapes over HTTP | raw probes on both servers and the module, identity, safe names | 2 |
| T1.2 | B lifecycle | the chain (clear-404, first turn, health mid-turn, reset x2, follow-up, clear, delete, chat again), dead-session clear/reset/delete, bare-name management calls, malformed requests, live delete mid-turn | delete before first answer, mid-turn clear and reset | 8 |
| T1.3 | C recovery | last-answer readback in flight, short wait expiring, wait on a completed turn, disconnect on a follow-up turn, CLI death smoke | the cap, pending waits across clear/reset/delete, first-turn disconnect, chat after the caller is gone | 6 |
| T1.4 | D worktrees | path and commit vs the clone's ref, harness-written isolation and dirty cleanup, follow-up keeps commit, clear/reset re-cut, same key on both ports (cross-port file), base as commit hash | prune with a foreign worktree, restart with a live CLI, worktree creation and post-worktree startup failure, reference advancement | 8 |
| T1.5 | E concurrency | three sessions at once (one port at a time), two turns on one session, delete beside a running turn | chat-during-delete orderings | 5 |
| T1.6 | F prompts | shared session: fences, unicode, inert shell-like literal, 50 KB, long answer; every response compared to the transcript | 4 KB boundary, agy repair, marker text, forced lengths | 3 |
| T1.7 | G failures | none | 502 chain, codex startup timeout, conversation wait give-up, reset during a turn; codex quota/drift/overload referenced | 0 |
| T1.8 | H, I, J | completion token per port (inside the T1.2 chain) | codex timeout dump and log lines, stop with pending requests, stop removes everything | 0 |

Total live units: 32, the FR-7 budget. On top of the units, every live session costs three or four `docker exec` round trips (about a second each: one health and one pane list at adoption, one pane list and one worktree list at teardown), so a task with N sessions per port adds about 4·N seconds per port; count it in the inventory as a fixed per-session cost. T1.2 and T1.4 are the largest; split each into two agents if the schedule needs it (chain vs edge stories; live vs hermetic).

## Stage 2 — Record and approve (runner, then operator)

| Task | What |
| ---- | ---- |
| T2.1 | Runner executes every recording story on both ports and produces the outcome list: name edges, bare-name management, base as hash, memory after reset, same-session concurrency |
| T2.2 | Operator approves each outcome, or names it a bug |
| T2.3 | Writers fix the assertions to the approved values; bugs get the observed-plus-expected-to-fail pair with the backlog item |

## Stage 3 — Baseline

| Task | What | Done when |
| ---- | ---- | --------- |
| T3.1 | Full hermetic run on the host | green, under two minutes |
| T3.2 | Full live run, ports in parallel, E one port at a time | green, under thirty minutes per port, baseline file saved |
| T3.3 | Findings list to the operator: parity differences, failures, unit count vs budget | every finding decided |

## Stage 4 — Reviews (per task, not per stage)

Each Stage 1 task's diff goes to both bridges as it lands, then to the main session. Stage 0 is reviewed before Stage 1 starts.

## Concurrency plan

- Stage 0: one agent, about the size of one task.
- Stage 1: up to eight agents at once. All hermetic work is fully parallel. Live work is parallel too, since each writer has its own run stamp and one or two sessions; the container has handled that load in every review round so far. T1.5 is the only task that stacks sessions, so it runs alone.
- Stage 2 and 3: one runner agent, ports in parallel.
- Dependencies: Stage 1 needs Stage 0. T1.8's live token check lives inside T1.2's chain, so T1.8 lands after T1.2. Everything else in Stage 1 is independent.
