# Behavior Test Suite for the Bridges — TestPRD

> **Status:** Draft — awaiting approval · **Date:** 2026-09-08 · **Owner:** dan · **Downstream:** test implementation, then `PRD-multi-repo.md`
> **Builds on:** the shipped two-bridge deployment (agy on 8000, codex on 8001) and its existing suites. **Precedes:** `PRD-multi-repo.md`; that change is not started until this suite is green on today's code.
> **Context:** the bridges work and hold real sessions. The point of this suite is that they keep working through the next change, and every change after it.
> **Not in this doc:** test code, fixture design, mocking technique — all follow in implementation.

---

## 1. Executive Summary

Before the repository change, the bridges get a suite that describes how they behave today, in stories a reader can check against the running system: given a session on a branch, when the branch is missing, then the bridge refuses and nothing spawns. The stories are about behavior, not code, so they survive the change. Every story runs against both bridges from one file, so any difference between agy and codex is a finding, not a surprise.

**Rule 1: green today.** The suite runs against the current container before any change. A story that fails today is reported as a finding and decided by the operator; it is never silently rewritten to pass.

**Rule 2: one name helper.** Every story, in the new files and the retained suites alike, builds its session name through one helper. Today it yields `<name>@<base>`; after the repository change it yields `slitled/<name>@<base>`. That one line is the migration of the suite, with one named exception: the few name-grammar stories that assert on what a slash in a name does today are marked as changing with the repository PRD and are updated with it.

**Rule 3: never touch sessions the suite did not create.** Names are unique per run, every session is deleted by the story or group that opened it, and the stop-everything endpoint is never called live: it is tested hermetically, and the existing live stop tests are moved behind an opt-in flag.

Terms: a **story** is one test, written given-when-then; **live** stories run against the container on both ports; **hermetic** stories run on the host with the CLI and git mocked, and need no container; the **cap** is the 180-second per-request limit after which the answer is recovered through the last-answer endpoint.

---

## 2. Background and Problem

Current state: two live suites (agy 49, codex 30) and six hermetic ones (worktree 33, bell 24, notify 38, verify gate 7, quota 69, conversation detection 13) exist and pass; the live pair takes about fifteen minutes, plus an eleven-step container smoke of the worktree feature. They were written change by change, so coverage follows the history of fixes, not the surface of the product.

- **P1 — Coverage by accident.** Memory, recovery, clear, reset, delete, stop, isolation and special characters are covered. Concurrency, the cap, long answers, the exact health shape and most name edges are not, or only on one bridge.
- **P2 — Two copies of the same story.** The agy and codex suites repeat the same tests by hand and have drifted: codex has 30 where agy has 49. Nothing says which differences are intended.
- **P3 — No baseline.** Nothing records what the container did on a given day, so after a change "it passes" cannot be compared to "it passed before".
- **P4 — A shared container.** Live suites already call the stop-everything endpoint, which ends every session on the bridge, including the operator's own, and a health check just before it cannot rule out a session the operator opens a second later.

Why now: the repository change touches session names, the git module and both servers at once. It is the first change that can break all three bridges' behaviors in one edit.

---

## 3. Goals and Non-Goals

- **G1.** One common story file, parametrized over both ports, for everything both bridges must do the same way. Per-agent files only for agent-specific behavior.
- **G2.** Every story in given-when-then form, named after the behavior, readable without the code.
- **G3.** Edge cases on every surface: names, prompts, branches, lifecycle order, concurrency, recovery, failure paths.
- **G4.** Green on today's container, with a saved baseline log per run, before any implementation starts.
- **G5.** The suite survives the repository change by changing one helper, and is rerun as the release gate of `PRD-multi-repo.md`.
- **G6.** Safe on a shared container: unique names, cleanup with a deadline, live stop-everything off by default and only under the opt-in flag.
- **G7.** Runtime of the new live stories under thirty minutes per port, the two ports run in parallel; the retained suites keep their own fifteen minutes and run separately.

Non-goals:

- **N1.** Tests for the new repository behavior (prefix, unknown repository, two repositories). Those are written with that change, from its PRD, and added to this suite then.
- **N2.** Performance benchmarks beyond the spawn-time check that already exists as a number.
- **N3.** Testing the CLIs themselves, tmux, or git. Only what the bridge promises over HTTP and on disk.
- **N4.** Changing container configuration for a test. Live stories use the container as deployed; anything that needs a knob turned is hermetic.
- **N5.** Rewriting the existing suites. They stay, keep running and count in the baseline; stories they already cover are referenced, not duplicated. The only edits to them: the name helper, the opt-in flag on their live stop tests, and any cleanup that relied on stop-everything replaced by deleting the run's own sessions.
- **N6.** Load testing. Concurrency stories use a handful of sessions, enough to prove isolation, not capacity.

---

## 4. Scenarios

Stories are grouped by surface. Each group lists the behaviors, the edges, and whether it runs live, hermetic, or both. Existing coverage is marked (existing) and not rewritten.

**A — Session names.** Both, hermetic on both servers plus the worktree module, live over HTTP.
- Plain name with base (existing); base with a slash such as `origin/dev` (existing); empty base means branchless (existing). Two `@` signs: over HTTP the route refuses with 422; the module, called directly, splits at the first (existing). The stories keep both levels apart.
- Empty name, name of only `@`, surrounding whitespace, a slash inside the name, unicode letters, a 200-character name, a URL-encoded slash: Phase 1 records today's outcome for each, the operator approves the list, and each story then asserts that one fixed outcome, identical on both servers. The exact raw inputs are probed hermetically, where no live session can be reached; the live probes keep the run stamp and cover only the accepted shapes.
- Session identity is the full key `<name>@<base>`: two keys that differ anywhere, including only in the base, are two sessions with two worktrees. Safe directory and tmux names: deterministic, distinct for distinct keys, collision-free for keys that differ only in case or in a trailing slash.
- Live probes over HTTP cover the accepted shapes: a plain name, a name with dashes and underscores, a name with a base, a base with a slash. Everything else in this group is hermetic.

**B — Lifecycle.** Live, both ports.
- First turn spawns and answers; second turn remembers (existing); clear forgets and respawns (existing); reset resets the turn count (existing); delete removes the session from health and its worktree from disk (existing).
- Order edges, live as one lifecycle chain per port on a single session: clear on a name that never chatted is 404, first turn, health during a running turn shows the session alive with the count not yet advanced, reset twice in a row, a follow-up turn right after the reset that records whether conversation memory and the cached last answer survived, then clear, delete, chat again under the same name yields a fresh session with turn count one. Clear and reset drop the worktree and cut a fresh one; delete removes it. Delete arriving between spawn and the first answer is hermetic, since the two happen inside one request. Clear and reset on a session whose CLI process has died behave differently today (clear refuses with 503, reset revives); both are asserted, and delete on a dead session removes it from health and its worktree from disk.
- During a running turn, clear, reset and delete wait for the turn to finish, then act. Hermetic for all three, where the ordering is provable. Live, delete only, on a follow-up turn: the turn's caller gets its answer, the delete succeeds, the end state is no session and no worktree; arrival order of the two responses is not asserted.
- Management endpoints without the base suffix: last, clear, reset and delete called with the bare name of a session opened as `<name>@<base>`, built with the helper's bare form, once with a single session under that name and once with two sessions on different bases; today's outcome recorded, approved and asserted.
- Health after each step: the session is listed with the right alive flag (false after its process died) and the right turn count, or absent; the required fields (status, active session count, session list with name, alive, turn count) are present with the right types and the count matches the list; extra fields are allowed.
- Unknown session on last, clear, reset, delete: 404 (existing); invalid name: 422 (existing). Malformed JSON body, missing prompt, non-string prompt, empty and whitespace-only prompt, and a `wait` that is negative or non-numeric: each outcome recorded and asserted, and none of them creates a session or a worktree. A `wait` above the maximum is clamped, not refused (Group C).

**C — Recovery.** Live, both ports; the cap itself hermetic.
- Last answer returns the completed answer (existing), the latest not an earlier one (existing), nothing from before a clear (existing).
- Last answer while a turn is running: with a wait long enough it returns when the turn ends; without a wait, or with a wait that expires first, it answers with the status the bridge uses for a running turn and no answer, the turn keeps running, and the answer is recoverable afterwards; the wait is capped at the configured maximum, and a wait on an already completed turn returns at once. The status for a prompt that never started is asserted where the bridge reports it.
- Every chat and last-answer response is checked for its required fields and their types, extra fields allowed, the same way as health. Last answer during a second turn, when a first answer exists: the story tells the earlier answer from the completion of the current one.
- The cap: a turn longer than 180 seconds answers 504, the session stays alive, the last-answer endpoint recovers the full answer afterwards, and a follow-up chat continues the same conversation. Hermetic only, with a mocked slow CLI and simulated time; no live story provokes the cap.
- Client gives up before the cap: the caller disconnects during a follow-up turn, not the first, since the first turn also does conversation discovery; the turn finishes on the bridge, the last-answer endpoint recovers it and a retried chat lands as the next turn, not a duplicate. Live on both ports. A disconnect during the first turn, while conversation discovery runs, is hermetic: the recorded outcome for the session, its worktree, the last answer and the next chat. Also hermetic: a new chat submitted after the first caller is gone, by cap or by disconnect, while that turn is still running; today's outcome, queued or refused, is fixed, and the story checks which prompts ran and which turn every returned or recovered answer belongs to.
- CLI death while a chat or a last-answer wait is pending: each pending request settles with today's recorded outcome, which for the last-answer wait is a 200 with no answer and a pending status, never a stale answer; health shows alive false; the next chat respawns and answers. Live smoke on both ports; the kill targets the session's own tmux pane or process id, never a process name.
- A last-answer wait already pending when clear, reset or delete takes effect, including recreation under the same name: hermetic; the waiter settles with the recorded result.

**D — Branches and worktrees.** Live, both ports; git failures hermetic.
- Branchless name gets the handshake and spawns nothing (existing); missing branch is a conversational 200 with no session (existing); base with a slash routes (existing).
- The worktree exists at the expected path, is detached, and its commit equals the clone's own `origin/<base>` reference, read inside the container before and after the spawn; either value is accepted when the two differ, since another session's fetch can move it. A remote listing is not used. That the reference advances and clear or reset follow it is proven hermetically with controlled references.
- Two sessions on the same base have two worktrees and do not share files: the harness writes a file into one worktree and checks the other, since the CLIs run read-only; clear, reset and delete then still succeed on that dirty worktree.
- Follow-up turns never move the commit; live, clear and reset respawn onto a commit equal to the clone's `origin/<base>` at that time; delete, clear and reset remove worktrees (existing); stop removing worktrees is hermetic (Group J).
- Worktree creation fails, or the CLI fails to start after the worktree was cut: hermetic; the HTTP result, no leftover worktree or half session, and a retry under the same name succeeds.
- A name addressed without a base while a session exists under `<name>@<base>`: the handshake, nothing spawned, the existing session untouched, because the bridge checks for a base before it looks at sessions. The same name with a different base: a second, distinct session (identity, Group A). A base given as a commit hash: today's outcome recorded, approved and asserted.
- The same key on both ports: two worktrees, two conversations, clearing one leaves the other answering and recoverable. This is the one story that talks to both ports at once; it lives in a small cross-port file, not the parametrized one.
- Fetch is coalesced within a minute and repeats after (existing, hermetic); a timed-out or failed fetch continues on the old snapshot with a warning (existing, hermetic).
- Stale worktrees left by a crash are pruned at startup, and a worktree that is not the bridge's own survives the prune (existing on codex, added on agy; hermetic, since the deployed bridges are never restarted). Bridge restart while a CLI is still alive in tmux: hermetic, recording today's behavior for that session, its worktree and its last answer, recovered or discarded.

**E — Concurrency.** Live, both ports.
- Three first turns on three names at once all answer, all listed in health, no cross-talk in the answers. Three is enough to prove isolation without becoming a load test; this group runs on one port at a time so the two ports never stack six CLIs on the container.
- Two turns on the same session at once: Phase 3 records whether the second waits or is refused; the operator approves; the story then asserts that one behavior. The story first confirms through health that the first turn is running, then submits the second. If it waits: both answers arrive, each matching its own prompt, and the conversation holds them in submission order; arrival order of the two HTTP responses is not asserted. If it is refused: the first completes, the refusal creates no turn, and a retry afterwards succeeds once.
- Delete while another session is mid-turn does not disturb it. Chat to a name being deleted: the orderings are controlled hermetically and each asserted; live, only the end state is checked, one consistent session and worktree or none, never a half-spawned one.
- Startup never overlaps for the same name (existing, hermetic).

**F — Prompts.** Live, both ports.
- Special characters, long prompt, very long single line (existing).
- Multi-line prompt with code fences, unicode and emoji, an inert literal that looks like a shell command inside a quote-back-verbatim instruction: all answered as text on both ports. A prompt containing the bridge's own completion marker text: hermetic, with the CLI faked, so the assertion is about the bridge's parsing and not the model's mood.
- Large prompts: live at one fixed size, about 50 KB, answered; no search for the limit.
- Every live response in this group is compared to what the CLI itself wrote in its transcript or rollout, whatever its length; a long-answer prompt is sent but no minimum length is asserted. The 4 KB boundary, agy's transcript repair, forced lengths and byte fidelity are hermetic with a fake CLI.

**G — Failure paths.** Hermetic on both servers unless marked.
- Quota exhausted: 429, session kept, reset time included when the CLI reports one; model drift: 409 and the model re-pinned before the next prompt; overload: retries inside the cap, then 503. All three exist only on codex and stay in the codex file (existing).
- Any other turn error: 502, both servers, as a hermetic chain: a good turn, a failed turn, a follow-up; the last-answer outcome and the turn count after the failure are recorded and asserted and the session stays usable. Startup timeout respawns once then succeeds, twice fails 5xx (existing on agy, added on codex). Conversation wait of 90 seconds on the first turn, then give up (existing detection tests, extended). Reset during a running turn: waits, then respawns (Group B).

**H — Completion signal.** Hermetic (existing bell and notify suites); live on both ports: the response's own completion field says the turn ended through the push signal, not the poll fallback; the expected token is per port, bell on agy and notify on codex.

**I — Logging.** Hermetic. Every request writes one tagged line with who asked what; every failure writes its reason; a timeout writes a dump under the timeout directory, with the screen and the log tail, and never fails the request itself (existing on agy, added on codex).

**J — Stop everything.** Hermetic only: every session ended, every worktree removed, health empty; stop while a chat and a last-answer wait are pending, recording whether it waits or interrupts and how each caller settles. The existing live stop tests, and any runner or fixture teardown that calls stop, move behind one opt-in flag the operator sets when the bridge is known to be theirs alone.

---

## 5. Solution Overview

| Piece | What it is | Runs |
| ----- | ---------- | ---- |
| Common stories | one file, every story parametrized over ports 8000 and 8001 | live |
| Cross-port story | one small file for the single story that talks to both ports at once | live |
| Agent stories | one file each for agy and codex, only behaviors the other bridge does not have | live or hermetic |
| Server stories | hermetic stories per server for failure paths and logging, mocked CLI and git | hermetic |
| Module stories | the existing worktree suite plus the new name and fetch edges | hermetic |
| Name helper | one function with three forms: full key, bare name for the no-base stories, raw for hermetic grammar probes; the prefix is the only line that changes later | both |
| Baseline log | one file per run: date, container image, per-story outcome and duration | live |

**Common before specific.** A story starts in the common file. It moves to an agent file only when the two bridges are meant to differ, and the reason is written in the story. Today's known difference: transcript repair exists only on agy; quota, drift and overload handling exist only on codex.

**Live stories use the container as deployed.** They read the repository path, ports and log directory from the same environment the existing suites use. They check disk state through the container, not the host. They never restart or reconfigure anything.

**Hermetic stories replace the CLI and git with fakes** and carry the existing hermetic marker, which skips any live teardown for them. For live runs the runner's stop-everything teardown is behind the same opt-in flag as the live stop tests and is off by default.

**Ownership is explicit.** A group whose stories do not depend on the turn count or on clearing (health shape, last-answer readback, prompt content) shares one session opened by a group fixture that deletes it at the end, also on failure; turns accumulate and no story in it asserts on the count. A story that mutates (clear, reset, delete, concurrency) opens its own session and deletes it through fixture teardown that runs on failure and cancellation too. Teardown has a deadline; if delete has not returned by then because a turn is still running, teardown kills only the pane or process id it verified belongs to a run-stamped name, then deletes again. Names carry the run stamp so two runs, or a run beside the operator's own sessions, never collide.

**The baseline is a file, not a memory.** Each live run writes its outcomes next to the log directory. The first green run on today's code is the reference; later runs are compared to it by story name.

---

## 6. Functional Requirements

**FR-1 — Story form.** Every new test is one behavior, named after it, with the given, when and then visible in its body. No story asserts on implementation details such as function names or internal state.

**FR-2 — Parity by construction.** Every common story runs on both ports from the same code. A story that passes on one port and fails on the other is a parity finding and is reported, not skipped.

**FR-3 — Name helper.** All session keys come from one helper, in one of its three forms. No story writes a key by hand; the raw form is allowed only in hermetic stories.

**FR-4 — Green on today's code.** The complete suite passes against the current container before the repository change starts. Stories that record today's outcome (Sections 4 A, B, D, E) are approved by the operator once and then assert one fixed value; alternatives never stay inside an assertion. A story that fails is a finding and blocks the gate until the operator decides: a known behavior, in which case the assertion is fixed to it, or a bug, in which case the story still asserts today's observed behavior so a regression elsewhere is caught, and a narrow companion story asserting the wanted behavior is marked expected to fail with the backlog item named. An expected failure that starts passing is reported.

**FR-5 — Isolation from live work.** Unique names per run, deletion of every opened session by its owner, no live stop-everything, no change to container configuration. Health before and after a run is compared as a sanity check, not as proof.

**FR-6 — Coverage.** Every group in Section 4 has its stories implemented; existing stories are referenced, not copied.

**FR-7 — Runtime.** Budget per port for the new live stories: at most thirty-two spawn-and-turn units, each counted at 45 seconds, twenty-four minutes, leaving six for checks and cleanup inside the thirty; shared group sessions and lifecycle chains are how the count stays under that. Every turn counts: setup, follow-up, retry, and the respawn after a clear or reset. The inventory is counted in Phase 1, before stories are written, and trimmed or moved to hermetic if it exceeds the budget. The ports run in parallel. The retained suites keep their own run. Every hermetic story that involves the cap, the conversation wait or the fetch interval uses simulated time; the hermetic suite finishes in two minutes.

**FR-8 — Baseline.** Every live run writes its per-story outcomes and durations to a file with the date and image identity.

**FR-9 — Survives the change.** After the repository change the helper is switched and the whole suite is expected green again with no other edit except the marked name-grammar stories (Rule 2); the new repository stories are added beside it.

---

## 7. Phasing

- **Phase 1 — Frame and names.** The common file with the port parametrization, the helper, cleanup, the baseline writer, and group A. _Done when: group A is green on both ports, a baseline file exists, and the live unit inventory is counted and under budget._
- **Phase 2 — Lifecycle, recovery, worktrees.** Groups B, C, D. _Done when: green on both ports; single-session lifecycle and recovery are complete._
- **Phase 3 — Concurrency and prompts.** Groups E, F. _Done when: the same-session concurrency story records today's behavior and all prompt edges pass._
- **Phase 4 — Failure paths, signals, logging, stop.** Groups G, H, I, J. _Done when: hermetic stories pass on the host, live smokes pass on both ports, the new live stories run under thirty minutes per port._
- **Phase 5 — Baseline and handover.** One full run, the baseline saved, findings listed. _Done when: the operator has decided every finding and the repository change may start._

Each phase: a fable subagent writes the stories, an opus subagent runs them against the live container, both bridges review the stories, the main session reviews last.

---

## 8. Success Metrics

- **M1** Every group in Section 4 has stories; the count per group is in the baseline file.
- **M2** Zero unexplained parity differences between the two ports; every explained one is written in a story.
- **M3** New live stories under thirty minutes per port including setup and cleanup, ports in parallel; the retained suites keep their fifteen; hermetic suite under two minutes.
- **M4** After the repository change: green again with the helper switched and no other edit to the suite.
- **M5** Zero sessions not created by the suite touched: no live stop-everything, every live name run-stamped, no request ever sent to a name the suite did not create. The health comparison before and after a run is printed for the operator, not asserted.

---

## 9. Risks, Dependencies, Open Items

| Risk | Mitigation |
| ---- | ---------- |
| The suite finds today's bridges disagree | That is the point; each difference is a finding for the operator, not a fix inside the suite |
| Live stories are flaky on a busy machine | Retries only on failures that happen before the request was delivered, such as connection refused; a chat whose outcome is unknown is recovered through the last-answer endpoint, a management request whose outcome is unknown fails the story; a story that needs timing precision is hermetic |
| Runtime grows past the budget | Read-only groups share one session; the unit count in FR-7 is checked in the baseline file |
| A recorded outcome depends on timing, not behavior | Orderings that depend on who wins are hermetic; live checks assert end state only |
| A story kills the operator's sessions | No live stop-everything; every story only touches its own run-stamped names |
| Recorded outcomes hide a later change | Each recorded outcome is approved once and then fixed in the assertion; a change fails the story |

Dependencies: the container running today's code, both ports up, access to the container for disk and reference checks.

Open items: the recorded outcomes (name edges, management endpoints without base, base as a commit hash, memory after reset, same-session concurrency) are found, not designed; whether each is the wanted behavior goes to the backlog after the baseline run.
