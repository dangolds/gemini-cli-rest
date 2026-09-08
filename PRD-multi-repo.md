# Multi-Repo Sessions on the Existing Bridges — PRD

> **Status:** Draft — awaiting approval · **Date:** 2026-09-08 · **Owner:** dan · **Downstream:** Architect (technical design), then implementation
> **Builds on:** the shipped two-bridge deployment (agy on 8000, codex on 8001). **Defers:** `PRD.md` (one bridge, one port, shared base, adapter registry) stays the long-term plan; this document ships the repository part of it on today's two servers and leaves the rest untouched.
> **Context:** internal developer tooling, one operator, bridges on a private network. Live sessions hold real work and are never killed by a deploy without an explicit go.
> **Not in this doc:** code, config encoding, container layout, implementation order inside the servers — all follow in technical design.

---

## 1. Executive Summary

A developer working in any of the operator's projects asks a bridge agent about that project, and the answer is about that project's code at a known commit. Today that is only possible for one project: both bridges are wired to a single repository, `slitled`, and a second one has nowhere to go.

**This initiative makes the repository part of the session name.** A session is `<repo>/<name>@<base>`. The bridges hold a list of repositories, each with its own clone in a shared volume, its own fetch and its own worktree root. A name without a repository is refused with the list of known repositories, exactly as a name without a branch is refused today. Every answer states the agent, repository, remote, branch and commit it ran against.

**What stays as is:** one port per agent, two separate servers, no shared base, no third agent. The two servers are edited separately; the module they already share for git and worktrees carries the repository list.

**Where the repository choice is made:** in the consumer project, not by the model and not by the bridge. Each project carries a file at its root naming its repository; the skill reads it and builds the session name. The bridge only checks that the name exists. A wrong file is caught on the first answer, when the skill compares the echoed remote to the checkout's own.

Terms: a **consumer project** is a repository where a developer runs Claude Code; a **skill** is the instruction set that Claude Code follows when the developer asks it to consult a bridge, and it is what actually sends the requests; a **warm session** is a CLI process kept running between prompts so it remembers the conversation; the **base** is the branch a session is cut from; the **echo** is the set of fields every answer carries to prove where it ran; the **completion signal** is the CLI telling the bridge a turn finished, and has nothing to do with git.

---

## 2. Background and Problem

Current state, verified against the running system: `worktree.py`, imported by both servers, holds one clone path, one fetch lock, one coalescing timer, and cuts every worktree from that clone. Each server splits the session name at the first `@` into name and base, refuses a missing base with a conversational handshake, and hashes the full key into a filesystem-safe directory and tmux name. The clone lives in a named volume at a fixed path.

- **P1 — One repository, hardwired.** No second clone, no way to name one, and sessions from two projects would collide on name.
- **P2 — Nothing in the answer says where it ran.** A caller cannot tell which repository or commit an answer came from without asking the model.
- **P3 — One fetch for everything.** A slow remote blocks every spawn; with several repositories that is one slow remote blocking all of them.

Why now: more repositories are wanted and the bridges are freshly stabilized, with all three live suites green.

---

## 3. Goals and Non-Goals

- **G1.** Repository in the session name: `<repo>/<name>@<base>`, mandatory, no default.
- **G2.** A repository list in compose, one entry per repository, clones in one shared volume.
- **G3.** Isolation by construction: each repository has its own clone, fetch lock and worktree root; a session directory holds one repository's worktree and nothing of another.
- **G4.** Provenance on every spawn, answer and recovered answer: agent, repository, remote, base, commit, and whether the fetch behind that commit succeeded.
- **G5.** Unknown repository refused with the list of known ones, never a fallback.
- **G6.** Consumer projects declare their repository in a root file; the skills build the session name from it and verify the echo.
- **G7.** Zero regression: both live suites and the worktree suite pass with prefixed names, plus a live check of two repositories at once.

Non-goals:

- **N1.** One port, a router, a shared base, an adapter contract, a third agent. All of that is `PRD.md`, later.
- **N2.** A default repository. A bare `<name>@<base>` is refused.
- **N3.** Secrets, per-repository or global. Backlog item 10.
- **N4.** Pre-spawned pools, sandboxing between CLIs, write access to repositories, sessions spanning repositories.
- **N5.** Reviewing the caller's local working copy. A session sees a snapshot of the pushed branch, fixed at spawn; uncommitted or unpushed changes are invisible to it and the caller pastes them into the prompt, as today.
- **N6.** Live reconfiguration, and removing or renaming a repository or changing its remote while clones and sessions exist. The list is read at boot; a changed remote for an existing name is refused (see FR-2). Removal is outside the supported workflow: it takes removing the list entry, restarting, and deleting the clone and its worktrees by hand, because any clone still named in the list is recreated.
- **N7.** A drain procedure. A restart ends every warm session and every answer not yet retrieved. The operator gives the go when that is acceptable.
- **N8.** Fork checkouts. A project whose `origin` is a fork of the registered remote fails the echo check by design and is not supported.

---

## 4. Scenarios

- **S1 — Ordinary consult.** In project A's checkout the codex skill reads the root file, learns the repository is A, and calls `/chat/A/fix-login@main` on port 8001. The answer carries repository A, its remote, `main` and the commit. The skill compares the remote to the checkout's `origin` and continues.
- **S2 — Same name, two projects.** Project A and project B both open `review@dev`. The keys are `A/review@dev` and `B/review@dev`, two processes in two directories.
- **S3 — Same key, both agents.** `A/review@dev` is opened on port 8000 and on port 8001. Each bridge keeps its own worktree root and its own session table, so the two conversations, worktrees and recovered answers are independent; clearing one does nothing to the other.
- **S4 — No repository in the name.** A call to `/chat/review@dev` spawns nothing and answers with the handshake: name a repository, here are the known ones and the exact form to send.
- **S5 — Unknown repository.** A call to `/chat/Z/review@dev` spawns nothing and answers with the list of known repositories.
- **S6 — Adding repository C.** The operator grants the mounted SSH key read access, adds one compose entry naming C and its remote, restarts on an explicit go, and adds the root file to project C, which already carries the two consumer skills like every consumer project. The bridge clones C in the background after it is up; health shows C cloning, then available. A call naming C while the clone runs is refused with that state and told to resend shortly; nothing waits on a clone. Sessions on A and B are served meanwhile.
- **S7 — Repository C unavailable.** The clone fails or access is revoked before any clone exists. Health shows C unavailable with the reason; calls naming C are refused with that reason; the next call naming C starts the clone again in the background, at most once per coalescing interval, and is itself refused with the cloning state; a call landing while the interval blocks a retry is refused with the unavailable reason. A and B keep serving.
- **S8 — Fetch fails on an existing clone.** B's fetch times out, or B's access is revoked after the clone exists. A new spawn on B continues on the last successful snapshot and its echo marks the fetch as failed with the time of the last good one. Existing B sessions and their answers are untouched. A's spawns are not delayed.
- **S9 — Missing branch.** A call to `/chat/B/x@no-such-branch` is refused, as today; nothing spawns. The usual cause is a branch not pushed yet, so the skill's message says to push it or consult on a pushed base.
- **S10 — Audit.** An operator checks last night's answer: it says codex, repository B, a remote URL and a commit. That line is the evidence.
- **S11 — Follow-up and refresh.** A second prompt to `A/fix-login@main` lands in the same warm conversation on the same worktree, frozen at its spawn commit. To get newer code the caller clears the session: the conversation is discarded, the worktree is removed, the fetch step runs again and a new worktree is cut at its result under the same name. Delete ends the session and forgets the name; nothing is recreated.
- **S12 — Cutover.** The updated skills send prefixed names. Against the old bridge a prefixed name is just a longer session name on the single repository and no echo comes back, so the skills are merged first, right after the go and right before the deploy, and when no echo comes back they continue only if the root file names the original repository and stop otherwise, saying the bridge does not serve that repository yet. Only the original project works until the deploy. Sessions opened under old-style names before the merge are no longer reachable through the skills from that moment, so the merge happens in the same window as the deploy and those sessions count among the ones the operator deletes or accepts (FR-12). Then the bridge is deployed; old-style names get the handshake and the remote check becomes active. Rollback is redeploying the previous bridge image, which returns to that pre-deploy state: original project only, no echo.
- **S13 — Bad consumer file.** The file is missing, malformed, the checkout has no `origin`, or the developer is on a detached HEAD so there is no current branch to use as base. The skill stops with a message saying which of those it found and what to add. It never guesses.

---

## 5. Solution Overview

The shared git and worktree module grows from one repository to a table of them, keyed by name. Each entry owns a clone path under the shared volume, a fetch lock, a last-fetch record and a worktree root. Everything the module does today per bridge it now does per repository. Both servers keep their own code and gain the same small edits: split the repository off the front of the session name, pass it to the module, record provenance at spawn, extend health and the handshake.

| Concern | Where it lives | Change |
| ------- | -------------- | ------ |
| Repository list and clones | compose, shared volume, background clone after boot | new |
| Fetch, coalescing, timeout, worktree cut and removal | the shared git module, now per repository | extended |
| Session name parsing, handshake, health, provenance | each server, separately | small edits, twice |
| Repository declaration | a root file in each consumer project | new |
| Building the session name and checking the echo | the two consumer skills | edited |

Session name shape and the echo, in product terms:

```
/chat/<repo>/<name>@<base>        e.g.  /chat/A/fix-login@main
```

```json
{ "agent": "codex", "repo": "A", "remote": "git@github.com:org/a.git", "base": "main",
  "commit": "c5ec632", "fetch": "ok", "fetched_at": "2026-09-08T03:49:23Z" }
```

**Name trap:** the repository is everything before the first `/`, the base is everything after the first `@`, so a base such as `origin/dev` still works. The existing hashing of the full key keeps directories and tmux names safe.

**Provenance is captured at spawn and stored with the session.** Every answer and every recovered answer carries the values from its own spawn: the commit the worktree was cut from, the fetch that produced it, and its time. Later fetches for other sessions do not change them. Only refusals that have no session behind them, the handshake and the unknown-repository, unavailable-repository, cloning and missing-branch refusals, carry a reduced echo: the agent, the known repositories and the reason for the refusal. Everything with a session behind it carries the full echo: follow-up turns, recovered answers, and error answers from a running session such as quota exhausted. An answer fetched after the 180-second cap carries the same echo as if it had arrived in time.

**Freshness, unchanged but per repository.** A spawn fetches first, coalesced within a minute, capped by the timeout. A timed-out or failed fetch continues on the last successful snapshot with no age limit, and the echo says so. In the echo, `fetch` is the outcome of the refresh step considered at this spawn: `ok` (a fetch ran and succeeded), `coalesced` (a successful fetch from the last minute was reused), or `failed` (the step timed out or failed and the last successful snapshot was used); `fetched_at` is always the time of the last successful fetch at spawn. A session is frozen at its spawn commit until cleared. Fetch problems never touch existing sessions or their answers.

**Consumer side.** Each consumer project carries a root file named `.bridge.json`:

```json
{ "repo": "A" }
```

The skills walk up to the git root, read it, and build `<repo>/<name>@<base>` with the base taken from the developer's current branch, as today. Outside a declared checkout, or with a malformed file, no `origin` or a detached HEAD, the skill stops with a clear message. After each answer that carries a full echo the skill compares the echoed remote to the checkout's `origin`, normalized to host and path, and stops on a mismatch. A refusal carries no remote; its message is shown to the developer as is.

**Health** lists every repository with its clone state: available, cloning, or unavailable with the reason, alongside the active sessions.

---

## 6. Functional Requirements

**FR-1 — Repository list.** Compose declares the repositories as name and remote pairs. A name is one path segment: letters, digits, dash and underscore, no slash, no `@`. Both bridges load the same list at boot, validate names and refuse to start on a malformed entry.

**FR-2 — Clones.** After the bridge is up, every listed repository without a clone is cloned in the background into the shared volume, each with a timeout, while the others serve. The two servers share the volume, so a clone is guarded by a lock file in the volume: whichever server takes it clones, the other sees the repository as cloning and never clones the same directory; a stale lock older than the clone timeout is taken over. The current state always wins: while a clone runs, a call naming that repository is refused with the cloning state and told to resend, nothing waits on a clone; after success it is served; after failure the repository is unavailable with the reason. A call naming an unavailable repository starts one more clone attempt in the background, at most once per coalescing interval, and is refused with the cloning state; a call that lands while the interval blocks a retry is refused with the unavailable reason. A failed clone removes its half-written directory before releasing the lock, so the next attempt starts clean. Clone state, reason and last-attempt time live in the shared volume next to the lock, so both servers report the same health and honor the same interval. An existing clone whose remote differs from the list entry marks the repository unavailable with that reason and is never re-cloned or deleted by the bridge; the operator fixes the list entry (a restart, since the list is read at boot) or removes the clone on disk, after which the next call re-clones without a restart.

**FR-3 — Session names.** A session is `<repo>/<name>@<base>`. A missing repository or a missing base is refused with the handshake naming what is missing, the known values and the exact form to send; nothing spawns. An empty name is refused the same way. The name itself may contain any character except `@`; because a repository name has no slash, the first `/` always separates the two. An unknown repository is refused with the known names. A missing base branch is refused as today.

**FR-4 — Isolation.** Each repository has its own clone, fetch lock, coalescing timer and worktree root, and each bridge keeps its own worktree roots and session table. A session directory holds one repository's worktree. The agy launch passes only that repository's clone as its extra directory, the CLI option that widens what the agent may read beyond the worktree.

**FR-5 — Fetch per repository.** A spawn fetches its own repository, coalesced within a minute per repository, capped by the timeout. A timed-out or failed fetch continues on the last successful snapshot and reports it; one repository's fetch never delays another's; fetch problems never affect existing sessions.

**FR-6 — Provenance.** Spawn records agent, repository, the clone's actual remote, base, commit, fetch outcome and fetch time with the session. Every response with a session behind it, first turn, follow-ups and recovered answers alike, carries those recorded values. Refusals without a session carry the agent, the known repositories and the reason.

**FR-7 — Health.** Health lists every repository with its state and reason, alongside the active sessions.

**FR-8 — Consumer file and skills.** Each consumer project declares its repository in a root file, `.bridge.json`. The two skills read it from any subdirectory, build the session name with the current branch as base, stop with a specific message when the file, the checkout, the current branch or `origin` is missing or malformed, and, on every answer that carries a full echo, stop when the echoed repository does not equal the root file or the echoed remote does not match `origin`, both normalized to host and path so SSH and HTTPS forms compare equal. A refusal's message is shown to the developer instead. An answer with no echo at all (the bridge before this deploy) is accepted only when the root file names `slitled` and the checkout's `origin` is the `slitled` remote the skill carries for this purpose; otherwise the skill stops and says the bridge does not serve it yet (S12).

**FR-9 — Clear and delete.** Clear discards the conversation, removes the worktree, runs the same fetch step as any spawn (coalesced within a minute, stale snapshot on failure, reported in the new echo) and cuts a new worktree at whatever that step yields, under the same name. Delete ends the conversation, removes the worktree and forgets the name; nothing is recreated. Clear is the only way a session moves to a newer commit, and it replaces the session's recorded provenance for every turn that follows. Clear and delete both discard an answer not yet retrieved, including one still running past the 180-second cap; recovery is only for the session as it is.

**FR-10 — Behavior preserved.** The following stay exactly as today on both bridges. Quota exhausted: HTTP 429, the session is kept, the reset time is included when the CLI reports one. Model drift: the CLI switched model after a quota error; the bridge answers 409 and re-pins the model before the next prompt. Overload: the model reports it is at capacity; the bridge retries within the 180-second cap, then answers 503. Any other turn error: 502. The hard cap: a request returns within 180 seconds and a longer turn's answer is recoverable afterwards. The completion signal: the CLI tells the bridge a turn finished. Startup retry: a CLI that does not become ready in time is respawned once. Conversation wait: on the first turn the bridge keeps waiting, up to 90 seconds, while the CLI is visibly busy before giving up.

**FR-11 — Tests.** Offline tests for the module's per-repository behavior and each server's name parsing and provenance, marked hermetic. The live worktree suite gains a two-repository check: same session name on two repositories at once, correct clone, correct commit, no mixing, run against the real container. The release gate is the existing live suites passing with prefixed names plus that check, plus the failure outcomes in S4, S5, S7 and S8 exercised once, plus S3: the same key opened on both ports at once, one cleared, the other still answering and still recoverable, and the cleared one answering again on a fresh worktree with refreshed provenance. Through both real skills, once each: an ordinary consult (S1) with the echo check passing, the stops in S13 (missing file, malformed file, no `origin`, detached HEAD), a root file naming the wrong registered repository, which must stop on the remote mismatch, and, before the deploy, a project naming a repository other than `slitled` stopping on the no-echo rule. Delete is exercised once: after it the session is gone from health and the worktree is removed.

**FR-12 — Deploy.** One explicit go covers both the skill merge and the deploy, in that order and in one window. Before the go the operator looks at health on both ports and either deletes the listed sessions or accepts losing them, because the merge alone already makes old-name sessions unreachable (S12). No automatic drain (N7).

---

## 7. Phasing

- **Phase 1 — One working end-to-end flow.** The repository table in the shared module, the compose list and volume, background cloning, the codex server edits, and the two-repository live check against the real container. _Done when: two repositories serve the same session name at once through the codex bridge, with correct provenance._
- **Phase 2 — agy.** The same edits on the agy server, from the codex diff. _Done when: the agy and worktree live suites pass with prefixed names and the two-repository check passes on both ports._
- **Phase 3 — Consumers.** The root file and skill edits in the consumer projects, ready on a branch, not merged. _Done when: from that branch, the original project still consults both bridges against the not yet deployed bridge, and a project naming another repository stops on the no-echo rule (S12)._
- **Phase 4 — Cutover.** Explicit go after the operator has checked health on both ports (FR-12); then the skill merge, then the deploy, in one window. _Done when: the two-repository check and the consumer stops in FR-11 pass through the real skills against the deployed bridge._

The current repository is registered as `slitled`. The second repository for the live check is this bridge repository itself, registered as `bridge`. It costs nothing and is always available.

---

## 8. Success Metrics

- **M1** Adding a repository: zero code changes. The operator journey is one compose entry, SSH read access, one restart on an explicit go, and one root file in the project; the consumer skills are already installed there, as in every consumer project.
- **M2** Regression: the agy, codex and worktree live suites pass with prefixed names, and the outcomes in S3, S4, S5, S7 and S8 behave as written.
- **M3** Two repositories served concurrently with the same session name, correct clone and commit each, in the live check.
- **M4** Routing: on every answer that has a session behind it, the echoed repository equals the root file and the echoed remote equals the checkout's `origin`, normalized; zero mismatches. Refusals without a session are exempt.
- **M5** Ordinary spawn time unchanged: about five seconds from request to ready when the clone exists and the fetch succeeds, measured as today; first-time cloning is excluded.

---

## 9. Risks, Dependencies, Open Items

| Risk | Mitigation |
| ---- | ---------- |
| Old-style names break on deploy day | Skills merged first and compatible with both bridge versions (S12); the handshake tells any straggler exactly what to send; rollback is the previous image |
| A list entry points at the wrong remote | The echo carries the clone's real remote and the skill compares it to the checkout's `origin` on every answer |
| Clones grow the disk | One shared volume; the disk-cleanup backlog item applies per repository |
| The same edit is made twice, differently | Codex server first, agy second from its diff, both reviewed by both bridges; the shared module holds everything that can be shared without a refactor |
| A deploy kills live work | Explicit go only, after the operator checks health and deletes or accepts the listed sessions (FR-12); no drain (N7) |
| A background clone hangs | Per-clone timeout; the other repositories serve meanwhile; health shows the state |

Dependencies: the SSH key already mounted must have read access to every listed remote; the consumer skills live in the consumer repositories and are edited there.

Open items: none.
