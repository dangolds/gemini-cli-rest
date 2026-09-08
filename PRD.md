# Multi-Agent, Multi-Repo Bridge — PRD

> **Status:** Draft — awaiting approval · **Date:** 2026-09-08 · **Owner:** dan · **Downstream:** Architect (technical design), then implementation
> **Builds on:** the shipped two-bridge deployment (agy on 8000, codex on 8001, one repository). **Supersedes** backlog items 13 (server dedupe) and 14 (multi-repo) in `features.md`, which this document merges into one effort. Backlog item 10 (network gate) stays separate.
> **Context:** internal developer tooling, one operator, bridges on a private network. Live sessions hold real work and are never killed by a deploy without an explicit go.
> **Not in this doc:** code, adapter interfaces, config schema, container layout — all follow in technical design.

---

## 1. Executive Summary

Today two bridges front two CLI agents. Each has its own port and its own 2,500-line server, and the two servers are the same program written twice. Both are wired to one repository. Adding an agent means copying a server; adding a repository is not possible; every fix lands twice or drifts.

**This initiative turns the two bridges into one bridge with a registry.** One process, one port. Every request names the agent and the repository it wants, in plain words. Any agent can work on any repository. Each repository has its own clone, its own fetch, its own worktree root. Every answer states which agent, repository, remote, branch and commit it ran against. Adding an agent is one adapter file plus one config entry; adding a repository is one config entry. The design is sized for ten CLI products and five repositories.

**The governing rule: the calling model never assembles a route.** The repository name comes from a file at the root of the project the caller is standing in, and the agent name comes from the skill invoking the bridge. Both are trusted inputs; the bridge checks them against its registry and refuses anything it does not know. Nothing is inferred or auto-selected.

What does not change: warm CLI sessions kept alive in tmux; the 180-second HTTP cap with the answer recoverable afterwards from the same session; sessions keyed on a branch and cut as detached read-only worktrees from the pushed tip; a fetch before spawn, coalesced within a minute as today; completion signalled by the CLI, not by polling; quota and model errors reported as errors, never worked around; the CLI-specific code that took weeks of live failures to get right, which moves unchanged.

Terms used throughout: a **warm session** is a CLI process kept running between prompts so it remembers the conversation; the **base** is the branch a session is cut from; **model drift** is the CLI silently switching to another model after a quota error; a **kill switch** is a config flag that disables one optional behavior; **completion push** is the CLI telling the bridge a turn finished, unrelated to git.

---

## 2. Background and Problem

Current state, verified against the running system: two FastAPI servers, `server.py` for agy and `codex_server.py` for codex, each on its own port, each holding the full tmux lifecycle, worktree gating, endpoints, timeouts, logging and response cache. Roughly 85 percent of the code is shared by copy. Prompts are pasted into the CLI's terminal; answers are read back from the CLI's own transcript files, not from the screen. One repository is hardwired through a single environment variable. A consumer skill picks an agent by picking a port and has no way to say which repository it means.

- **P1 — One port per agent.** Ten agents would be ten ports, ten compose entries, ten health checks, and ten places for a skill to get a URL wrong.
- **P2 — One repository, hardwired.** The clone path is a single environment variable. A second repository has nowhere to go, and sessions from different repositories would collide on name.
- **P3 — Two copies of one server.** Every backlog fix is done twice. Three items in this quarter's backlog were implemented in both files and reviewed in both. The copies have already drifted once.
- **P4 — Routing by the calling model's judgment.** Nothing in the request states the repository, nothing in the answer proves it, and the only guard is the calling agent typing the right URL.

Why now: a third agent is wanted, more repositories are wanted, and four open backlog items (prompt-drop detection, early-return wait, disk cleanup, network gate) would each be written a third time if the copy is made again. The bridges are freshly stabilized: all three live suites pass with zero warnings, so the migration starts from a known-good baseline.

---

## 3. Goals and Non-Goals

- **G1.** One port. The agent and the repository are names carried in the request body, never encoded in the address.
- **G2.** Any agent works on any repository. The bridge does not police the pairing.
- **G3.** Deterministic routing. The repository name comes from a file in the calling project; the agent name comes from the skill. The calling model types neither.
- **G4.** No mixups. Each repository has its own clone, fetch lock and worktree root, and a session is cut from exactly one of them. This is about routing correctness, not confinement: the CLIs run with full access inside the container, as today.
- **G5.** Proof in every answer: agent, repository name, remote URL, base branch and commit, so a caller can verify by string comparison, not by asking.
- **G6.** Adding an agent touches one adapter file and one config entry. Adding a repository touches one config entry. The shared base is never edited for either.
- **G7.** One base. CLI-specific behavior lives only in adapters; everything else exists once.
- **G8.** Zero regression. Each migrated agent passes its existing live suite before the next one moves. The two current ports keep working as aliases until the consumer skills switch.

Non-goals:

- **N1.** Per-repository secrets and per-repository agent allowlists. Rejected: all callers are the operator's own skills, and the echo in every answer is the audit trail.
- **N2.** Pre-spawned session pools. A normal spawn costs about five seconds; the slow cases are upstream login stalls already handled by startup retry.
- **N3.** The bridge choosing an agent, load-balancing, or falling back to another agent. Every call is explicit.
- **N4.** A rewrite from a blank page. The base is extracted from the codex bridge, and adapter code is moved, not reauthored.
- **N5.** Multiple users or tenants, and deployment outside the private network. The network gate is backlog item 10 and follows this effort.
- **N6.** Write access to repositories. Worktrees stay detached and read-only; the bridge never pushes.
- **N7.** Sessions spanning repositories. One session, one repository. Consulting about another repository means running the skill from that repository's checkout.
- **N8.** Reviewing the caller's local working copy. A session sees the pushed tip of the base branch only. Uncommitted or unpushed changes are invisible to it; the caller pastes them into the prompt, as today.
- **N9.** Sandboxing the CLIs from each other or from the container.

---

## 4. Scenarios

- **S1 — Ordinary consult.** A developer working in project A's checkout invokes the codex skill. The skill reads the project file, learns the repository is A, and sends one request naming codex and A. The answer arrives with repository A, its remote, the base branch and the commit it was cut from. Local uncommitted edits are not on the bridge side; the skill pastes the diff into the prompt.
- **S2 — Two opinions.** The same prompt goes to agy and codex as two requests to the same port, differing only in the agent name. The skill weighs both, as today.
- **S3 — Wrong repository name.** A request names a repository the registry does not hold. The bridge refuses it and lists the valid names. No process starts, nothing is guessed.
- **S4 — Same session name, two repositories.** Two projects both open a session called `review` on the same day. The sessions are distinct because the key carries the repository and the agent.
- **S5 — Follow-up on a warm session.** A second prompt to an existing key lands in the same conversation on the same worktree, frozen at the commit it was spawned on. A newer commit is picked up by clearing or deleting the session and spawning again.
- **S6 — Agent number ten.** A new CLI is added by installing it in the image, logging it in once, writing its adapter and one registry entry. No port, no compose change on the caller side, no edit to the base. Health lists it.
- **S7 — Repository number five.** A registry entry names the repository and its remote. The bridge clones it on the next boot into the shared clones volume and cuts worktrees from it thereafter.
- **S8 — Audit.** An operator wonders whether last night's answer came from the right code. The saved answer states repository B, its remote and a specific commit. That line is the evidence.
- **S9 — Quota exhausted on one agent.** codex hits its usage limit. Its session stays alive and the caller receives the quota error with the reset time. agy and the other agents are untouched.
- **S10 — One repository unavailable.** Access to repository C is revoked, or its first clone fails. Health marks C unavailable with the reason; requests naming C are refused with that reason; every other repository keeps serving. A fetch that times out on a healthy repository continues with the last snapshot and says so in the echo.
- **S11 — Migration in flight.** An old skill still calls port 8001 with no names. The alias maps that port to codex on the default repository. The skill is switched at leisure.

---

## 5. Solution Overview — One Bridge, Two Registries

The bridge holds two tables loaded at boot and validated before it serves. Changes take effect at the next restart; one operator, explicit go, as for any deploy.

| Registry | Entry holds | Added by |
| -------- | ----------- | -------- |
| Agents | name, adapter, model pin, effort, service tier, timeouts, kill switches, max concurrent sessions | one config entry plus one adapter file, plus the CLI installed and logged in |
| Repositories | name, remote | one config entry; the clone lands in the shared clones volume |

Every request carries both names in the body. The shape, in product terms:

```json
{ "agent": "codex", "repo": "A", "session": "fix-login", "base": "main", "prompt": "..." }
```

Session identity becomes `A/codex/fix-login@main`. The repository and agent are part of the key, so names cannot collide across either. Recovery, status, clear and delete address a session by the same four names. Worktrees live under a per-repository root and are cut from that repository's clone at the remote tip of the named base. Fetch runs before spawn, coalesced so spawns within a minute share one fetch, with a hard timeout and a per-repository lock. **Freshness trap:** a spawn is normally up to a minute behind the remote, and older when the fetch times out (see FR-6); a session is frozen at its spawn commit until cleared.

| Capability | Base (once) | Adapter (per agent) | Config | Consumer skill |
| ---------- | ----------- | ------------------- | ------ | -------------- |
| Endpoints, request validation, registries | **Owns** | — | — | — |
| tmux session lifecycle, kill and exit wait, startup retry | **Owns** | supplies the launch command and ready marker | timeouts | — |
| Repository clone, fetch, worktree cut and removal | **Owns** | — | repository entries | — |
| Prompt delivery into the live CLI | — | **Owns** | — | — |
| Reading the answer back from the CLI's transcript | — | **Owns** | — | — |
| Completion push (bell, hook) | receives events | **Owns** the signal | kill switches | — |
| Quota, model drift, overload classification | maps to status codes | **Owns** the detection | model pin, retries | reacts to the status |
| Hard cap, answer recovery, response cache | **Owns** | — | cap value | polls after a cap |
| Echo of agent, repo, remote, base, commit, last fetch | **Owns** | — | — | compares to its constants |
| Choosing the repository | — | — | — | **Owns**, from the project file |
| Choosing the agent | — | — | — | **Owns**, by name |
| Bridge host address | — | — | operator's client config | reads it |

**The adapter contract in one breath.** An adapter answers six questions about its CLI: how to launch it, what its ready marker is, how to hand it a prompt, how to read the answer back, how it signals completion, and which of its errors mean quota, model drift or overload. A capability the CLI lacks, such as effort or service tier, is declared unsupported in its registry entry and shown as such in health. **Adapter trap:** the six answers are where all the hard-won behavior lives; the two existing adapters are the current agy and codex functions moved as they are. "Headless first" applies to new agents only: where a new CLI offers a headless mode that keeps a warm session, its adapter uses it instead of the terminal path.

**Consumer side.** Each consumer repository carries a small file at its git root:

```json
{ "repo": "A" }
```

One generic skill walks up from the current directory to the git root, reads that file and sends the request. The skill for a given agent differs only in the agent name it sends. The bridge host comes from the operator's client configuration, not from the project file, so the same checkout works against a local or a remote bridge. A skill run outside a declared checkout stops with a clear message rather than guessing.

**Caller-visible contract, unchanged from today.** One turn at a time per session; the caller sends the next prompt only after the previous turn reports done. The bridge does not guard against a second prompt on a busy session (owner decision: the calling agent is expected to check); recovery returns the latest completed turn together with its turn number, so a caller can tell which prompt it belongs to. Clearing or deleting a session discards any turn still running. An HTTP call returns within the cap; a turn that runs longer keeps running, and its answer is fetched afterwards by session. The minimum error outcomes on every agent: unknown agent or repository is refused before anything runs; quota exhausted keeps the session alive and carries the reset time when the CLI reports one, otherwise says it is unknown; model drift is refused and the session is re-pinned before its next prompt; overload is retried within the cap and then reported; any other turn error is reported as such. The caller retries a prompt only when the bridge says the turn never started.

**Discovery.** Health returns the agent list with each agent's configured model and effort and its readiness, and the repository list with each repository's availability. The two current ports stay as aliases, each meaning one agent on the default repository, until every skill sends the names.

---

## 6. Functional Requirements

**FR-1 — Registries.** The bridge loads an agent registry and a repository registry at boot, validates every entry, and refuses to start on a malformed one. A repository whose clone fails or whose remote refuses access is marked unavailable with the reason and does not block the others. Health lists both registries with that status.

**FR-2 — Named requests.** Every request names an agent and a repository in the body. Either name missing from its registry is refused with the list of valid names. A port alias supplies both names for requests that carry neither; a request on an alias port that carries names uses the names.

**FR-3 — Per-repository isolation.** Each repository has its own clone under one shared clones volume, its own fetch lock and its own worktree root. A session directory holds one repository's worktree and nothing of another repository.

**FR-4 — Session keys.** Sessions are keyed on repository, agent, name and base together. Two sessions differing in any one of the four are distinct processes in distinct directories. A repeated key continues the warm conversation on the worktree it was spawned on; clear and delete are the only ways to move it to a newer commit.

**FR-5 — Echo.** Every spawn and every answer states the agent, the repository name, the clone's actual remote URL, the base branch, the commit the worktree was cut from, and when the last successful fetch happened. The remote is read from the clone, not from the registry entry.

**FR-6 — Freshness.** Every spawn fetches its repository first, subject to the one-minute coalescing window and the fetch timeout, and cuts the worktree from the remote tip of the named base. A missing base is refused, never substituted. A timed-out fetch continues with the last successful snapshot, however old; the echo carries the time of that last successful fetch so the caller can see the staleness. No age limit is enforced.

**FR-7 — Adapter boundary.** All CLI-specific behavior lives in the agent's adapter. The base contains no agent name and no CLI command string. A new agent adds a file and a config entry and edits nothing else; whether that holds is measured on the third agent, not assumed.

**FR-8 — Config-driven knobs.** Model pin, effort, service tier, timeouts, retry counts, kill switches and the maximum concurrent sessions are per-agent config, not code and not process-wide environment variables.

**FR-9 — Capacity.** Each agent has a configured maximum of concurrent sessions. A spawn beyond it is refused with a clear message; existing sessions are never evicted to make room.

**FR-10 — Consumer file and skill.** A consumer repository declares its bridge repository name in a root file. The skill finds the file from any subdirectory, reads the name and sends it. Outside a declared checkout the skill stops rather than guessing.

**FR-11 — Failure semantics preserved.** The error outcomes listed in Section 5 hold on every agent, with one status code per outcome fixed in the Phase 0 spec. A quota error never kills the session and never triggers a model change; the reset time is present when the CLI reports it and marked unknown otherwise. An agent whose login has expired reports that as an error on spawn; other agents are unaffected.

**FR-12 — Hard cap preserved.** No request blocks longer than the cap. The answer stays recoverable afterwards from the same session for every agent, and a recovered answer belongs to the turn that produced it.

**FR-13 — Aliases.** The existing ports remain valid throughout the migration as a documented exception to FR-2, mapped to one agent each on the default repository. They are retired only after every consumer skill calls the single port with names, verified by the alias ports logging zero requests over a full week.

**FR-14 — Deploy safety.** A planned rebuild or restart happens only on an explicit operator go, after health shows zero active sessions. Active means every warm session, idle ones included, because an unretrieved answer lives only in its session and dies with it. A crash is different: the container healthcheck restarts the process automatically, and the sessions are lost the same way. This holds during the migration phases as for any deploy.

**FR-15 — Tests.** Each adapter has an offline suite with the CLI mocked and a live suite against the running container. Mocked tests carry the hermetic marker so they never touch live sessions. A migrated agent is done when its live suite matches today's count, and the repository phase adds a live check of two repositories served concurrently through the real consumer skill.

---

## 7. Phasing

Each phase ends with the live suites green; the next phase does not start otherwise. Existing code moves, it is not rewritten. Every phase deploys with an explicit go and ends live sessions like any rebuild.

- **Phase 0 — Design spec.** The third CLI is chosen first, so the adapter contract is designed against three real CLIs, not two. Registry shape, adapter contract, session key, request and echo shapes, consumer file, alias rules. A short written spec, reviewed by both bridges, approved by the owner before code.
- **Phase 1 — The base, with codex on it.** The shared base is extracted from the codex bridge; codex becomes the first adapter and the agent registry exists with one entry. Port 8001 becomes the alias. _Done when: codex live suite 30 of 30._
- **Phase 2 — Repositories, proven on codex.** The repository registry, the shared clones volume, prefixed session keys, the echo, the consumer file and the generic skill. The current repository is the default so nothing existing breaks. _Done when: two repositories serve concurrently through the real skill with the same session name and no collision._
- **Phase 3 — agy joins.** The agy functions move into the second adapter; the agy server is deleted; port 8000 becomes the alias; both agents serve on one port. _Done when: agy live suite 49 of 49 and worktree suite 11 of 11 on the single process, plus the two-agent, two-repository live check._
- **Phase 4 — Third agent.** The first agent built purely on the contract. If it needs a base edit, the edit is made and the contract corrected; the zero-edit goal is then measured on the fourth agent. _Done when: its offline and live suites exist and pass._
- **Phase 5 — Retire aliases.** Every skill calls the single port with names; after a week of zero alias traffic the ports collapse to one. Backlog item 10 follows on this footing.

---

## 8. Success Metrics

- **M1** Adding an agent: one adapter file, one config entry, zero base edits, measured on the third agent.
- **M2** Adding a repository: one config entry, zero code changes, zero compose changes.
- **M3** Regression: agy 49, codex 30, worktree 11 live tests pass after each phase, matching today's counts.
- **M4** Routing: on every skill call the echoed agent equals the one the skill sent, the echoed repository equals the consumer file's declaration, and the echoed remote equals the checkout's `origin` after normalization (host and path only; scheme, user and `.git` ignored). Checked by the skill; zero mismatches.
- **M5** Concurrency: two repositories in use at the same time with no cross-talk in the Phase 2 live check; two agents on two repositories at the same time in the Phase 3 live check.
- **M6** Spawn time unchanged: about five seconds from request to ready in the ordinary case.
- **M7** Duplication: one base instead of two servers; each adapter holds only the six answers.
- **M8** Deploy safety: zero live sessions lost to an unrequested restart.

---

## 9. Risks, Dependencies, Open Items

| Risk | Mitigation |
| ---- | ---------- |
| The migration breaks behavior that only shows against the real CLIs | One agent per phase, moved not rewritten, gated on the same live suites that pass today |
| The terminal path gets more fragile with every adapter | Headless mode for new agents where the CLI has one; the adapter boundary confines the damage to one agent |
| A registry entry points at the wrong remote | Boot-time validation; the echo carries the clone's actual remote and the skill compares it, normalized, to the checkout's `origin`, so a wrong entry fails on the first call. A checkout whose `origin` is a fork of the registered remote fails that check by design |
| Five clones grow the disk | One shared clones volume; the disk-cleanup backlog item applies per repository |
| A deploy kills live work | Explicit go only, health shows active sessions first |
| The consumer file is missing or wrong in a project | The skill stops instead of guessing; the remote comparison catches a wrong name on the first call |
| One shared process means one shared failure | Per-agent session limits and per-repository availability keep a refused or unavailable agent or repository from blocking requests for the others. Memory or disk exhaustion still takes the whole process down; that is accepted for one operator, and recovery is the automatic crash restart with all sessions lost (FR-14) |

Dependencies: consumer skills in each repository must read the file and send the names; a shared clones volume in compose; the SSH key already mounted must have read access to every registered remote; each CLI installed in the image and logged in once; the third agent's CLI chosen before Phase 0.

Open items: which CLI is the third agent and whether it has a warm headless mode (must close before Phase 0 starts); a drain step that stops intake and waits for idle before a restart (lean: not now, the explicit-go rule covers one operator); whether the bridge host lives in a per-machine client config file or an environment variable on the caller (lean: file, one line, checked into nothing).
