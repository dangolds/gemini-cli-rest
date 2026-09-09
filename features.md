# Improvement backlog

Findings from the 2026-09-06 audit of the bridge logs, codex rollouts and
`/root/.codex/logs_2.sqlite`. One section per issue: what is wrong, the
evidence, and the intended fix. Tick the box when shipped.

## 1. codex `exec` tool broken in the container  — [x] DONE 2026-09-06 (code + review by both bridges + throwaway-container run; LIVE 2026-09-08: rebuilt, container healthy, agy 49/49, codex 30/30, worktree 11/11, codex 0.153.4 + code-mode host stamped)
- **Problem:** codex 0.15x runs shell commands through a separate helper
  binary, `codex-code-mode-host`. Only `codex` is installed, so every command
  the model tries fails: `failed to spawn code-mode host
  /usr/local/bin/codex-code-mode-host: No such file or directory`.
- **Evidence:** 147 errors in `logs_2.sqlite`, 36 September rollouts; release
  ships `codex-code-mode-host-x86_64-unknown-linux-musl.tar.gz`.
- **Fix:** download the helper tarball next to `codex` in `Dockerfile` and in
  the update block of `entrypoint-codex.sh`.

## 2. Empty codex answers are quota errors returned as 504  — [x] DONE 2026-09-06 (quota task_complete → /chat 429 `usage_limit` {message, resets_at, model, session}, /last 200 status=usage_limit; session kept alive; thread re-pinned via `codex resume <id> -m CODEX_MODEL` before the next prompt (flag + rollout `thread_settings_applied` drift); 409 `model_drift` if a turn's `turn_context.model` ≠ pin; offline tests in test_codex_quota.py; LIVE 2026-09-08: rebuilt, container healthy, agy 49/49, codex 30/30, worktree 11/11; quota/re-pin paths not hit live yet — no usage-limit error occurred)
- **Problem:** "Turn error: You've hit your usage limit ... try again at HH:MM"
  ends the turn with an empty answer. The bridge returns a generic 504 and
  `/last` reports done with 0 chars. Codex then silently switches the thread to
  `gpt-5.6-luna` medium, defeating the `CODEX_MODEL` pin.
- **Evidence:** 69 usage-limit errors since Aug 29, 63 empty 504s, 165 empty
  `/last` answers, 11 rollouts drifted model.
- **Fix:** treat an empty completion as failure; read the error from
  `logs_2.sqlite` / rollout error event; return a distinct status (429) with the
  reset time; dump diagnostics on empty completions; add effective model/effort
  (from `turn_context`) to every response.

## 3. `git fetch` in `worktree.resolve_base` has no timeout  — [x] DONE 2026-09-06
- **Shipped:** every git call in worktree.py is capped by `WORKTREE_GIT_TIMEOUT` (60s; process group killed on expiry), fetches are coalesced under a lock to one per `WORKTREE_FETCH_MIN_INTERVAL` (60s) per bridge process, a failed/timed-out fetch logs a WARNING (reason + elapsed) and the spawn continues on the existing `origin/*` refs; compose `GIT_SSH_COMMAND` gains `-o BatchMode=yes -o ConnectTimeout=15`. Live after next `--build`.
- **Problem:** fetch runs on every spawn from both bridges with no timeout;
  failures are swallowed. One request hung ~19h.
- **Evidence:** request 8e18b0a9, 2026-09-01 12:38 → 09-02 08:08 (502).
- **Fix:** `asyncio.wait_for` (30–60s) around the fetch, `-o ConnectTimeout`
  in `GIT_SSH_COMMAND`, one fetch per minute under a shared lock, log failures.

## 4. codex drops prompts, client only learns after the hard cap  — [ ]
- **Problem:** 49 "NEVER INGESTED" since Aug; clients polled `/last` up to 17
  times.
- **Fix:** report `never_started` as soon as no `task_started` appears within
  the stall window, with an explicit error so the client re-sends.

## 5. No in-flight guard on `/chat`  — [-] WON'T FIX (dan, 2026-09-06: the calling agent is smart enough to recognise a previous answer)
- **Problem:** after a hard timeout the next prompt can receive the previous
  turn's answer.
- **Fix:** in `send()`, if the session is still busy return 409 "turn still
  running, poll /last".

## 6. codex notify hook dies with E2BIG  — [x] DONE 2026-09-06 (fast path re-wired from the argv-based `notify` program to a codex `Stop` lifecycle hook that gets the event JSON on stdin — `-c hooks.Stop=[…]` + `--dangerously-bypass-hook-trust`, async, python hook writes ids-only lines to the same events.jsonl; `CODEX_NOTIFY_LEGACY=1` escape hatch; offline tests in test_notify.py incl. a 300 KB stdin run and the E2BIG repro; LIVE 2026-09-08: rebuilt, container healthy, agy 49/49, codex 30/30, worktree 11/11, 48 responses completed via the Stop hook)
- **Problem:** the payload is passed as argv; big turns exceed the arg limit
  (23×), bridge falls back to polling.
- **Fix:** hook reads the payload from stdin or writes only thread/turn ids
  (verify codex `notify` contract in docs first).

## 7. `service_tier` not pinned  — [x] DONE 2026-09-06 (CODEX_SERVICE_TIER=default in compose; both bridges LGTM; LIVE 2026-09-08: rebuilt, container healthy, agy 49/49, codex 30/30, worktree 11/11)
- **Problem:** config.toml in the volume drifted to `service_tier = "fast"`;
  only model/effort are pinned per launch.
- **Fix:** `CODEX_SERVICE_TIER` env passed as `-c service_tier=…` per launch.

## 8. codex hard-timeout rate 28% (agy 5%)  — [ ]
- **Problem:** clients block 180s then poll `/last?wait=30`.
- **Fix:** optional `?wait=N` on `/chat` (capped at hard timeout) so clients
  return early and poll `/last`.

## 9. 72% of sessions are single-turn  — [-] WON'T FIX (dan, 2026-09-08: a normal spawn is ~5s end to end — worktree add <1s, agy/codex ready marker 2-3s; the slow cases are agy post-login stalls, already covered by #15. A pool would go stale per branch and hold idle logins for a few seconds of saving.)
- **Problem:** every spawn pays fetch + worktree + CLI start + DELETE.
- **Fix:** small pool of pre-spawned sessions per branch handed out on first
  use (bigger feature, plan first).

## 10. Open network, no auth, `~/.ssh` mounted  — [ ]
- **Problem:** ports bound on 0.0.0.0, no auth, sandbox off. Anyone who can
  reach the ports gets a shell-capable agent.
- **Fix (requested by dan, 2026-09-06, NOT implemented yet):** require a
  header on every request, e.g. `X-Bridge-Secret`, checked in the access-log
  middleware of both bridges. A hardcoded secret is acceptable (env override
  optional): the odds of someone finding the server, knowing the payload shape
  and the value are near zero. `/health` should stay open for probes. Tests
  (`httpx` calls + offline `TestClient`), README/CODEX.md curl examples and the
  consumer skills must send the header. Later, optionally: bind `127.0.0.1`
  when local, deploy key instead of `~/.ssh`.

## 11. No compose healthcheck  — [x] DONE 2026-09-06 (both bridges LGTM; LIVE 2026-09-08: rebuilt, container healthy, agy 49/49, codex 30/30, worktree 11/11, compose reports healthy)
- **Fix:** `healthcheck` curling both `/health` endpoints.

## 12. Unbounded disk growth  — [ ]
- **Problem:** `/root/.codex` 1.4G (`logs_2.sqlite` 484M, 1285 rollouts), 501
  agy brain dirs, 200+ `[projects.*]` trust entries, 431 timeout dumps;
  `_rollout_files()` rglobs everything per poll.
- **Fix:** startup job deleting rollouts/brain dirs/dumps older than 7 days,
  trim trust entries, scope rollout scan to recent date dirs, add `procps` to
  the image, log CLI versions at startup.

## 13. `server.py` / `codex_server.py` ~85% duplicated  — [ ] PLANNED: merged with #14 into PRD.md (2026-09-08, draft reviewed by codex, awaiting approval)
- **Fix:** shared `bridge_common.py` base with thin agy/codex adapters (large
  refactor, plan first).

## 14. Support multiple repos, each with its own secret  — [ ] PLANNED: preceded by TestPRD.md (behavior suite on today's code, 7 review rounds, codex and agy Ready on round 7); then PRD-multi-repo.md (2026-09-08, nine review rounds, codex xhigh Ready on round 6, codex max-effort Ready on round 9, agy advisory; ships on the existing two servers, one port per agent, repo prefix in the session name, no secrets). The one-port/shared-base version stays in PRD.md for later
- **Ask (dan, 2026-09-06):** one Docker deployment should serve several
  repos, not only `slitled-platform`. Each repo gets its own secret so calls
  for different repos can never mix. The consumer skill stays the same, only
  the hardcoded secret differs per repo (suggestion).
- **Today:** a single `WORKTREE_REPO=/app/slitled-platform` clone, one volume,
  sessions keyed `<name>@<base>` cut worktrees from that one repo; no auth at
  all (see #10).
- **Sketch:** a repo registry in env/compose, e.g. `REPOS=slitled:/app/slitled-platform:<secret-a>,other:/app/other:<secret-b>`
  (or one env per repo), one named volume + clone per repo; the request
  carries `X-Bridge-Secret`, the bridge maps secret → repo and cuts the
  worktree from that clone; session keys become `<repo>/<name>@<base>` so
  names cannot collide across repos; `/health` lists repos; `git fetch`
  per repo. Both bridges share the registry. Depends on the header gate in
  #10 (the secret is what selects the repo, so #10 and #14 ship together).

## 15. Live-suite failures: agy startup stalls and codex capacity errors  — [x] DONE 2026-09-06
- **agy (`TestSpecialCharacters` 503 "startup timed out after 60s"):** agy's own
  per-process logs show it stalls when two of its processes initialize at the
  same instant (both write the same per-second `log/cli-<start-second>.log`,
  12:08:51) or when one starts 90ms after others were killed and are still
  shutting down ("Waiting for migrations to complete", 12:01:31: stuck right
  after "OAuth: authenticated successfully", `loadCodeAssist` never issued);
  once (11:28:33) a genuine Google network stall (`keyringAuth: timed out`,
  `dial tcp …:443: i/o timeout`). During this work's bridge review (12:46:32)
  a LONE start (35s after the previous instance had shut down) stalled the
  same way — nothing in agy's log for 58s after "OAuth: authenticated
  successfully" — so the stall also happens without any overlap, which is
  why a respawn is shipped on top of the serialization. The bridge spawned
  with no protection, and `tmux kill-session` returned before the process
  was gone.
- **Shipped (server.py):** a bridge-wide `_STARTUP_LOCK` (one agy in its
  startup phase at a time; git fetch stays outside it), `_kill` waits for the
  pane pid to exit (`AGY_EXIT_WAIT`, then SIGKILL of the process group), a
  startup-timeout dump (screen + tail of agy's process log) under
  `LOG_DIR/timeouts/`, and one respawn (`AGY_STARTUP_RETRIES`) before the 503
  now reads "(after 2 attempts)". Tests: `test_server.py::TestStartupResilience`.
- **codex (`TestChatMemory` 502 on turn 2):** the rollout's `task_complete.error`
  was `{"message": "Selected model is at capacity. Please try a different
  model.", "codex_error_info": "server_overloaded"}` — returned in 3.6s, not
  retried by codex, and classified by the bridge as a generic `error` → 502.
- **Shipped (codex_server.py):** `send()` retries that verdict (backoff
  `CODEX_OVERLOAD_BACKOFF` doubling, `CODEX_OVERLOAD_RETRIES`, only with
  `CODEX_OVERLOAD_MIN_BUDGET` of the hard timeout left; each retry's collection
  capped at the remainder), then `/chat` answers 503
  `{error: "server_overloaded", attempts, …}`; other terminal errors stay 502,
  `usage_limit`/`model_drift` are never retried. Tests:
  `test_codex_quota.py::TestOverload*`.
- **Follow-up (2026-09-06, agy 1.1.27 after the rebuild):** three `/chat` calls
  502'd "Could not determine agy conversation id (no new transcript appeared)"
  20.7s after "sending prompt" on sessions that had reached ready normally
  (2–15s). agy's own log for one of them: 14:36:37.030 "Streaming conversation
  99c95a76-…", 14:36:37.337 `loadCodeAssist`, then nothing until 14:36:57.632
  "Forwarding user message to conversation 99c95a76-…" — the same post-login
  backend call behind today's `⢿  Signing in...` startup stalls (>60s, already
  covered by the startup retry), with container→Google round trips at ~100ms.
  The bridge gave up at exactly `CONVERSATION_DETECT_TIMEOUT` (20s) although
  the process was alive and the screen showed a spinner. Caveat from the
  bridge log: in all three cases the test's DELETE (tmux kill) landed ~50ms
  after the 502, and agy's "Stream completed … Forwarding user message" lines
  followed the kill — so whether agy would have forwarded on its own is not
  proven; the new give-up dump is what settles that next time.
  **Shipped (server.py):** `_detect_new_conversation` treats the 20s window as
  the *expected* time; once it expires it keeps polling while the process is
  alive and the screen shows `BUSY_MARKERS` or `STARTUP_BUSY_MARKERS`
  (`Signing in`), one tmux capture per ~1s, up to `CONVERSATION_DETECT_MAX`
  (default = `RESPONSE_STALL_TIMEOUT`, 90s); an idle screen still gives up at
  20s as before, a dead process at once. Every give-up now writes
  `LOG_DIR/timeouts/<session>-detect-g<gen>-turn<n>.log` (screen + agy log
  tail, reason `conversation_detect_timeout`, path in the 502). The
  `_submit_first` half-window re-paste probe passes `extend=False` (it exists
  to learn quickly whether the re-paste landed). `send()` passes
  `_collect_response(hard_timeout=RESPONSE_HARD_TIMEOUT − elapsed)` (as the
  codex bridge does) so detection + collection never outlive the hard cap.
  Tests: `test_detect_conversation.py` (virtual clock: busy at 35s → id, idle
  → 20s give-up, busy forever → 90s + dump, `extend=False`, dead process).
- **Test-infra hazard fixed (2026-09-07):** the live-server fixtures in
  test_server.py / test_codex_server.py post `/stop` (kills EVERY live session
  on the bridge) after each non-hermetic test; the hermetic list was a hardcoded
  tuple of class names, and the new `TestStartupResilience` class was missing
  from it, so every offline test run stopped the owner's live sessions 13
  times. Hermetic classes are now tagged `@pytest.mark.hermetic` (registered in
  pytest.ini) and the fixtures key on the marker.
- **Live-verified (2026-09-08):** rebuilt with everything above, container
  healthy; agy 49/49, codex 30/30, worktree 11/11, zero warnings/errors in the
  bridge logs and no new timeout dumps. The earlier `TestSpecialCharacters`
  503s and the conversation-id 502s did not recur; none of the new recovery
  paths (startup retry, detect extension, overload retry, quota) had to fire.

## 16. A base with a dot (e.g. `release/1.2`) breaks the tmux session  — [x] fixed 2026-09-09 (worktree.tmux_safe_name; needs a container rebuild to deploy); found 2026-09-08 during the TestPRD Stage 0 review
- Both servers derive the tmux session name from `worktree.safe_name(key)`, which keeps dots. tmux 3.5a stores a session name with `.`/`:` replaced by `_` (`agy-x.y` is listed as `agy-x_y`) and then cannot resolve the dotted name as a target: `has-session -t agy-x.y` fails with "can't find pane: y". So every tmux call after `new-session` misses for such a session.
- Verified on a throwaway tmux socket in the container, not on a bridge socket. Fix: normalise `.`/`:` in the tmux name (server side) or exclude them in `safe_name`. The test suite's teardown and fake CLI already mirror tmux's real behaviour.
