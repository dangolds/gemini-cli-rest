# Bugs

Every bug gets an entry here. A new or freshly fixed bug goes under **NOT VERIFIED**.
It is never verified on the day it is fixed. The deploy-day check only confirms
the fix is live. About a week later, check the logs of real traffic since the
deploy. If the bug did not reproduce, move it to **VERIFIED**. Keep checking
weekly and add a line to its check log each time. If it reproduces again, move
it back to NOT VERIFIED. Raw check output is saved in `logs/bugchecks/`.

## NOT VERIFIED

### B1: Reviews read the `dev` clone instead of the branch (2026-09-24)

- **Symptom:** gemini/codex reviews of a branch looked at `dev` code. agy often
  failed with "failed to read file: stat /app/slitled-platform/...: no such file"
  for files that exist only on the branch (81 errors since 2026-09-17).
- **Cause:** `docker-compose.yml` still had `AGY_EXTRA_ARGS` / `CODEX_EXTRA_ARGS=--add-dir /app/slitled-platform`
  from before per-branch worktrees. Both CLIs show every `--add-dir` to the model
  as a workspace root, and agy listed the `dev` clone first, so the model's first
  tool call often ran there. That affected 171 of 389 agy conversations and 48 of
  300 codex rollouts.
- **Fix:** removed both grants from `docker-compose.yml`; regression test
  `test_worktree.py::test_compose_does_not_grant_the_main_clone`. Reviewed by
  codex + gemini + fable (Ready, round 2). Deployed 2026-09-24.
- **Weekly check:** every command prints `0`.
  ```bash
  # agy: no recent process was given the clone as a workspace
  docker exec gemini-cli-rest-bridges-1 sh -c 'grep -ho "workspaceDirs=\[[^]]*\]" $(ls -t /root/.gemini/antigravity-cli/log/cli-*.log | head -5)' | grep -c /app/slitled-platform
  # codex: no recent rollout lists the clone as a workspace root
  docker exec gemini-cli-rest-bridges-1 sh -c 'grep -l "<root>/app/slitled-platform</root>" $(ls -t /root/.codex/sessions/*/*/*/rollout-*.jsonl | head -20)' | wc -l
  # offline guard
  .venv/bin/python -m pytest test_worktree.py -q -p no:cacheprovider -k main_clone
  ```
  Only count logs and rollouts from after the deploy (2026-09-24 ~00:00 UTC).
  Also scan agy transcripts since the deploy for tool calls on `/app/slitled-platform`
  (expected: only when a prompt names that path itself).
- **Check log:**
  - 2026-09-24, before deploy: agy 7, codex 20 (bug present, as expected).
  - 2026-09-24, deploy check (does not count as verification): container
    rebuilt, no `EXTRA_ARGS` in its env. A live `b1-verify-…@main` session on
    both bridges saw only its worktree: agy `workspaceDirs=[<wt> <wt>]`, codex
    one `<root>`. Both read `7cb764e1` = origin/main, not dev `b037be7b`. Output in
    `logs/bugchecks/B1-2026-09-24.txt`.
  - Next: weekly log check around 2026-10-01.

### B2: codex says "never started, re-send it" while turn 1 is running (2026-09-24)

- **Symptom:** after a first turn hits the 3-minute cap, `/last` on the codex
  bridge answers `status=never_started` ("codex dropped the prompt… re-send it")
  even though codex is working. A client that obeys re-sends on top of a
  running turn. 46 of 51 never_started reports in all codex logs were false
  (the turn answered later), all on turn 1. Example: `perf-view-…@dev`,
  2026-09-24 00:13–00:14, answered at 00:15:23 after 6m42s.
- **Cause:** `codex_server.py` `send()`, first-turn branch, reads
  `baseline_starts` from the rollout after binding it. The rollout is created
  when codex takes the prompt, so it already holds this turn's own
  `task_started`, which gives baseline 1 instead of 0. So "started past the
  baseline" is never true on turn 1. That causes the false never_started, and it
  also switches off the in-flight guard, so a long silent think on turn 1 can
  be cut off as "stalled" after 90s. The screen check doesn't save it: during a
  long silent reasoning step the busy marker isn't reliable (perf-view was in
  one 129s reasoning step at all three false reports). The bug has been there
  since `27e8478` (2026-06-22).
- **Fix:** `send()` first turn uses `baseline_starts = 0` (a fresh rollout holds
  nothing from before the prompt). Tests: `test_codex_rollout.py` B2 section
  (in-flight turn 1 → hard_timeout, not flagged; a real turn-1 drop still
  flagged; retried first turn), `test_last.py` (/last says `pending`),
  `test_notify.py` baseline `(0, 1)` → `(0, 0)`. On the old code the in-flight
  test gives `stalled` + flagged, and with the fix `hard_timeout` + not flagged.
  Plan and implementation approved by codex + gemini + fable. Deployed
  2026-09-24 01:55 UTC (image `0278c368`).
- **Weekly check:** prints `0` false alarms since the deploy (a NEVER_STARTED
  turn that later got a HIT).
  ```bash
  cd logs && cat codex-rest.log.* codex-rest.log 2>/dev/null | python3 -c "
  import sys,re; ns={}; hit=set(); SINCE='<deploy time, e.g. 2026-09-24 01:00>'
  for l in sys.stdin:
      if l[:16] < SINCE: continue
      m=re.search(r\"/last (NEVER_STARTED|HIT) session '([^']+)' ref=(\S+) turn=(\d+)\",l)
      if m: (ns.setdefault(m.groups()[1:],l[:19]) if m.group(1)=='NEVER_STARTED' else hit.add(m.groups()[1:]))
  print('false never_started:', sum(k in hit for k in ns), 'of', len(ns))"
  ```
- **Check log:**
  - 2026-09-24, before fix: 46 false of 51 (all history).
  - 2026-09-24 01:55, deploy check (does not count as verification): container on image
    `0278c368`, `baseline_starts = 0` present in `/app/codex_server.py`, both
    bridges healthy. Weekly check: set `SINCE='2026-09-24 01:55'`.
  - 2026-09-24 01:16 and 01:36, still on the old code (before the 01:55 deploy):
    `perf5-…` and `decodeconc-plan-…` turn 1 flagged NEVER INGESTED with
    baseline 1, and both were answered on the next `/last`. These are further
    proof of the bug, not a failure of the fix.
  - Next: weekly log check around 2026-10-01.

### B3: both bridges fetch the same clone at once and one fails on a ref lock (2026-09-24)

- **Symptom:** `worktree fetch failed after 3.5s (rc=1): error: cannot lock ref
  'refs/remotes/origin/SSRD-268-fixing-view-page': is at 3c319921… but expected
  ac5aa9…` (agy, 00:56:55). The client opened a codex and a gemini session for
  the same branch in the same second, so two `git fetch`es ran on
  `/app/slitled-platform` together.
- **Impact so far:** none. The other bridge's fetch won and moved the ref first,
  so both worktrees got the new tip `3c319921`. Risk: the losing bridge can read
  a ref that the winner hasn't written yet and cut a stale worktree.
- **Cause:** `worktree.py` coalesces fetches per bridge process only (one lock,
  `WORKTREE_FETCH_MIN_INTERVAL`), but agy and codex are separate processes on one
  clone.
- **Fix:** _(not fixed; low severity)_
- **Weekly check:** for every `worktree fetch failed` line since the fix, the
  session's worktree commit must equal the branch tip that the other bridge fetched.
  ```bash
  grep -h "worktree fetch failed" logs/agy-rest.log logs/codex-rest.log | cut -c1-120
  ```
- **Check log:**
  - 2026-09-24, found: 1 occurrence, no stale worktree.
  - 2026-09-24 01:05, 01:13, 01:33: 3 more, each time the client opened a codex
    and a gemini session on `SSRD-268` in the same second while the branch
    was moving (`138a2962`, `5b99aea4`, `608c5cde`). Each error shows the ref
    already at the new tip, so the losing bridge read the new commit.

### B4: a "branch not found" reply leaves a dead session entry (2026-09-24)

- **Symptom:** codex `/health` stays `degraded` because of
  `revert268-codex-…@SSRD-268-fixing-view-page` (`alive:false`, turn 0). The
  request came before the branch was pushed, so the bridge rightly answered
  "branch not found", but it kept a session record for it. The client deleted
  the gemini twin and not this one, so it stays until someone DELETEs it.
- **Cause:** the not-found path registers the session before the base is
  resolved, and the record is never dropped (both bridges; seen before with `ssrd238-…`).
- **Fix:** _(not fixed; cosmetic: affects /health only)_
- **Weekly check:** `/health` shows no `alive:false, turn_count:0` entries older
  than a few minutes.
  ```bash
  curl -s localhost:8001/health; curl -s localhost:8000/health
  ```
- **Check log:**
  - 2026-09-24, found: 1 on codex (`revert268-codex-1790209001-4242`). The
    2026-09-24 01:55 restart cleared it (not a fix).

### B5: codex's "approaching rate limits" popup eats the prompt and switches the model (2026-09-24)

- **Symptom:** 01:38:14 and 01:38:23, two codex sessions (`decodeconc-plan-…`
  turn 2, `codex-review-…` turn 3): the bridge logged "composer is EMPTY (codex
  consumed the paste) — re-pasting". The rollouts show 3× `thread_settings_applied
  → gpt-6-luna` before the turn started, and the turn ran on **gpt-6-luna medium**
  instead of gpt-6-astra xhigh. The bridge caught it (`model_drift`, withheld the
  answer, 409). The client re-sent, the bridge re-pinned astra, and the next turn
  was fine. Cost: a wasted turn and about 1–2 minutes per session.
- **Cause:** the account's 5-hour codex limit was at **99%** (rollout
  `token_count`, 01:37:02). codex 0.156.1 then shows an "Approaching rate
  limits" popup (Switch to a cheaper model / Keep current model / Keep current
  model (never show again)). The pasted prompt landed in the popup, and its
  Enter picked "switch". The bridge sets nothing to hide it: the binary has
  `hide_rate_limit_model_nudge`, and neither `codex_server.py` nor the volume's
  `config.toml` sets it.
- **History:** all 12 `model_drift` turns in the codex logs (from 2026-09-08
  on: 09-08, 09-17 ×2, 09-18, 09-19 ×6, 09-24 ×2) came 5–160s after a "codex
  consumed the paste", so this popup explains every model switch seen so far.
  None was a silent fallback at 100% usage, which is what CODEX.md used to claim.
- **Fix:** `codex_server.py` passes `-c notice.hide_rate_limit_model_nudge=true`
  on every launch, including the `resume` re-pin (kill switch
  `CODEX_HIDE_MODEL_NUDGE=0`). Verified: codex 0.156.1 type-checks the key (a
  string value makes it refuse to start), and the popup's guard reads the
  merged config, CLI overrides included (`rate_limits.rs:428`, per fable).
  Tests: `test_notify.py` (fresh + resume carry the flag, knob off omits it),
  `test_codex_quota.py` (the re-pin keeps it). CODEX.md re-pin paragraph
  corrected. Plan and implementation approved by codex + gemini + fable.
  Deployed 2026-09-24 05:05 UTC (image `5754c320`).
- **Weekly check:** no `model_drift` (and no usage-limit turn served by luna)
  right after a "consumed the paste" since the deploy.
  ```bash
  grep -h "ended with model_drift\|consumed the paste\|hit the USAGE LIMIT.*luna" logs/codex-rest.log | cut -c1-160
  ```
- **Check log:**
  - 2026-09-24, found: 2 occurrences (01:38), both recovered by the re-pin;
    12 in the logs since 2026-09-08.
  - 2026-09-24 05:05 UTC, deploy check (not verification): in the live
    container `_build_command()` carries `notice.hide_rate_limit_model_nudge=true`
    on fresh and resume launches. No codex session since the restart yet, so
    real traffic is unchecked. Next check ~2026-10-01.

## VERIFIED

_(none yet)_
