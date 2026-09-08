# Codex CLI REST Bridge

A REST API that wraps OpenAI's **Codex CLI** (`codex`) in interactive mode,
exposing named multi-session chat with conversation continuity. It is a sibling
of the agy/Gemini bridge (`server.py`, see [README.md](README.md)) and exposes
the **identical REST contract**, just on port **8001** instead of 8000.

Use it to get a second opinion from codex's current flagship model alongside Gemini.

## How it works (same architecture as the agy bridge)

Both bridges now share one model: each named session runs its own **live** CLI
process inside a detached **tmux** session that acts as a terminal-emulator
mediator. The bridge drives the TUI by pasting the prompt as a tmux bracketed
paste and pressing Enter, and reads the model's reply as structured data from
the CLI's own on-disk transcript — never by scraping the raw escape stream.

| | agy bridge (`server.py`) | codex bridge (`codex_server.py`) |
| --- | --- | --- |
| Process model | one **live** `agy` per session, hosted in tmux | one **live** `codex` per session, hosted in tmux |
| Continuity | the warm process holds context in memory | the warm process holds context in memory |
| Transcript read | `brain/<id>/.../transcript.jsonl` | `~/.codex/sessions/.../rollout-*.jsonl` |
| Completion push (fast path) | terminal bell via tmux `alert-bell` hook | codex `Stop` hook (stdin JSON) → `events.jsonl` |
| "Done" detection (fallback) | new DONE model step + idle screen (debounced) | a new `task_complete` event (definitive) |
| Context wipe (`/clear`) | respawn in a fresh project dir | respawn in a fresh working dir |
| On crash/restart | next turn starts fresh | next turn starts fresh |

**Reading the reply is cleaner for codex.** Its rollout records explicit turn
boundaries — `task_started{turn_id}` → `response_item`s → `task_complete{turn_id,
last_agent_message}`. The arrival of a *new* `task_complete` is a definitive
end-of-turn signal (no screen-scrape debounce), and `last_agent_message` is the
final answer. A turn that has *started* but not yet *completed* counts as
in-flight, so a long pure-reasoning turn that writes nothing for ~90s is never
mistaken for a stall.

**Completion push (fast path).** The bridge launches codex with a `Stop`
lifecycle hook — `-c 'hooks.Stop=[{hooks=[{type="command",command="<hook>",async=true,timeout=10}]}]'
--dangerously-bypass-hook-trust` — pointing at a small Python script it writes
to `CODEX_NOTIFY_DIR/notify-hook.py` at startup. codex runs it the moment a
turn completes with the event JSON on **stdin** (session_id, turn_id, cwd,
last_assistant_message, …); the script appends one compact ids-only line
(`{"type":"agent-turn-complete","thread-id":…,"turn-id":…,"cwd":…,"ts":…}`) to
`CODEX_NOTIFY_DIR/events.jsonl`, writes nothing to stdout and always exits 0,
so it can never block or steer a turn (it is registered `async`, so codex does
not wait on it either). The bridge checks that small file every
`CODEX_RESPONSE_FAST_POLL` (0.3s) and, on a matching new event, reads the
rollout once and returns — the rollout stays the single source of truth for
the answer text. The `task_complete` polling above is retained as fallback: a
missed notification degrades to slower (the full check runs every ~3s), never
to hung. `/chat` responses report the path taken in an optional `"via"` field:
`"notify"` on the fast path, otherwise the legacy exit reasons
(`"rollout_done"`, `"stalled"`, `"hard_timeout"`).

*Why a `Stop` hook and not codex's `notify` program:* `notify` passes the whole
payload — the full answer included — as ONE argv string, and Linux caps a
single argv string at 128 KiB (`MAX_ARG_STRLEN`). On every answer above that,
codex's own spawn of the hook failed with `Argument list too long (os error 7)`
(visible only in codex's `~/.codex/logs_2.sqlite`: `run_legacy_after_agent_hook:
after_agent hook failed; continuing … hook_name=legacy_notify`) and the push
was silently lost. A hook defined through session flags has no persisted trust
hash (codex lists it `"trustStatus": "untrusted"` and a `trusted_hash` passed
the same way is ignored), hence `--dangerously-bypass-hook-trust` — the hook
is the bridge's own file. Note the flag is process-wide: any other untrusted
hook codex discovers (`config.toml`, a checkout's `.codex/`) runs too, which
is in line with a process already launched with approvals and the sandbox
off. The
`-c hooks.Stop=…` override replaces any `Stop` hooks from `config.toml` for
the bridge's codex processes only. `CODEX_NOTIFY_LEGACY=1` restores the old
argv-based `notify` wiring (same script, same log) as an escape hatch. (`[tui]`
notifications / OSC 9 is *not* used — it emits nothing in a detached tmux
pane.)

**Why a warm process** (rather than the simpler `codex exec` per turn): it keeps
conversation context in-process across turns and mirrors the agy bridge
one-to-one, so a single skill drives both bridges identically.

## Configuration (auto-approve + xhigh reasoning)

`entrypoint-codex.sh` seeds `/root/.codex/config.toml` on first run (only if
absent) so the interactive TUI never blocks on an approval or sandbox prompt:

```toml
model = "gpt-6-astra"              # current flagship (2026-09); bump when a new one lands
model_reasoning_effort = "xhigh"   # single-agent ceiling; max/ultra (4 parallel agents) burn more
approval_policy = "never"          # never pause for approval
sandbox_mode = "danger-full-access" # the container is the sandbox
```

Do not rely on that seed for the model or effort. It only runs when `config.toml` is
absent, and the file lives in the persistent `codex-config` volume — so once
the volume exists the seed never runs again, while the TUI keeps rewriting the
file whenever the model or effort is switched (it had silently drifted to
`medium`, and `service_tier` to `fast`). `CODEX_MODEL` is therefore passed as
`-m …`, `CODEX_EFFORT` as `-c model_reasoning_effort=…` and `CODEX_SERVICE_TIER`
as `-c service_tier=…` on every launch, which beats the file every time.

Every `codex` process is also launched with
`--dangerously-bypass-approvals-and-sandbox` as belt-and-suspenders — this also
preempts the TUI's "do you trust this directory?" prompt for the per-session
working dirs. The bridge additionally dismisses any other startup interstitial
(model NUX, tips) defensively, and pointedly does **not** auto-confirm an
"Update available" prompt (its default button runs `npm install`); keep the
image on the latest codex so that prompt never appears.

## Run it

Both bridges run in a **single container** (the `bridges` service in
`docker-compose.yml`): agy on :8000 and codex on :8001, started by
`entrypoint-all.sh`. If either bridge process dies the container restarts and
brings both back.

```bash
docker compose up -d --build                              # one image, one container, both bridges
docker exec -it gemini-cli-rest-bridges-1 codex login     # one-time INDEPENDENT codex login (persists in the codex-config volume)
curl -s http://localhost:8001/health                      # codex     (agy is on :8000)
```

**Auth is independent per environment — never shared.** The container logs in
on its own (the `codex login` above), and that login lives in the `codex-config`
volume. **Do not bind-mount or copy the host's `~/.codex/auth.json` into the
container.** codex's ChatGPT-plan OAuth refresh token *rotates* (single-use), so
if the host and container share the same token family, whichever side refreshes
first revokes the other's session — both then fail with "refresh token was
revoked". Multiple independent logins on the same plan are allowed, so log into
the host and the container separately. (For a fully unattended container,
OpenAI recommends API-key auth instead of the interactive login.)

**Auto-update on (re)start.** `entrypoint-codex.sh` re-fetches the latest codex
standalone binary on every container start (best-effort — a network blip never
blocks boot), and `entrypoint.sh` runs `agy update` similarly. Keeping codex
current also means the TUI's "Update available" prompt never appears. Keep the
host on a matching codex version (`codex update`) to avoid version skew.

To let codex read your repos, mount them under `/repos` in the `bridges` service
and set `CODEX_EXTRA_ARGS=--add-dir /repos` (mirrors agy's `/repos` setup).

## API

Identical to the agy bridge, on `http://localhost:8001`:

| Action | Command | Purpose |
| --- | --- | --- |
| Send message | `POST /chat/{session}` | Send prompt, creates session if new |
| Recover answer | `GET /last/{session}?wait=N` | Re-read the last COMPLETED answer (no re-ask) when a `/chat` response was lost |
| Clear context | `POST /clear/{session}` | Start a fresh codex session (true wipe) |
| Reset | `POST /reset/{session}` | Same as clear (codex has no process to reboot) |
| Delete session | `DELETE /chat/{session}` | Forget the session |
| Health check | `GET /health` | List all active sessions |
| Kill all | `POST /stop` | Drop ALL sessions |

Example:

```bash
curl -s -X POST http://localhost:8001/chat/review \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Review this design for race conditions."}'
```

## Usage limit (429), model drift (409) and the re-pin

A turn that dies on the ChatGPT usage limit still *completes* in codex's eyes:
the rollout gets a `task_complete` with a null answer and an `error`
(`codex_error_info: "usage_limit_exceeded"`, message "You've hit your usage
limit … or try again at 2:12 PM."). The bridge classifies that turn instead of
returning an empty 504:

* `POST /chat` answers **429** with a JSON `detail`:
  `{"error": "usage_limit", "message": "<codex's message>", "resets_at":
  "<ISO-8601, container-local zone, or null when unsure>", "model":
  "<model that served the turn>", "pin": "<CODEX_MODEL>", "session": "<name>"}`.
  `resets_at` handles both the time-only form (today, tomorrow if already
  past) and the dated form ("Aug 20th, 2026 7:21 AM"). The turn is logged at
  WARNING and dumped under `LOG_DIR/timeouts/` like a timeout.
* `GET /last` answers **200** with `done: true`, `status: "usage_limit"`,
  `response: ""` plus the additive `error` (the message) and `resets_at`
  fields — never a plain done-with-empty-text.
* **The session stays alive.** Nothing is retried and no reset credit is
  used: a human resets the usage (or waits for `resets_at`) and simply sends
  the next prompt to the same session.

**Re-pin on resume.** After the quota error codex silently switches the thread
to a fallback model (`gpt-5.6-luna` medium — visible as a
`thread_settings_applied` event and the next `turn_context`), and no config
flag prevents it. Before the next prompt the bridge therefore checks both its
in-memory flag (set by the 429/409 turn) and the rollout (a
`thread_settings_applied` naming a model other than `CODEX_MODEL` since the
last completed turn — so a bridge restart changes nothing), and when either
says so it re-pins: kills the codex process (its exit is confirmed, since codex
holds a per-thread lock), relaunches in the same tmux session with
`codex resume <thread-id>` plus every usual launch flag (`-m CODEX_MODEL`,
effort, tier, notify hook, `--add-dir`), waits for the TUI, then pastes the
prompt as usual. `resume` appends to the same rollout, so the conversation and
the answer read-back continue unchanged. Logged as `re-pinning session X: model
drifted to Y, resuming thread Z with -m <pin>`. With `CODEX_MODEL` empty
(no pin) drift detection and the re-pin are off.

**Belt and braces (409).** If a completed turn's `turn_context.model` is not
`CODEX_MODEL`, its text is withheld: `/chat` answers **409** with
`{"error": "model_drift", "model": "<seen>", "pin": "<CODEX_MODEL>", …}`, `/last`
reports `status: "model_drift"`, and the next prompt re-pins first — re-send
it. (Any other terminal `task_complete.error` with no answer text is a **502**
`{"error": "error", "message": …}` rather than an empty 504.)

**Capacity (503).** The one transient error is retried by the bridge: a
`task_complete.error` with `codex_error_info: "server_overloaded"` ("Selected
model is at capacity. Please try a different model." — codex returns it in ~3s
and does not retry) makes the bridge sleep `CODEX_OVERLOAD_BACKOFF` (doubling)
and re-submit the same prompt as a new turn on the same thread, up to
`CODEX_OVERLOAD_RETRIES` times while at least `CODEX_OVERLOAD_MIN_BUDGET` of the
call's hard timeout remains; each retry is logged at WARNING. Once that is
exhausted (or no budget is left) `/chat` answers **503** with
`{"error": "server_overloaded", "message": …, "attempts": N, "session": …,
"model": …, "pin": …}` — try again later. `/last` reports such a turn as
`status: "error"` with the message. The usage limit is never retried.

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `CODEX_CMD` | `codex` | Path to the codex binary |
| `CODEX_RESPONSE_HARD_TIMEOUT` | `180` | Absolute max per turn — a request never blocks longer (3 min) |
| `CODEX_RESPONSE_STALL_TIMEOUT` | `90` | Give up after this long with no progress (idle, rollout not growing) |
| `CODEX_NOTIFY` | `1` | Use a codex `Stop` hook as the fast done-signal; set to `0` to disable and revert to pure rollout polling |
| `CODEX_NOTIFY_LEGACY` | `0` | Set to `1` to wire the hook as codex's argv-based `notify` program instead (loses the push on answers > 128 KiB: E2BIG) |
| `CODEX_NOTIFY_DIR` | `/tmp/codex-rest-notify` | Directory holding the hook script (`notify-hook.py`) and the `events.jsonl` it appends to |
| `CODEX_RESPONSE_FAST_POLL` | `0.3` | Seconds between checks of the notify events file while a turn is in flight |
| `CODEX_RESPONSE_FULL_CHECK_EVERY` | `10` | Run the full fallback poll (rollout + liveness) every Nth notify-check wake (~3s) |
| `CODEX_STARTUP_TIMEOUT` | `60` | Max wait for the TUI to reach its idle prompt |
| `CODEX_REPIN_EXIT_TIMEOUT` | `15` | Re-pin: how long to wait for the killed codex process to exit (it holds the thread lock) before SIGKILL and `codex resume` |
| `CODEX_OVERLOAD_RETRIES` | `2` | Re-submits of the same prompt (new turn, same thread) after a capacity error (`codex_error_info: server_overloaded`, "Selected model is at capacity"), which codex itself gives up on in seconds; `0` disables. Never applies to the usage limit or a model drift |
| `CODEX_OVERLOAD_BACKOFF` | `5` | Seconds slept before the first capacity retry; doubles per retry (5, 10, …) |
| `CODEX_OVERLOAD_MIN_BUDGET` | `45` | A capacity retry runs only if the `/chat` call still has at least this many seconds of its `CODEX_RESPONSE_HARD_TIMEOUT` left after the backoff; the retry's own collection is capped at that remainder, so the whole call never outlives the hard timeout |
| `CODEX_SUBMIT_REPASTE_MAX` | `2` | Re-pastes allowed when codex consumed the paste (composer empty, nothing ingested); duplicate-guarded against the rollout; `0` disables |
| `CODEX_SUBMIT_REPASTE_DELAY` | `3.0` | Settle time before each re-paste |
| `CODEX_SUBMIT_GRACE` | `2.0` | Hold a fresh session's first paste this long after the ready marker (which precedes real input readiness); `0` = off |
| `CODEX_PASTE_VISIBLE_WAIT` | `1.5` | Wait for the paste to render in the composer before pressing Enter; `0` = legacy fixed settle |
| `CODEX_SLOW_DUMP_SECS` | `90` | Dump a diagnostic for any turn slower than this, even on success |
| `CODEX_LAST_MAX_WAIT` | `180` | Cap on `GET /last?wait=N` so it never blocks longer than a `/chat` |
| `CODEX_EXTRA_ARGS` | _(empty)_ | Extra flags for every `codex` process, e.g. `--add-dir /repos` |
| `CODEX_MODEL` | _(empty)_ | Model slug, passed as `-m …` on every launch; empty = codex's own default. `docker-compose.yml` sets `gpt-6-astra`. Set there rather than in `config.toml`, which drifts (see [Configuration](#configuration-auto-approve--xhigh-reasoning)) |
| `CODEX_EFFORT` | _(empty)_ | Reasoning effort, passed as `-c model_reasoning_effort=…` on every launch; empty = whatever `config.toml` says. `docker-compose.yml` sets `xhigh`. Same drift reason |
| `CODEX_SERVICE_TIER` | _(empty)_ | Service tier (`default`, `priority`, `flex`; `fast` is the legacy alias of `priority`), passed as `-c service_tier=…` on every launch; empty = whatever `config.toml` says. `docker-compose.yml` sets `default` (the file had drifted to `fast`, which burns the ChatGPT usage budget faster). Same drift reason |
| `CODEX_TMUX_SOCKET` | `codex-rest` | Dedicated tmux socket (distinct from agy's `agy-rest`) |
| `SESSIONS_ROOT` | `/tmp/codex-rest-sessions` | Per-session working dirs |
| `CODEX_HOME` | `~/.codex` | Where codex stores auth + sessions (rollouts are read from here) |
| `LOG_DIR` | `/app/logs` | Where the rolling log + per-incident diagnostic dumps are written |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

(A legacy `CODEX_EXEC_TIMEOUT` is still honored as the hard-timeout ceiling if set.)

## Tests

```bash
./runTestCodex.sh        # requires the codex-rest server running on :8001
```
