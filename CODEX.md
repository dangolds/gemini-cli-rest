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
| Completion push (fast path) | terminal bell via tmux `alert-bell` hook | codex `notify` hook → `events.jsonl` |
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

**Completion push (fast path).** The bridge launches codex with
`-c 'notify=["<hook>"]'`; codex runs the hook on `agent-turn-complete` with a
JSON payload (thread-id, turn-id, cwd, last-assistant-message), and the hook
appends it to `CODEX_NOTIFY_DIR/events.jsonl`. The bridge checks that small
file every `CODEX_RESPONSE_FAST_POLL` (0.3s) and, on a matching new event,
reads the rollout once and returns — the rollout stays the single source of
truth for the answer text. The `task_complete` polling above is retained as
fallback: a missed notification degrades to slower, never to hung. (`[tui]`
notifications / OSC 9 is *not* used — it emits nothing in a detached tmux
pane.) `/chat` responses report the path taken in an optional `"via"` field:
`"notify"` on the fast path, otherwise the legacy exit reasons
(`"rollout_done"`, `"stalled"`, `"hard_timeout"`).

**Why a warm process** (rather than the simpler `codex exec` per turn): it keeps
conversation context in-process across turns and mirrors the agy bridge
one-to-one, so a single skill drives both bridges identically.

## Configuration (auto-approve + ultra reasoning)

`entrypoint-codex.sh` seeds `/root/.codex/config.toml` on first run (only if
absent) so the interactive TUI never blocks on an approval or sandbox prompt:

```toml
# no `model` pin: codex uses its own default, which tracks the current
# flagship (gpt-5.6-sol as of 2026-07). Pinning freezes you on an old model.
model_reasoning_effort = "ultra"   # 4 parallel agents: ~4x token burn, faster time-to-result
approval_policy = "never"          # never pause for approval
sandbox_mode = "danger-full-access" # the container is the sandbox
```

Do not rely on that seed for the effort. It only runs when `config.toml` is
absent, and the file lives in the persistent `codex-config` volume — so once
the volume exists the seed never runs again, while the TUI keeps rewriting the
file whenever the model or effort is switched (it had silently drifted to
`medium`). `CODEX_EFFORT` is therefore passed as `-c model_reasoning_effort=…`
on every launch, which beats the file every time.

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

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `CODEX_CMD` | `codex` | Path to the codex binary |
| `CODEX_RESPONSE_HARD_TIMEOUT` | `180` | Absolute max per turn — a request never blocks longer (3 min) |
| `CODEX_RESPONSE_STALL_TIMEOUT` | `90` | Give up after this long with no progress (idle, rollout not growing) |
| `CODEX_NOTIFY` | `1` | Use codex's `notify` hook as the fast done-signal; set to `0` to disable and revert to pure rollout polling |
| `CODEX_NOTIFY_DIR` | `/tmp/codex-rest-notify` | Directory where the notify hook appends `events.jsonl` |
| `CODEX_RESPONSE_FAST_POLL` | `0.3` | Seconds between checks of the notify events file while a turn is in flight |
| `CODEX_RESPONSE_FULL_CHECK_EVERY` | `10` | Run the full fallback poll (rollout + liveness) every Nth notify-check wake (~3s) |
| `CODEX_STARTUP_TIMEOUT` | `60` | Max wait for the TUI to reach its idle prompt |
| `CODEX_SLOW_DUMP_SECS` | `90` | Dump a diagnostic for any turn slower than this, even on success |
| `CODEX_LAST_MAX_WAIT` | `180` | Cap on `GET /last?wait=N` so it never blocks longer than a `/chat` |
| `CODEX_EXTRA_ARGS` | _(empty)_ | Extra flags for every `codex` process, e.g. `--add-dir /repos` |
| `CODEX_EFFORT` | `ultra` | Reasoning effort, passed as `-c model_reasoning_effort=…` on every launch. Set here rather than in `config.toml`, which drifts (see [Configuration](#configuration-auto-approve--ultra-reasoning)) |
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
