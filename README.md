# Antigravity CLI REST Bridge

REST API server that wraps Google's Antigravity CLI (`agy`) interactive mode, giving you HTTP endpoints for **named multi-session chat** with full conversation continuity. Each session gets its own live, warm CLI process — run as many parallel conversations as you need, with no per-request startup cost.

## Quick Start (Docker Compose)

```bash
# 1. Start the server
docker compose up -d --build

# 2. First time only — authenticate inside the container
docker exec -it gemini-cli-rest-agy-rest-1 agy
# Complete auth in browser, then Ctrl+C to exit

# 3. Restart so the server picks up the auth
docker compose restart

# 4. Chat! (session "default" is created automatically)
curl -s -X POST http://localhost:8000/chat/default \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is 2+2?"}'
```

Auth persists across restarts (stored in a Docker named volume).

## Endpoints

### `POST /chat/{name}` — Send a message

Creates the session on first use. Follow-up messages retain full conversation context.

```bash
curl -s -X POST http://localhost:8000/chat/research \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain the CAP theorem in 2 sentences"}'
```

Response:
```json
{
  "response": "The CAP theorem states that a distributed system can only simultaneously provide two out of three guarantees: Consistency, Availability, and Partition Tolerance.",
  "session": "research",
  "turn": 1,
  "elapsed_ms": 9573
}
```

Follow-up in the same session:
```bash
curl -s -X POST http://localhost:8000/chat/research \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Now compare it to the PACELC theorem"}'
```

Meanwhile, a completely separate conversation:
```bash
curl -s -X POST http://localhost:8000/chat/coding \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a Python fibonacci generator"}'
```

### `GET /last/{name}` — Re-read the last answer (recover a lost response)

Returns a session's most recent **completed** answer **without re-asking** — for when a `POST /chat` response never reached you (the 3-min hard cap, or a connectivity blip after the CLI had already answered). The answer is durable in the CLI's own transcript/rollout regardless of whether the HTTP response arrived, so `/last` simply reads it back.

It is binary by design: you get the answer **only once the turn is done**, never a partial and never a *previous* turn's answer in its place.

```bash
# instant snapshot
curl -s http://localhost:8000/last/research

# or wait up to 30s for an in-flight turn to finish, then return it
curl -s "http://localhost:8000/last/research?wait=30"
```

Response when the turn has finished:
```json
{ "done": true, "response": "…the answer…", "turn": 2, "session": "research", "elapsed_ms": 12 }
```

While the turn is still running (or was never ingested) — poll again:
```json
{ "done": false, "response": null, "turn": 2, "session": "research", "elapsed_ms": 30001 }
```

`wait` is capped at `LAST_MAX_WAIT` (default 180s) so `/last` never blocks longer than a `/chat` would. Recovery works while the **server is up** (the warm process holds the session); it does not survive a full server restart.

### `POST /clear/{name}` — Clear conversation context

Wipes the conversation by respawning agy in a fresh project directory (~5s). A respawn is required because agy keeps **cross-conversation memory per project**: its own `/clear` starts a new conversation that still receives summaries of the previous ones — and the agent can read their transcripts to recover "cleared" context.

```bash
curl -s -X POST http://localhost:8000/clear/research
```

### `POST /reset/{name}` — Full process restart

Kills the session's agy process and spawns a new one. Use when the CLI is stuck or misbehaving (~25s).

```bash
curl -s -X POST http://localhost:8000/reset/research
```

### `DELETE /chat/{name}` — Delete a session

Kills and permanently removes a session and its process.

```bash
curl -s -X DELETE http://localhost:8000/chat/research
```

### `POST /stop` — Stop all sessions

Kills every active session. Clean slate.

```bash
curl -s -X POST http://localhost:8000/stop
```

### `GET /health` — Health check

Lists all active sessions and their status.

```bash
curl -s http://localhost:8000/health
```

```json
{
  "status": "ok",
  "active_sessions": 2,
  "sessions": [
    {"name": "research", "alive": true, "turn_count": 3},
    {"name": "coding", "alive": true, "turn_count": 1}
  ]
}
```

## Use Cases

- **Parallel research** — Run multiple topic-specific sessions simultaneously (`/chat/ml-papers`, `/chat/api-docs`, `/chat/competitor-analysis`)
- **Automation scripts** — Each script gets its own named session with isolated context, no cross-contamination
- **Agent contexts** — Give each autonomous agent a dedicated session (`/chat/agent-planner`, `/chat/agent-coder`, `/chat/agent-reviewer`)
- **Interactive + batch** — Keep a long-running exploratory session open while firing off one-shot queries in disposable sessions

## Docker Compose Commands

```bash
docker compose up -d --build   # Build and start
docker compose down             # Stop
docker compose restart          # Restart (keeps auth)
docker compose logs -f          # View live logs (stdout)
```

## Logs & diagnostics

Both bridges write **persistent log files** in addition to stdout, plus a
**per-incident dump for every slow or failed turn** — so you can find out *why*
a turn was slow or timed out without `docker exec`'ing into a live container.

Everything lands under `LOG_DIR` (`/app/logs`), which docker-compose mounts to
**`./logs`** on the host:

```
./logs/
├── agy-rest.log              # rolling agy bridge log (10MB x 5)
├── codex-rest.log            # rolling codex bridge log
└── timeouts/                 # one file per slow/failed turn — START HERE
    ├── <session>-turn<N>-<conversation-id>.log         # agy
    └── codex-<session>-turn<N>-<rollout>.log           # codex
```

**When does a dump get written?** Whenever a turn hits the hard cap, stalls, or
simply runs **slower than `*_SLOW_DUMP_SECS` (90s)** — even if it succeeded.
A turn faster than that leaves no dump.

**What's in each dump** (both bridges follow the same shape now):

- **agy** — the resolved conversation id + transcript path, the *timestamped*
  tail of the transcript (the step-by-step record — where the time actually
  went, e.g. a long `RUN_COMMAND` or a `node: command not found` retry loop),
  and the **rendered screen** at the moment the bridge stopped waiting. That
  screen tells you whether a missed answer was already on-screen (transcript
  flush lag) or agy was still `Generating...`.
- **codex** — the resolved session id + rollout path, the tail of the rollout
  events (`task_started` / `response_item` / `task_complete` with their
  `turn_id`s — the turn-by-turn record), and the **rendered screen** at give-up
  time. Full detail lives in codex's own rollout at
  `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`.

**Turn timeouts.** A single request is hard-capped at **3 minutes**
(`RESPONSE_HARD_TIMEOUT` / `CODEX_RESPONSE_HARD_TIMEOUT = 180`); each bridge also
gives up early if its CLI makes no progress for `*_STALL_TIMEOUT` (90s). Work
that genuinely needs longer should be handled by the client **re-polling**, not
by raising these caps. That re-poll is [`GET /last/{name}`](#get-lastname--re-read-the-last-answer-recover-a-lost-response):
the turn keeps running in the warm process after a request gives up, its answer
is written to the CLI's transcript/rollout, and `/last` reads it back once it is
done — so a capped or dropped `/chat` never loses the reply.

## Running Without Docker

Requires Python 3.11+, tmux, and the Antigravity CLI.

```bash
# Install Antigravity CLI (to ~/.local/bin/agy)
curl -fsSL https://antigravity.google/cli/install.sh | bash
agy  # authenticate once, then /quit or Ctrl+C

# Install Python deps (tmux from your package manager if missing)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run
uvicorn server:app --host 0.0.0.0 --port 8000
```

## Configuration

All config via environment variables (set in `docker-compose.yml` or shell):

| Variable | Default | Description |
|---|---|---|
| `AGY_CMD` | `agy` | Path to the agy CLI binary |
| `AGY_SKIP_PERMISSIONS` | `false` | Pass `--dangerously-skip-permissions` to agy. Prefer `"toolPermission": "always-proceed"` in agy's `settings.json` (the Docker image seeds this) |
| `AGY_EXTRA_ARGS` | | Additional CLI args (space-separated) |
| `AGY_STATE_DIR` | `~/.gemini/antigravity-cli` | agy's state directory (conversation transcripts live here) |
| `SESSIONS_ROOT` | `/tmp/agy-rest-sessions` | Per-session working directories |
| `TMUX_SOCKET` | `agy-rest` | Dedicated tmux server socket name |
| `RESPONSE_POLL_INTERVAL` | `0.5` | Seconds between completion-detection polls |
| `RESPONSE_STALL_TIMEOUT` | `90` | Give up if agy makes no progress (idle, transcript not growing) for this long |
| `RESPONSE_HARD_TIMEOUT` | `180` | Absolute hard cap on a turn (3 min), regardless of progress |
| `RESPONSE_SLOW_DUMP_SECS` | `90` | Write a diagnostic dump for any turn slower than this (even successful ones) |
| `STARTUP_TIMEOUT` | `60` | Max seconds to wait for CLI startup |
| `LOG_DIR` | `/app/logs` | Rolling logs + per-incident dumps written here (mounted to `./logs`) |
| `LOG_LEVEL` | `INFO` | Logging level |

The codex bridge (port 8001) shares this tmux architecture and has matching, `CODEX_`-prefixed knobs — `CODEX_RESPONSE_HARD_TIMEOUT` (`180`), `CODEX_RESPONSE_STALL_TIMEOUT` (`90`), `CODEX_STARTUP_TIMEOUT` (`60`), `CODEX_SLOW_DUMP_SECS` (`90`), `CODEX_TMUX_SOCKET` (`codex-rest`) — and shares `LOG_DIR` / `LOG_LEVEL`. See [CODEX.md](CODEX.md) and [Logs & diagnostics](#logs--diagnostics).

The model is **not** selected per-request — set `"model"` in agy's `settings.json` (`~/.gemini/antigravity-cli/settings.json`).

## Architecture

```
Client (curl/app)          Server (FastAPI)                 tmux              agy
─────────────────     ──────────────────────────     ───────────────    ─────────────
                         ChatManager
                      ┌──────────────────────┐
POST /chat/foo ──────→│  session "foo"       │       tmux session
                      │  AgySession ─────────────→  "agy-foo"  ─────→  live agy TUI
                      │                      │       paste-buffer -p → typed input
                      │   response text ←──────── transcript.jsonl ←── model output
  ← JSON response ←──│                      │       capture-pane → busy/idle check
                      ├──────────────────────┤
POST /chat/bar ──────→│  session "bar"       │       tmux session
                      │  AgySession ─────────────→  "agy-bar"   ─────→  live agy TUI
                      └──────────────────────┘
GET   /last/{name} ─→ re-read the last COMPLETED answer (recover a lost response)
POST  /clear/{name} → respawn in a fresh project dir (true context wipe)
POST  /reset/{name} → kill + respawn session's process
DELETE /chat/{name} → stop + remove session entirely
POST  /stop ────────→ stop all sessions
GET   /health ──────→ list all sessions + status
```

Each named session owns a live `agy` process hosted in a detached **tmux** session (dedicated socket, so it never touches your own tmux). tmux acts as a terminal-emulator mediator with three clean channels:

1. **Input** — prompts are injected with `tmux load-buffer` + `paste-buffer -p` (bracketed paste), so newlines and TUI shortcut characters (`!`, `@`, `/`, backticks) always arrive as literal text.
2. **Response content** — read from agy's structured per-conversation transcript (`brain/<conversation>/.system_generated/logs/transcript.jsonl`), not scraped off the screen. The bridge takes the completed (`DONE`) model steps that appeared after the prompt was sent.
3. **Busy/idle detection** — `tmux capture-pane` returns the *rendered* screen (no ANSI escapes); a turn is complete when a new model step exists in the transcript **and** the screen shows the idle status bar (`? for shortcuts`) with no `Generating...` indicator.

You can watch any session live while the bridge drives it:

```bash
tmux -L agy-rest attach -t agy-research   # Ctrl+B then D to detach
```

## Limitations

1. **No streaming** — Responses return only after fully collected.
2. **Prompts starting with `/`** may be interpreted as agy slash commands.
3. **Model selection** is global (agy `settings.json`), not per-session.
4. **agy's brain is shared** — sessions are isolated per project directory, but the agent has filesystem access; a prompt that explicitly asks it to read another conversation's transcript under `~/.gemini/antigravity-cli/brain/` could cross session boundaries.
