# Gemini CLI REST Bridge

REST API server that wraps Google's Gemini CLI interactive mode, giving you HTTP endpoints for **named multi-session chat** with full conversation continuity. Each session gets its own isolated CLI process — run as many parallel conversations as you need.

## Quick Start (Docker Compose)

```bash
# 1. Start the server
docker compose up -d --build

# 2. First time only — authenticate inside the container
docker exec -it geminidev-gemini-rest-1 gemini
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

### `POST /clear/{name}` — Clear conversation context

Clears the conversation history without restarting the process. Instant (~1s).

```bash
curl -s -X POST http://localhost:8000/clear/research
```

### `POST /reset/{name}` — Full process restart

Kills the session's Gemini CLI process and spawns a new one. Use when the CLI is stuck or misbehaving (~10s).

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
docker compose logs -f          # View logs
```

## Running Without Docker

Requires Node.js 20+ and Python 3.11+.

```bash
# Install Gemini CLI
npm install -g @google/gemini-cli
gemini  # authenticate once

# Install Python deps
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
| `GEMINI_CMD` | `gemini` | Path to the gemini CLI binary |
| `GEMINI_MODEL` | *(auto)* | Model override (e.g. `gemini-2.5-pro`) |
| `GEMINI_YOLO` | `true` | Auto-approve all tool actions |
| `GEMINI_SCREEN_READER` | `true` | Screen-reader mode for cleaner output |
| `GEMINI_EXTRA_ARGS` | | Additional CLI args (space-separated) |
| `RESPONSE_IDLE_TIMEOUT` | `5` | Seconds of silence before considering response complete |
| `RESPONSE_MAX_TIMEOUT` | `120` | Hard cap on response wait time (seconds) |
| `STARTUP_TIMEOUT` | `60` | Max seconds to wait for CLI startup |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG` for raw output chunks) |

### Tuning the idle timeout

- **Too low** (< 3s): May cut off responses mid-generation
- **Too high** (> 10s): Adds unnecessary latency to every response
- **Recommended**: Start with `5`, increase to `8-10` if you see truncated responses

## Architecture

```
Client (curl/app)          Server (FastAPI)                  Gemini CLI
─────────────────     ──────────────────────────     ─────────────────
                         ChatManager
                      ┌──────────────────────┐
POST /chat/foo ──────→│  session "foo"       │
                      │  GeminiProcess ──────────→ pexpect (PTY) → stdin
                      │  ← read until idle ←────── stdout
  ← JSON response ←──│                      │
                      ├──────────────────────┤
POST /chat/bar ──────→│  session "bar"       │
                      │  GeminiProcess ──────────→ pexpect (PTY) → stdin
                      │  ← read until idle ←────── stdout
  ← JSON response ←──│                      │
                      └──────────────────────┘
POST  /clear/{name} → sends /clear to session's CLI
POST  /reset/{name} → kill + respawn session's process
DELETE /chat/{name} → stop + remove session entirely
POST  /stop ────────→ stop all sessions
GET   /health ──────→ list all sessions + status
```

The server manages sessions via `ChatManager`. Each named session owns a `GeminiProcess` — a `gemini --screen-reader --yolo` child process on a pseudo-terminal (PTY via `pexpect`). Sessions are created lazily on first `/chat/{name}` request and wait for the `"Type your message"` ready marker before accepting input. Responses are detected by idle-timeout (configurable silence threshold). All ANSI escape codes and TUI artifacts are stripped, and model output is extracted from `"Model:"` prefixed lines.

## Limitations

1. **No streaming** — Responses return only after fully collected.
2. **Idle-timeout heuristic** — Long model pauses (tool execution) may cause truncation. Increase `RESPONSE_IDLE_TIMEOUT` if needed.
