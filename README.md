# Gemini CLI REST Bridge

REST API server that wraps Google's Gemini CLI interactive mode, giving you HTTP endpoints for chat with full conversation continuity.

## Quick Start (Docker Compose)

```bash
# 1. Start the server
docker compose up -d --build

# 2. First time only — authenticate inside the container
docker exec -it geminidev-gemini-rest-1 gemini
# Complete auth in browser, then Ctrl+C to exit

# 3. Restart so the server picks up the auth
docker compose restart

# 4. Chat!
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is 2+2?"}'
```

Auth persists across restarts (stored in a Docker named volume).

## Endpoints

### `POST /chat` — Send a message

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain the CAP theorem in 2 sentences"}'
```

Response:
```json
{
  "response": "The CAP theorem states that a distributed system can only simultaneously provide two out of three guarantees: Consistency, Availability, and Partition Tolerance.",
  "turn": 1,
  "elapsed_ms": 9573
}
```

Follow-up messages retain full conversation context:
```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Now compare it to the PACELC theorem"}'
```

### `POST /clear` — Clear conversation context

Clears the conversation history without restarting the process. Instant (~1s).

```bash
curl -s -X POST http://localhost:8000/clear
```

### `POST /reset` — Full process restart

Kills the Gemini CLI process and spawns a new one. Use when the CLI is stuck or misbehaving (~10s).

```bash
curl -s -X POST http://localhost:8000/reset
```

### `GET /health` — Health check

```bash
curl -s http://localhost:8000/health
```

```json
{"status": "ok", "alive": true, "turn_count": 3}
```

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
Client (curl/app)          Server (FastAPI)           Gemini CLI
─────────────────     ──────────────────────     ─────────────────
POST /chat ──────────→ pexpect (PTY) ──────────→ stdin
                       ← read until idle ←─────── stdout
     ← JSON response ←
POST /clear ────────→ sends /clear command
POST /reset ────────→ kill + respawn
GET  /health ───────→ process status check
```

The server spawns `gemini --screen-reader --yolo` as a child process using a pseudo-terminal (PTY via `pexpect`). It waits for the `"Type your message"` ready marker before accepting requests. Responses are detected by idle-timeout (configurable silence threshold). All ANSI escape codes and TUI artifacts are stripped, and model output is extracted from `"Model:"` prefixed lines.

## Limitations

1. **Single-session** — One conversation at a time. Concurrent `/chat` requests are serialized.
2. **No streaming** — Responses return only after fully collected.
3. **Idle-timeout heuristic** — Long model pauses (tool execution) may cause truncation. Increase `RESPONSE_IDLE_TIMEOUT` if needed.
