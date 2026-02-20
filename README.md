# Gemini CLI REST Bridge

REST API server that wraps Google's Gemini CLI interactive mode, giving you HTTP endpoints for chat with full conversation continuity.

## Architecture

```
Client (curl/app)          Server (FastAPI)           Gemini CLI
─────────────────     ──────────────────────     ─────────────────
POST /chat ──────────→ pexpect (PTY) ──────────→ stdin
                       ← read until idle ←─────── stdout
     ← JSON response ←
POST /reset ─────────→ kill + respawn
GET  /health ────────→ process status check
```

**How it works:**

1. On startup, the server spawns `gemini --screen-reader --yolo` as a child process using a pseudo-terminal (PTY via `pexpect`)
2. `POST /chat` writes the prompt to stdin, then collects stdout chunks until an **idle timeout** (default 5s of silence) signals the response is complete
3. `POST /reset` kills the process and spawns a fresh one (new conversation)
4. All ANSI escape codes and TUI artifacts are stripped from the output

**Key design decisions:**

- **PTY over raw pipes** — Gemini CLI uses Ink (React-based TUI). A PTY ensures it behaves as if connected to a real terminal, while `--screen-reader` mode simplifies the output
- **Idle-timeout response detection** — Since there's no reliable end-of-response marker, we detect completion by waiting for output to stop. This is configurable via `RESPONSE_IDLE_TIMEOUT`
- **Async lock** — Only one request can talk to the subprocess at a time, preventing interleaved output
- **`--yolo` mode** — Auto-approves tool actions so the server doesn't block on confirmation prompts

## Prerequisites

- **Node.js 20+** (for Gemini CLI)
- **Python 3.11+**
- **Gemini CLI** installed and authenticated:
  ```bash
  npm install -g @google/gemini-cli
  gemini   # run once to authenticate via browser
  ```

## Setup

```bash
cd gemini-rest
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Or with Docker:
```bash
docker build -t gemini-rest .
docker run -p 8000:8000 \
  -v ~/.gemini:/root/.gemini \
  -e GOOGLE_API_KEY=your_key \
  gemini-rest
```

## Usage

### Chat

```bash
# First message starts the conversation
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain the CAP theorem in 2 sentences"}' | jq

# Follow-up (same session, has full context)
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Now compare it to the PACELC theorem"}' | jq
```

**Response:**
```json
{
  "response": "The CAP theorem states that...",
  "turn": 1,
  "elapsed_ms": 3420
}
```

### Reset conversation

```bash
curl -s -X POST http://localhost:8000/reset | jq
```

### Health check

```bash
curl -s http://localhost:8000/health | jq
```

## Configuration

All config is via environment variables:

| Variable | Default | Description |
|---|---|---|
| `GEMINI_CMD` | `gemini` | Path to the gemini CLI binary |
| `GEMINI_MODEL` | *(default)* | Model override (e.g. `gemini-2.5-pro`) |
| `GEMINI_YOLO` | `true` | Auto-approve all tool actions |
| `GEMINI_SCREEN_READER` | `true` | Use screen-reader mode for cleaner output |
| `GEMINI_EXTRA_ARGS` | | Additional CLI args (space-separated) |
| `RESPONSE_IDLE_TIMEOUT` | `5` | Seconds of silence before considering response complete |
| `RESPONSE_MAX_TIMEOUT` | `120` | Hard cap on response wait time (seconds) |
| `STARTUP_TIMEOUT` | `30` | Max seconds to wait for CLI startup |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG` for raw output chunks) |

### Tuning the idle timeout

- **Too low** (< 3s): May cut off responses mid-generation, especially for complex prompts where the model pauses to think
- **Too high** (> 10s): Adds unnecessary latency to every response
- **Recommended**: Start with `5`, increase to `8-10` if you see truncated responses. Use `LOG_LEVEL=DEBUG` to watch the raw chunks and tune accordingly

## Limitations & Caveats

1. **Response detection is heuristic** — The idle-timeout approach works well in practice but isn't perfect. Very long pauses mid-response (e.g., tool execution) could cause premature cutoff. Increase `RESPONSE_IDLE_TIMEOUT` if this happens.

2. **Single-session** — The server manages one conversation at a time. Multiple concurrent `/chat` requests are serialized by the async lock. For multi-tenant use, you'd need to spawn multiple processes.

3. **ANSI stripping isn't perfect** — Despite `--screen-reader` and `TERM=dumb`, some TUI artifacts may leak through. The regex-based stripper handles the common cases but may need tuning for edge cases.

4. **No streaming** — Responses are returned only after the full response is collected. Adding SSE streaming would require a different read strategy.

## Extending

**Add streaming (SSE):**
```python
from fastapi.responses import StreamingResponse

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def generate():
        # yield chunks as they arrive instead of waiting for idle
        ...
    return StreamingResponse(generate(), media_type="text/event-stream")
```

**Multi-session support:**
```python
# Maintain a dict of GeminiProcess instances keyed by session ID
sessions: dict[str, GeminiProcess] = {}

@app.post("/chat/{session_id}")
async def chat(session_id: str, req: ChatRequest):
    if session_id not in sessions:
        sessions[session_id] = GeminiProcess()
        await sessions[session_id].start()
    return await sessions[session_id].send(req.prompt)
```
