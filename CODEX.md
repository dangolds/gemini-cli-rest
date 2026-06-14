# Codex CLI REST Bridge

A REST API that wraps OpenAI's **Codex CLI** (`codex`) in headless mode, exposing
named multi-session chat with conversation continuity. It is a sibling of the
agy/Gemini bridge (`server.py`, see [README.md](README.md)) and exposes the
**identical REST contract**, just on port **8001** instead of 8000.

Use it to get a second opinion from `gpt-5.5` alongside Gemini.

## How it differs from the agy bridge

The agy bridge drives a live interactive TUI through a tmux terminal-emulator
mediator. **Codex needs none of that** — it ships a first-class headless mode:

| | agy bridge (`server.py`) | codex bridge (`codex_server.py`) |
| --- | --- | --- |
| Process model | one **live** `agy` per session, hosted in tmux | **stateless** — each turn spawns `codex exec` |
| Continuity | tmux session + transcript polling | `codex exec resume <session-id>` |
| "Done" detection | rendered-screen busy/idle markers | process exit |
| Context wipe (`/clear`) | respawn in a fresh project dir | new working dir + forget the session id |

Why stateless is fine: codex process boot is ~0.25s (negligible next to the
model call), and OpenAI's server-side prompt cache spans separate `exec`
invocations — so keeping codex warm would buy nothing. Each named session is
just its codex session UUID plus a per-session working directory.

## Configuration (auto-approve + xhigh)

`entrypoint-codex.sh` seeds `/root/.codex/config.toml` on first run (only if
absent) so headless `codex exec` never blocks:

```toml
model = "gpt-5.5"
model_reasoning_effort = "xhigh"   # maximum reasoning
approval_policy = "never"          # never pause for approval
sandbox_mode = "danger-full-access" # the container is the sandbox
```

Every invocation also passes `--dangerously-bypass-approvals-and-sandbox` and
`--skip-git-repo-check` as belt-and-suspenders, so a stale config can't
reintroduce a prompt.

## Run it

Both bridges run in a **single container** (the `bridges` service in
`docker-compose.yml`): agy on :8000 and codex on :8001, started by
`entrypoint-all.sh`. If either bridge process dies the container restarts and
brings both back.

```bash
docker compose up -d --build           # one image, one container, both bridges
docker exec -it gemini-cli-rest-bridges-1 codex login   # one-time codex auth (persists in the codex-config volume)
curl -s http://localhost:8001/health   # codex     (agy is on :8000)
```

(Adjust the container name to match `docker compose ps`.) Auth and config persist
in the `codex-config` volume across restarts. To let codex read your repos, mount
them under `/repos` in the `bridges` service and set
`CODEX_EXTRA_ARGS=--add-dir /repos` (mirrors agy's `/repos` setup).

## API

Identical to the agy bridge, on `http://localhost:8001`:

| Action | Command | Purpose |
| --- | --- | --- |
| Send message | `POST /chat/{session}` | Send prompt, creates session if new |
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
| `CODEX_EXEC_TIMEOUT` | `300` | Max seconds for a single turn (xhigh can be slow) |
| `CODEX_EXTRA_ARGS` | _(empty)_ | Extra flags for every call, e.g. `--add-dir /repos` |
| `SESSIONS_ROOT` | `/tmp/codex-rest-sessions` | Per-session working dirs |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## Tests

```bash
./runTestCodex.sh        # requires the codex-rest server running on :8001
```
