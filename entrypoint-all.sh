#!/bin/bash
# Run BOTH bridges in one container: agy on :8000 and codex on :8001.
#
# It simply backgrounds the two existing single-app entrypoints. Each of those
# ends in `exec uvicorn ...`, so the backgrounded child becomes the uvicorn
# process directly — no seeding logic is duplicated here and neither
# entrypoint.sh nor entrypoint-codex.sh is modified.
#
# If EITHER bridge exits, this script exits too, so `restart: unless-stopped`
# reboots the whole container and brings both back. (The user accepts this
# coupling: a crash "is not supposed to happen", and if it does, fail loud.)
#
# NOTE: no `set -e` — a non-zero exit from a child is the EXPECTED signal here.
# Under `set -e`, `wait -n` returning a child's non-zero code would abort the
# script before the explicit cleanup below runs.
set +e

AGY_PID=""
CODEX_PID=""

# Forward termination to both children so `docker stop` shuts them down
# gracefully (agy's lifespan tears down its tmux sessions on SIGTERM).
term() {
    kill -TERM "$AGY_PID" "$CODEX_PID" 2>/dev/null || true
}
trap term TERM INT

./entrypoint.sh &
AGY_PID=$!
./entrypoint-codex.sh &
CODEX_PID=$!

# Block until the first child exits, then take the container down.
wait -n
code=$?
echo "[entrypoint-all] a bridge process exited (rc=$code) — stopping container" >&2
term
wait
exit "$code"
