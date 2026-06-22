#!/bin/bash
set -e

# Seed agy settings on first run: auto-approve tool permissions and trust the
# session workspaces so the TUI never blocks on an interactive prompt.
AGY_DIR=/root/.gemini/antigravity-cli
SETTINGS="$AGY_DIR/settings.json"
mkdir -p "$AGY_DIR"
if [ ! -f "$SETTINGS" ]; then
    cat > "$SETTINGS" <<'EOF'
{
  "toolPermission": "always-proceed",
  "enableTelemetry": false,
  "notifications": false,
  "trustedWorkspaces": ["/app", "/repos", "/tmp/agy-rest-sessions"]
}
EOF
fi

# Best-effort: pull the latest agy on every (re)start. Guarded so a missing
# network / transient failure can never stop the bridge from coming up (the
# `|| echo` keeps `set -e` from aborting here).
echo "[agy] update check (best-effort)..."
timeout 90 agy update </dev/null 2>&1 || echo "[agy] update skipped/failed (continuing)"

# --loop asyncio: uvloop's subprocess pipes mishandle the forked tmux daemon
exec uvicorn server:app --host 0.0.0.0 --port 8000 --loop asyncio
