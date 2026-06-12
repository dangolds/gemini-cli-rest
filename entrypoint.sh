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
  "trustedWorkspaces": ["/app", "/tmp/agy-rest-sessions"]
}
EOF
fi

# --loop asyncio: uvloop's subprocess pipes mishandle the forked tmux daemon
exec uvicorn server:app --host 0.0.0.0 --port 8000 --loop asyncio
