#!/bin/bash
set -e

# Seed codex config on first run: full-access auto-approve + xhigh reasoning so
# headless `codex exec` never blocks on an approval/sandbox prompt. Only seed if
# absent — the codex-config volume persists auth.json + config.toml (and the
# one-time `codex login`) across container restarts.
CODEX_DIR=/root/.codex
CONFIG="$CODEX_DIR/config.toml"
mkdir -p "$CODEX_DIR"
if [ ! -f "$CONFIG" ]; then
    cat > "$CONFIG" <<'EOF'
model = "gpt-5.5"
model_reasoning_effort = "xhigh"
approval_policy = "never"
sandbox_mode = "danger-full-access"
EOF
fi

# codex_server uses plain create_subprocess_exec (no tmux), so the uvloop
# workaround the agy bridge needs does not apply here.
exec uvicorn codex_server:app --host 0.0.0.0 --port 8001
