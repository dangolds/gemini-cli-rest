#!/bin/bash
set -e

# Seed codex config on first run: full-access auto-approve + high reasoning so
# the interactive TUI never blocks on an approval/sandbox prompt. Only seed if
# absent — the codex-config volume persists auth.json + config.toml (and the
# one-time `codex login`) across container restarts.
CODEX_DIR=/root/.codex
CONFIG="$CODEX_DIR/config.toml"
mkdir -p "$CODEX_DIR"
if [ ! -f "$CONFIG" ]; then
    cat > "$CONFIG" <<'EOF'
model = "gpt-5.5"
model_reasoning_effort = "high"
approval_policy = "never"
sandbox_mode = "danger-full-access"
EOF
fi

# NOTE: codex authenticates INDEPENDENTLY here — run `codex login` inside the
# container once (persists in the codex-config volume). Do NOT copy/bind-mount
# the host's auth.json in: codex's ChatGPT OAuth refresh token rotates, so a
# shared token family makes the two environments revoke each other's session.

# Best-effort: update codex ONLY if the installed version differs from the
# latest release. We check the version cheaply via the GitHub API first and skip
# the ~37MB download when already current. (The curl-installed binary is not
# package-managed, so `codex update` can't self-update it reliably.) All guarded
# so a network blip / API rate-limit can never stop the bridge booting.
echo "[codex] update check (best-effort)..."
CODEX_CUR=$(codex --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
CODEX_LATEST=$(timeout 20 curl -fsSL https://api.github.com/repos/openai/codex/releases/latest 2>/dev/null \
                 | grep -oE '"tag_name"[^,]*' | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
if [ -n "$CODEX_LATEST" ] && [ "$CODEX_CUR" != "$CODEX_LATEST" ]; then
    echo "[codex] $CODEX_CUR -> $CODEX_LATEST: downloading update..."
    CODEX_URL="https://github.com/openai/codex/releases/latest/download/codex-x86_64-unknown-linux-musl.tar.gz"
    if timeout 120 curl -fsSL -o /tmp/codex.tar.gz "$CODEX_URL" && tar -xzf /tmp/codex.tar.gz -C /tmp; then
        if mv -f /tmp/codex-x86_64-unknown-linux-musl /usr/local/bin/codex && chmod +x /usr/local/bin/codex; then
            echo "[codex] now $(codex --version 2>&1)"
        else
            echo "[codex] binary replace failed (continuing with existing)"
        fi
    else
        echo "[codex] download failed (continuing with $CODEX_CUR)"
    fi
    rm -f /tmp/codex.tar.gz /tmp/codex-x86_64-unknown-linux-musl 2>/dev/null || true
else
    echo "[codex] up to date ($CODEX_CUR) or version unknown — skipping download"
fi

# codex_server now hosts a live `codex` TUI per session inside tmux (like the
# agy bridge), so the uvloop workaround applies here too: --loop asyncio avoids
# uvloop mishandling the subprocess pipes inherited by the forked tmux daemon.
# (codex_server's lifespan also pre-starts the tmux server with DEVNULL stdio.)
exec uvicorn codex_server:app --host 0.0.0.0 --port 8001 --loop asyncio
