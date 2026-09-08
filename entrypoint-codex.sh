#!/bin/bash
set -e

# Seed codex config on first run: full-access auto-approve, plus the same
# gpt-6-astra / xhigh pin the launch flags enforce, so the interactive TUI never
# blocks on an approval/sandbox prompt. Only seed if
# absent — the codex-config volume persists auth.json + config.toml (and the
# one-time `codex login`) across container restarts.
CODEX_DIR=/root/.codex
CONFIG="$CODEX_DIR/config.toml"
mkdir -p "$CODEX_DIR"
if [ ! -f "$CONFIG" ]; then
    cat > "$CONFIG" <<'EOF'
model = "gpt-6-astra"
model_reasoning_effort = "xhigh"
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
# Every assignment is guarded (`|| VAR=`) so a rate-limited API, a missing
# binary or an unmatched grep can never trip `set -e` and stop the boot.
CODEX_CUR=$(codex --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1) || CODEX_CUR=
CODEX_TAG=$(timeout 20 curl -fsSL --connect-timeout 10 https://api.github.com/repos/openai/codex/releases/latest 2>/dev/null \
                 | grep -oE '"tag_name"[^,]*' | grep -oE 'rust-v[0-9]+\.[0-9]+\.[0-9]+' | head -1) || CODEX_TAG=
CODEX_LATEST=${CODEX_TAG#rust-v}
REL="https://github.com/openai/codex/releases/download"
CLI_TGZ=codex-x86_64-unknown-linux-musl
HOST_TGZ=codex-code-mode-host-x86_64-unknown-linux-musl
HOST_BIN=/usr/local/bin/codex-code-mode-host
HOST_STAMP=/usr/local/bin/.codex-code-mode-host.version   # release the helper came from
if [ -n "$CODEX_LATEST" ] && [ "$CODEX_CUR" != "$CODEX_LATEST" ]; then
    echo "[codex] $CODEX_CUR -> $CODEX_LATEST: downloading update (cli + code-mode host)..."
    # codex 0.15x runs its `exec` tool through the separate `codex-code-mode-host`
    # binary (see Dockerfile); the two must come from the SAME release. Stage both
    # in /tmp and only swap them in once both are ready, so a failed download can
    # never leave a new cli beside an old/missing helper.
    if timeout 120 curl -fsSL --connect-timeout 10 --retry 2 -o /tmp/codex.tar.gz "$REL/$CODEX_TAG/$CLI_TGZ.tar.gz" \
       && timeout 120 curl -fsSL --connect-timeout 10 --retry 2 -o /tmp/codex-host.tar.gz "$REL/$CODEX_TAG/$HOST_TGZ.tar.gz" \
       && tar -xzf /tmp/codex.tar.gz -C /tmp && tar -xzf /tmp/codex-host.tar.gz -C /tmp \
       && chmod +x "/tmp/$CLI_TGZ" "/tmp/$HOST_TGZ"; then
        # Swap with rollback: keep the old cli as codex.old until the helper is
        # in place too, so a failed second mv can't leave a new cli + old helper.
        if { [ ! -f /usr/local/bin/codex ] || mv -f /usr/local/bin/codex /usr/local/bin/codex.old; } \
           && mv -f "/tmp/$CLI_TGZ" /usr/local/bin/codex && mv -f "/tmp/$HOST_TGZ" "$HOST_BIN"; then
            echo "$CODEX_LATEST" > "$HOST_STAMP" || true
            rm -f /usr/local/bin/codex.old || true
            echo "[codex] now $(codex --version 2>&1 || echo unknown)"
        else
            [ ! -f /usr/local/bin/codex.old ] || mv -f /usr/local/bin/codex.old /usr/local/bin/codex || true
            echo "[codex] binary replace failed (rolled back, continuing with $CODEX_CUR)"
        fi
    else
        echo "[codex] download failed (continuing with $CODEX_CUR)"
    fi
    rm -f /tmp/codex.tar.gz /tmp/codex-host.tar.gz "/tmp/$CLI_TGZ" "/tmp/$HOST_TGZ" 2>/dev/null || true
else
    echo "[codex] up to date ($CODEX_CUR) or version unknown — skipping download"
fi

# Repair: helper missing (image built before it was added) or stamped for a
# different release than the installed cli. Fetch it from the INSTALLED cli's
# release so the pair always matches.
# Re-read the cli version from the binary actually on disk (not the value we
# assumed above) so the helper is matched to what really got installed.
CODEX_CUR=$(codex --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1) || CODEX_CUR=
HOST_HAVE=$(cat "$HOST_STAMP" 2>/dev/null) || HOST_HAVE=
if [ -n "$CODEX_CUR" ] && { [ ! -x "$HOST_BIN" ] || [ "$HOST_HAVE" != "$CODEX_CUR" ]; }; then
    echo "[codex] code-mode host missing/mismatched (have '$HOST_HAVE', need $CODEX_CUR): installing from release rust-v$CODEX_CUR..."
    if timeout 120 curl -fsSL --connect-timeout 10 --retry 2 -o /tmp/codex-host.tar.gz "$REL/rust-v$CODEX_CUR/$HOST_TGZ.tar.gz" \
       && tar -xzf /tmp/codex-host.tar.gz -C /tmp && chmod +x "/tmp/$HOST_TGZ" && mv -f "/tmp/$HOST_TGZ" "$HOST_BIN"; then
        echo "$CODEX_CUR" > "$HOST_STAMP" || true
        echo "[codex] code-mode host installed"
    else
        echo "[codex] code-mode host install failed (exec tool broken; retried next boot)"
    fi
    rm -f /tmp/codex-host.tar.gz "/tmp/$HOST_TGZ" 2>/dev/null || true
fi

# codex_server now hosts a live `codex` TUI per session inside tmux (like the
# agy bridge), so the uvloop workaround applies here too: --loop asyncio avoids
# uvloop mishandling the subprocess pipes inherited by the forked tmux daemon.
# (codex_server's lifespan also pre-starts the tmux server with DEVNULL stdio.)
# --no-access-log: see entrypoint.sh.
exec uvicorn codex_server:app --host 0.0.0.0 --port 8001 --loop asyncio --no-access-log
