#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "[runTestCodex] Creating venv at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --quiet --upgrade pip
    "$VENV_DIR/bin/pip" install --quiet pytest httpx
fi

exec "$VENV_DIR/bin/pytest" test_codex_server.py -v "$@"
