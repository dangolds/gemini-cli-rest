#!/bin/bash
set -e

# Ensure the .gemini config directory exists
mkdir -p /root/.gemini

# Trust /app so Gemini CLI doesn't prompt interactively
TRUST_FILE="/root/.gemini/trustedFolders.json"
if [ ! -f "$TRUST_FILE" ] || ! grep -q "/app" "$TRUST_FILE" 2>/dev/null; then
    echo '{ "/app": "TRUST_FOLDER" }' > "$TRUST_FILE"
fi

exec uvicorn server:app --host 0.0.0.0 --port 8000
