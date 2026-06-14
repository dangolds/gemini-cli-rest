FROM python:3.12-slim

# tmux hosts the live agy processes; curl installs agy
RUN apt-get update && \
    apt-get install -y --no-install-recommends tmux curl ca-certificates git openssh-client && \
    rm -rf /var/lib/apt/lists/*

# Install Antigravity CLI (installs to /root/.local/bin/agy)
RUN curl -fsSL https://antigravity.google/cli/install.sh | bash

# Install the LATEST OpenAI Codex CLI (standalone static musl binary — no Node
# runtime). GitHub's "latest release" redirect tracks new versions without a
# pin, mirroring how agy installs via its vendor script above. The tarball holds
# a single binary we rename to `codex`.
RUN curl -fsSL -o /tmp/codex.tar.gz \
      "https://github.com/openai/codex/releases/latest/download/codex-x86_64-unknown-linux-musl.tar.gz" \
 && tar -xzf /tmp/codex.tar.gz -C /usr/local/bin \
 && mv /usr/local/bin/codex-x86_64-unknown-linux-musl /usr/local/bin/codex \
 && chmod +x /usr/local/bin/codex \
 && rm /tmp/codex.tar.gz

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN python -m venv /app/.venv && \
    /app/.venv/bin/pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Codex bridge (separate app on :8001; the codex-rest compose service overrides
# the entrypoint below to run it). agy's default entrypoint is unchanged.
COPY codex_server.py .
COPY entrypoint-codex.sh .
RUN chmod +x entrypoint-codex.sh

# Orchestrator that runs BOTH bridges in one container (default entrypoint).
# entrypoint.sh / entrypoint-codex.sh remain usable for single-bridge runs.
COPY entrypoint-all.sh .
RUN chmod +x entrypoint-all.sh

ENV PATH="/app/.venv/bin:/root/.local/bin:$PATH"
ENV AGY_CMD="agy"

EXPOSE 8000
EXPOSE 8001

ENTRYPOINT ["./entrypoint-all.sh"]
