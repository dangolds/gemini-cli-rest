FROM python:3.12-slim

# tmux hosts the live agy processes; curl installs agy
RUN apt-get update && \
    apt-get install -y --no-install-recommends tmux curl ca-certificates git openssh-client && \
    rm -rf /var/lib/apt/lists/*

# Install Antigravity CLI (installs to /root/.local/bin/agy)
RUN curl -fsSL https://antigravity.google/cli/install.sh | bash

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN python -m venv /app/.venv && \
    /app/.venv/bin/pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENV PATH="/app/.venv/bin:/root/.local/bin:$PATH"
ENV AGY_CMD="agy"

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
