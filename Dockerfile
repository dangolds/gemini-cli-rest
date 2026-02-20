FROM node:20-slim AS base

# Install Python + system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Install Gemini CLI globally
RUN npm install -g @google/gemini-cli

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN python3 -m venv /app/.venv && \
    /app/.venv/bin/pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENV PATH="/app/.venv/bin:$PATH"
ENV GEMINI_CMD="gemini"

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
