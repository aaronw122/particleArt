#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="prtkl"
REMOTE_HOST="homeserver"
REMOTE_DIR="~/${PROJECT_NAME}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Deploying ${PROJECT_NAME} to ${REMOTE_HOST}..."

echo "--- Creating remote directory..."
ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_DIR}"

echo "--- Syncing project files..."
rsync -avz --delete \
    --include='server.py' \
    --include='requirements-server.txt' \
    --include='Dockerfile' \
    --include='docker-compose.yml' \
    --include='.dockerignore' \
    --include='web/***' \
    --exclude='*' \
    "${SCRIPT_DIR}/" "${REMOTE_HOST}:${REMOTE_DIR}/"

echo "--- Building and starting containers..."
ssh "${REMOTE_HOST}" "cd ${REMOTE_DIR} && docker compose up -d --build --force-recreate"

echo "--- Waiting for Ollama to be healthy..."
for i in $(seq 1 30); do
    if ssh "${REMOTE_HOST}" "docker exec ${PROJECT_NAME}-ollama-1 ollama list" &>/dev/null; then
        echo "    Ollama is ready."
        break
    fi
    echo "    Attempt ${i}/30 — waiting 5s..."
    sleep 5
done

echo "--- Pulling Ollama model (if not cached)..."
ssh "${REMOTE_HOST}" "docker exec ${PROJECT_NAME}-ollama-1 ollama pull llama3.1:8b"

echo "--- Verifying containers..."
ssh "${REMOTE_HOST}" "docker ps --filter name=${PROJECT_NAME} --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"

echo "==> Deploy complete."
