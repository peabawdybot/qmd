#!/usr/bin/env bash
set -euo pipefail

# --- PATH (crontab has no nvm) ---
export PATH="/home/claw/.nvm/versions/node/v22.22.0/bin:/home/claw/.local/bin:/usr/local/bin:/usr/bin:/bin"

# --- Config ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QMD_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$QMD_DIR/.env"
QMD="/home/claw/.local/bin/qmd"

# --- Pre-flight checks ---
if [[ ! -f "$ENV_FILE" ]]; then
  echo "FATAL: $ENV_FILE not found" >&2
  exit 1
fi

if [[ ! -x "$QMD" ]]; then
  echo "FATAL: qmd not found at $QMD" >&2
  exit 1
fi

# Source env vars
set -a
source "$ENV_FILE"
set +a

# Validate required vars
for var in QMD_LLM_BACKEND QMD_EMBED_API_KEY QMD_CHAT_API_KEY QMD_RERANK_API_KEY; do
  if [[ -z "${!var:-}" ]]; then
    echo "FATAL: $var is not set in $ENV_FILE" >&2
    exit 1
  fi
done

# --- Run ---
echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') Starting qmd update --pull"
"$QMD" update --pull

echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') Starting qmd embed (API mode)"
"$QMD" embed

echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') Done"
