#!/usr/bin/env bash
# Runs the visualizer against local datasets, together with the policy
# evaluation service (backend/policy_server.py) when a Python with lerobot is
# available. Settings come from the environment, else from a local .env file:
#   LOCAL_LEROBOT_DATA_ROOT  (required) folder holding datasets/<org>/<dataset>/
#                            LeRobot datasets and train/<run>/ training outputs
#   POLICY_PYTHON            python executable with lerobot + torch installed
#                            (default: python3 on PATH)
#   POLICY_PORT              port of the policy service (default: 7862)
set -euo pipefail
cd "$(dirname "$0")/.."

# Load KEY=value lines from .env; variables already set in the environment win.
if [ -f .env ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    [[ "$line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    key="${BASH_REMATCH[2]}"
    value="${BASH_REMATCH[3]}"
    [ -n "${!key+x}" ] && continue
    value="${value%\"}"; value="${value#\"}"
    value="${value%\'}"; value="${value#\'}"
    export "$key=$value"
  done < .env
fi

if [ -z "${LOCAL_LEROBOT_DATA_ROOT:-}" ]; then
  echo "LOCAL_LEROBOT_DATA_ROOT is not set; set it in .env (see .env.example)" >&2
  exit 1
fi
export LOCAL_LEROBOT_DATA_ROOT
export DATASET_URL=/api/local/datasets
POLICY_PYTHON="${POLICY_PYTHON:-python3}"
POLICY_PORT="${POLICY_PORT:-7862}"

if command -v "$POLICY_PYTHON" >/dev/null 2>&1; then
  "$POLICY_PYTHON" backend/policy_server.py --port "$POLICY_PORT" &
  POLICY_PID=$!
  trap 'kill $POLICY_PID 2>/dev/null || true' EXIT INT TERM
  export POLICY_BACKEND_URL="http://127.0.0.1:$POLICY_PORT"
else
  echo "POLICY_PYTHON ($POLICY_PYTHON) not found; policy evaluation disabled" >&2
fi

next dev "$@"
