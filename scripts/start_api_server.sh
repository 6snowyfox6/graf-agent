#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
APP_MODULE="${APP_MODULE:-openai_api:app}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"

cd "$REPO_ROOT"

echo "[INFO] Repo root: $REPO_ROOT"
echo "[INFO] Host: $HOST"
echo "[INFO] Port: $PORT"
echo "[INFO] App:  $APP_MODULE"

if [[ -d "$VENV_DIR" ]]; then
  echo "[INFO] Activating venv: $VENV_DIR"
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
else
  echo "[WARN] Virtual environment not found: $VENV_DIR"
  echo "[WARN] Continuing with system Python"
fi

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "$REPO_ROOT/openai_api.py" ]]; then
  echo "[ERROR] File not found: $REPO_ROOT/openai_api.py"
  exit 1
fi

python - <<'PY'
import importlib
mods = ["fastapi", "uvicorn"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
if missing:
    raise SystemExit(
        "[ERROR] Missing packages: " + ", ".join(missing) +
        "\nInstall them with:\npython -m pip install fastapi \"uvicorn[standard]\""
    )
print("[INFO] Python dependencies are OK")
PY

mkdir -p "$REPO_ROOT/api_runs"

echo "[INFO] Starting API server..."
echo "[INFO] Docs:  http://127.0.0.1:$PORT/docs"
echo "[INFO] Models: http://127.0.0.1:$PORT/v1/models"

exec python -m uvicorn "$APP_MODULE" --host "$HOST" --port "$PORT"