#!/usr/bin/env bash
# Goku Instant Transmission webcam for macOS.
set -euo pipefail
cd "$(dirname "$0")"

VENV_PY=".venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "[Goku] Creating the Python environment..."
  python3 -m venv .venv
fi
if ! "$VENV_PY" -c "import cv2, mediapipe, numpy, pyvirtualcam" >/dev/null 2>&1; then
  echo "[Goku] Installing dependencies (needs internet once)..."
  "$VENV_PY" -m pip install --upgrade pip
  "$VENV_PY" -m pip install -r requirements.txt
fi
exec "$VENV_PY" goku_cam.py "$@"
