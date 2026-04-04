#!/usr/bin/env sh
set -eu

if [ -z "${MODEL_PATH:-}" ]; then
  echo "MODEL_PATH is required"
  exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
  echo "Model file does not exist: $MODEL_PATH"
  exit 1
fi

exec uvicorn runtime:app --host 0.0.0.0 --port "${PORT:-8000}"