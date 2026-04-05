#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python3}"
MODEL_PATH="$PROJECT_ROOT/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"

echo "=== start_server.sh ==="
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "CUDA_HOME=$CUDA_HOME"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "MODEL_PATH=$MODEL_PATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python не найден: $PYTHON_BIN"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Модель не найдена по пути: $MODEL_PATH"
    echo "Сначала скачай модель."
    exit 1
fi

"$PYTHON_BIN" -c "import sys; print('python =', sys.executable)"
"$PYTHON_BIN" -c "import llama_cpp; print('llama_cpp =', llama_cpp.__file__)"

exec "$PYTHON_BIN" -m llama_cpp.server \
    --model "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port 8000 \
    --n_gpu_layers 18 \
    --n_ctx 4096 \
    --n_batch 128 \
    --n_ubatch 128 \
    --chat_format qwen
