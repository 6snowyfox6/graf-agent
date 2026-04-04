#!/bin/bash

# Сервер Gemma 3 4B — критик и vision
# Запускается на порту 8001

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
MODEL_PATH="$DIR/../models/google_gemma-3-4b-it-Q4_K_M.gguf"
CUDA_DIR="/usr/local/cuda-12.8"

if [ -d "$CUDA_DIR" ]; then
    export CUDA_HOME="$CUDA_DIR"
    export PATH="$CUDA_DIR/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_DIR/lib64:${LD_LIBRARY_PATH:-}"
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Модель Gemma не найдена: $MODEL_PATH"
    echo "Сначала запустите: python3 scripts/download_models.py"
    exit 1
fi

echo "Запускаем Gemma 3 4B на порту 8001..."
python3 -m llama_cpp.server \
    --model "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8001 \
    --n_gpu_layers -1 \
    --n_ctx 8192 \
    --chat_format chatml
