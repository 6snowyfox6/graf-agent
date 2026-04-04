#!/bin/bash

# ПРИМЕЧАНИЕ ПО УСТАНОВКЕ (выполните один раз перед запуском):
# pip install huggingface_hub
# pip install llama-cpp-python[server]
#
# Если у вас есть видеокарта NVIDIA и вы хотите ускорение, устанавливайте так:
# CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python[server] --upgrade

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
MODEL_PATH="$DIR/../models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
CUDA_DIR="/usr/local/cuda-12.8"

if [ -d "$CUDA_DIR" ]; then
    export CUDA_HOME="$CUDA_DIR"
    export PATH="$CUDA_DIR/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_DIR/lib64:${LD_LIBRARY_PATH:-}"
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Модель не найдена по пути: $MODEL_PATH"
    echo "Сначала запустите: python scripts/download_models.py"
    exit 1
fi

echo "Запускаем локальный OpenAI-совместимый сервер через llama_cpp.server..."
# n_gpu_layers=-1 означает "выгрузить максимум слоев на видеокарту, если она доступна"
# host=0.0.0.0 позволяет отправлять запросы и с других устройств в локальной сети
python3 -m llama_cpp.server \
    --model "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --n_gpu_layers -1 \
    --n_ctx 16384 \
    --chat_format qwen
