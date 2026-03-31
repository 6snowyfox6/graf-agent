import os
from huggingface_hub import hf_hub_download

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

MODELS = [
    {
        "repo_id": "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "description": "Qwen2.5 7B (генератор)",
    },
    {
        "repo_id": "bartowski/google_gemma-3-4b-it-GGUF",
        "filename": "google_gemma-3-4b-it-Q4_K_M.gguf",
        "description": "Gemma 3 4B (критик / vision)",
    },
]

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    for model in MODELS:
        dest = os.path.join(MODELS_DIR, model["filename"])
        if os.path.exists(dest):
            print(f"✅ {model['description']} уже скачана: {dest}")
            continue

        print(f"Скачивание {model['description']} ({model['filename']})...")
        model_path = hf_hub_download(
            repo_id=model["repo_id"],
            filename=model["filename"],
            local_dir=MODELS_DIR,
        )
        print(f"✅ Загружена в: {model_path}")

    print("\nВсе модели готовы! Запустите серверы:")
    print("  ./scripts/start_server.sh        # Qwen на порту 8000")
    print("  ./scripts/start_server_gemma.sh   # Gemma на порту 8001")

if __name__ == "__main__":
    main()
