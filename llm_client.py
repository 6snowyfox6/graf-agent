"""Thin wrapper around the local llama-cpp-python OpenAI-compatible endpoints."""

from __future__ import annotations

from typing import Any

import requests

from config import (
    DEFAULT_LLM_TIMEOUT,
    GEMMA_URL,
    GENERATOR_MODEL,
    get_endpoint,
)


def ask_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: tuple[int, int] = DEFAULT_LLM_TIMEOUT,
    temperature: float = 0.15,
    max_tokens: int | None = 1024,
    response_format: dict[str, Any] | None = None,
) -> str:
    """Send a chat-completion request and return the assistant message."""
    endpoint = get_endpoint(model)
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if response_format is not None:
        payload["response_format"] = response_format

    try:
        response = requests.post(endpoint, json=payload, timeout=timeout)
    except requests.exceptions.ReadTimeout:
        # Fallback for long-running local generator on port 8000:
        # try critic endpoint once to avoid full pipeline failure.
        if model == GENERATOR_MODEL and endpoint != GEMMA_URL:
            print("[WARN] generator timeout on primary endpoint, retrying via fallback endpoint 8001")
            response = requests.post(GEMMA_URL, json=payload, timeout=timeout)
        else:
            raise
    if response.status_code >= 400:
        print(f"[LLM ERROR] {response.status_code}: {response.text}")
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]
