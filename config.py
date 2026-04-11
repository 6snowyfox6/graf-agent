"""Centralised configuration for graf-agent.

All tunable constants (model endpoints, timeouts, limits) live here
so that they can be overridden via environment variables or a single
config file instead of hard-coding them across multiple modules.
"""

from __future__ import annotations

import os


# ---------------------------------------------------------------------------
# Model server endpoints
# ---------------------------------------------------------------------------
QWEN_PORT = int(os.getenv("GRAF_QWEN_PORT", "8000"))
GEMMA_PORT = int(os.getenv("GRAF_GEMMA_PORT", "8001"))
QWEN_URL = f"http://127.0.0.1:{QWEN_PORT}/v1/chat/completions"
GEMMA_URL = f"http://127.0.0.1:{GEMMA_PORT}/v1/chat/completions"

# ---------------------------------------------------------------------------
# Model identifiers (used in payload["model"])
# ---------------------------------------------------------------------------
GENERATOR_MODEL = os.getenv("GRAF_GENERATOR_MODEL", "qwen2.5-7b-instruct")
VISION_MODEL = os.getenv("GRAF_VISION_MODEL", "gemma-3-4b-it")
CRITIC_MODEL = os.getenv("GRAF_CRITIC_MODEL", "gemma-3-4b-it")

MODEL_ENDPOINTS: dict[str, str] = {
    GENERATOR_MODEL: QWEN_URL,
    VISION_MODEL: GEMMA_URL,
    CRITIC_MODEL: GEMMA_URL,
}


def get_endpoint(model: str) -> str:
    """Return the HTTP endpoint for *model*, falling back to Qwen URL."""
    return MODEL_ENDPOINTS.get(model, QWEN_URL)


# ---------------------------------------------------------------------------
# Timeouts (connect_seconds, read_seconds)
# ---------------------------------------------------------------------------
DEFAULT_LLM_TIMEOUT: tuple[int, int] = (30, 300)
VISION_TIMEOUT: tuple[int, int] = (30, 120)
GENERATOR_TIMEOUT: tuple[int, int] = (30, 180)

# ---------------------------------------------------------------------------
# Token limits
# ---------------------------------------------------------------------------
GENERATOR_MAX_TOKENS_GENERAL = 1100
GENERATOR_MAX_TOKENS_ARCHITECTURE = 2048

# ---------------------------------------------------------------------------
# API protection
# ---------------------------------------------------------------------------
MAX_PROMPT_CHARS = 16_000
PIPELINE_TIMEOUT_SEC = 600  # subprocess hard kill
MAX_CONCURRENT_RUNS = 2
