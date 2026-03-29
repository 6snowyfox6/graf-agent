---
trigger: always_on
---

# LLM API rules

- Use OpenAI-compatible chat completions contract
- Print server error body before raising on HTTP 4xx/5xx
- Keep model names configurable
- Never assume model IDs; inspect `/v1/models` when debugging
- Avoid breaking existing prompt -> JSON extraction flow