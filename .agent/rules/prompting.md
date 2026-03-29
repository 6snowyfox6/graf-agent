---
trigger: always_on
---

# Prompting rules

- For model architectures, do not require the user to list every block manually
- Ask the model to infer key architectural modules
- Reduce low-level details like ReLU, BatchNorm, kernel size, stride
- Target 5-8 high-level architecture blocks
- Prefer semantic modules over implementation details