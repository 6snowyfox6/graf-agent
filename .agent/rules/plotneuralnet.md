---
trigger: always_on
---

# PlotNeuralNet rules

- PlotNeuralNet is only for `model_architecture`
- Keep labels short and semantically meaningful
- Normalize known long labels into a controlled vocabulary
- Do not aggressively collapse all unknown labels into `Block`
- Prefer names like:
  - Input
  - Conv Stem
  - CNN Blocks
  - Transformer Blocks
  - Fusion
  - Cls Head
  - Output
- Increase block width before introducing more aggressive shortening