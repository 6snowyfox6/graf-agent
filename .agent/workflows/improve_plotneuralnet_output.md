---
description: 
---

---
description: Improve PlotNeuralNet output quality safely
---

1. Inspect the latest generated PDF or output artifact
2. Identify whether the problem is:
   - prompt contract
   - label normalization
   - block sizing
   - renderer routing
3. Prefer fixing:
   - block widths
   - controlled label normalization
   - safe shortening
4. Avoid replacing unknown labels with generic `Block` unless empty
5. Re-run syntax checks
6. Summarize exactly what changed
7. Write a short handoff entry in `.agent/workflows/handoff_log.md`
