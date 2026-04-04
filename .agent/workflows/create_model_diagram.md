---
description: 
---

---
description: Generate and improve a model architecture diagram without breaking Graphviz modes
---

1. Inspect `PROJECT_CONTEXT.md`
2. Inspect `diagram_types/model_architecture.json`
3. Inspect `main.py` routing for renderer selection
4. Inspect `plotneuralnet_renderer.py`
5. If labels are too long, improve normalization before touching prompt rules
6. Keep output compact and article-like
7. Do not break `general` or `pipeline` rendering
8. Update `.agent/workflows/handoff_log.md` with:
   - what changed
   - which outputs were regenerated
   - known issues and next recommended step
