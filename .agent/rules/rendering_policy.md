---
trigger: always_on
---

# Rendering policy

- `layout_hint=model_architecture` must route to PlotNeuralNet
- `layout_hint=infographic` must route to InfographicRenderer
- `layout_hint=pipeline` must route to pipeline Graphviz renderer
- all other diagrams default to general Graphviz renderer
- keep renderer and layout_hint consistent when improving JSON