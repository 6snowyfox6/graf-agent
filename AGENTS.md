# AGENTS.md

## Project
AI agent for generating:
- general diagrams
- pipeline diagrams
- neural network architecture diagrams via PlotNeuralNet

## Core architecture
- `main.py` orchestrates prompt -> JSON -> critique -> improve -> render
- `plotneuralnet_renderer.py` renders model architectures
- `infographic_renderer.py` renders infographics (SVG -> PNG/PDF)
- `diagram_types/model_architecture.json` must use PlotNeuralNet
- `diagram_types/infographic.json` must use InfographicRenderer
- `external/PlotNeuralNet/` is required for model rendering

## Rendering policy
- Use Graphviz for `general` and `pipeline`
- Use PlotNeuralNet for `model_architecture`
- Use InfographicRenderer for `infographic`
- Do not replace model architecture rendering with Graphviz unless explicitly asked

## Editing rules
- Prefer minimal, targeted changes
- Do not rewrite working code wholesale
- Preserve current JSON contracts unless explicitly migrating them
- Keep backward compatibility where possible

## Current project goals
- improve PlotNeuralNet output quality
- normalize labels safely
- support more model types without manually listing every layer
- later add a dedicated U-Net renderer

## Testing expectations
- after editing renderer logic, run syntax check
- test both Graphviz and PlotNeuralNet paths
- do not break `general` or `pipeline` rendering