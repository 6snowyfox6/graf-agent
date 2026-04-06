# PROJECT_CONTEXT.md

## Goal
Build an AI diagram agent that supports:
1. general system diagrams
2. pipeline diagrams
3. neural model architecture diagrams with PlotNeuralNet
4. infographic diagrams (SVG-based)

## Current status
- LLM connection works through OpenAI-compatible API
- remote Windows-hosted server tested successfully
- Graphviz rendering works
- PlotNeuralNet rendering works
- Infographic SVG rendering works (PNG/PDF via cairosvg)
- model architecture mode is connected
- current bottleneck is label normalization and layout quality

## Key files
- `main.py`
- `plotneuralnet_renderer.py`
- `infographic_renderer.py`
- `diagram_types/model_architecture.json`
- `diagram_types/infographic.json`

## Current design decisions
- `general` -> Graphviz
- `pipeline` -> Graphviz
- `model_architecture` -> PlotNeuralNet
- `infographic` -> InfographicRenderer (SVG)

## Known issues
- labels may wrap badly on blocks
- aggressive shortening can destroy semantic meaning
- need a safer normalization strategy
- U-Net-like layouts need a dedicated renderer, not the generic linear renderer

## Desired direction
- user should not manually list all blocks
- LLM should infer major architectural modules
- support hybrid / new / unknown model families
- keep architecture outputs compact and article-like

## Critic Influence Analytics (SHAP)
- Pipeline now supports optional critic influence analysis between `critique` and `improve`.
- Per-run artifacts are written to `outputs/<run_id>/`:
  - `critic_influence_report.json`
  - `critic_influence_summary.md`
  - optional SHAP plots: `critic_shap_beeswarm.png`, `critic_shap_bar.png`
- History is accumulated in `outputs/_critic_influence_history.jsonl`.
- Interpretation is surrogate-based:
  - features come from critique fields (scores, counts, text sizes),
  - targets come from draft->final structural deltas (node/edge/label/kind changes),
  - SHAP is used when available; safe fallback is used otherwise.
- CLI switch:
  - `--explain-critic-influence on|off` (default: `on`)
  - `--no-auto-servers` to run without server auto-start.
