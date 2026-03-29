# GEMINI.md

## Antigravity-specific priorities
- Always inspect `PROJECT_CONTEXT.md` before major edits
- Prefer editing existing files over creating parallel duplicates
- When changing renderer behavior, explain which rendering mode is affected
- If changing prompt contracts, update `docs/prompt_contracts.md`
- If changing model architecture behavior, inspect:
  - `diagram_types/model_architecture.json`
  - `plotneuralnet_renderer.py`
  - `main.py`

## Safety rules
- Never remove support for Graphviz general diagrams
- Never hardcode OS-specific paths unless explicitly requested
- Prefer configurable paths for LaTeX and PlotNeuralNet roots
- Keep labels short and readable in PlotNeuralNet output

## Output quality policy
- Prefer semantic block names over low-level operation names
- Use controlled normalization for model labels
- Do not aggressively collapse unknown labels into `Block`