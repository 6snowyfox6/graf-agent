from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from infographic_renderer import InfographicRenderer
from plotneuralnet_renderer import build_model_renderer


GeneralRendererFn = Callable[[dict[str, Any], str, str | Path], Any]
PipelineRendererFn = Callable[[dict[str, Any], str, str | Path], Any]


def _normalize_output_root(output_name: str, output_dir: str | Path) -> Path:
    output_root = Path(output_dir)
    # PlotNeuralNet/Infographic renderers create a subfolder with output_name.
    # If we already passed outputs/<run>, avoid outputs/<run>/<run>/<run>.
    if output_root.name == str(output_name):
        return output_root.parent
    return output_root


def render_diagram(
    diagram: dict[str, Any],
    output_name: str,
    output_dir: str | Path,
    render_general: GeneralRendererFn,
    render_pipeline: PipelineRendererFn,
) -> Any:
    renderer = diagram.get("renderer", "")
    layout_hint = diagram.get("layout_hint", "")
    output_root = _normalize_output_root(output_name, output_dir)

    if renderer == "plotneuralnet" or layout_hint == "model_architecture":
        plot_renderer = build_model_renderer(
            diagram,
            project_root=".",
            output_root=output_root,
        )
        return plot_renderer.render(diagram, output_name=output_name)

    if renderer == "infographic" or layout_hint == "infographic":
        info_renderer = InfographicRenderer(output_root=output_root)
        return info_renderer.render(diagram, output_name=output_name)

    if renderer == "pipeline" or layout_hint == "pipeline":
        return render_pipeline(diagram, output_name, output_dir)

    return render_general(diagram, output_name, output_dir)

