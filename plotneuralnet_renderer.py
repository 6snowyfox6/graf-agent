from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from textwrap import dedent
from typing import Any


class PlotNeuralNetRenderError(RuntimeError):
    pass


class PlotNeuralNetRenderer:
    """
    Автономный рендерер схем нейросетей через PlotNeuralNet.

    Ожидаемая структура:
    project_root/
    ├── main.py
    ├── plotneuralnet_renderer.py
    └── external/
        └── PlotNeuralNet/
            └── layers/
                └── init.tex
    """

    def __init__(
        self,
        project_root: str | Path | None = None,
        plotneuralnet_root: str | Path | None = None,
        output_root: str | Path | None = None,
        latex_command: str = "pdflatex",
    ) -> None:
        if project_root is None:
            project_root = Path(__file__).resolve().parent
        self.project_root = Path(project_root).resolve()

        if plotneuralnet_root is None:
            plotneuralnet_root = self.project_root / "external" / "PlotNeuralNet"
        self.plotneuralnet_root = Path(plotneuralnet_root).resolve()

        self.layers_dir = self.plotneuralnet_root / "layers"

        if output_root is None:
            output_root = self.project_root / "outputs"
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.latex_command = latex_command

    def render(self, diagram: dict[str, Any], output_name: str = "diagram") -> dict[str, str]:
        self._validate_environment()
        self._validate_diagram(diagram)

        safe_name = self._sanitize_filename(output_name)
        work_dir = self.output_root / safe_name
        work_dir.mkdir(parents=True, exist_ok=True)

        tex_path = work_dir / f"{safe_name}.tex"
        pdf_path = work_dir / f"{safe_name}.pdf"

        tex_content = self.build_tex(diagram)
        tex_path.write_text(tex_content, encoding="utf-8")

        self._compile_tex(tex_path)

        if not pdf_path.exists():
            raise PlotNeuralNetRenderError(f"PDF не был создан после компиляции: {pdf_path}")

        return {
            "status": "ok",
            "tex_path": str(tex_path),
            "pdf_path": str(pdf_path),
            "work_dir": str(work_dir),
        }

    def build_tex(self, diagram: dict[str, Any]) -> str:
        title = diagram.get("title", "Neural Architecture")
        nodes = diagram.get("nodes", [])
        edges = diagram.get("edges", [])

        mapped_nodes = [self._map_node(node, i) for i, node in enumerate(nodes)]
        ordered_nodes = self._topological_fallback_order(mapped_nodes, edges)

        node_blocks: list[str] = []
        edge_blocks: list[str] = []
        placed_ids: list[str] = []

        for i, node in enumerate(ordered_nodes):
            node_blocks.append(self._build_node_tex(node, i, placed_ids))
            placed_ids.append(node["id"])

        valid_ids = {n["id"] for n in ordered_nodes}
        for edge in edges:
            source = self._sanitize_id(str(edge.get("source", "")))
            target = self._sanitize_id(str(edge.get("target", "")))
            if source in valid_ids and target in valid_ids:
                edge_blocks.append(self._build_edge_tex(source, target))

        tex = f"""
\\documentclass[border=8pt, multi, tikz]{{standalone}}
\\usepackage[T2A]{{fontenc}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[russian,english]{{babel}}
\\usepackage{{import}}
\\subimport{{{self.layers_dir.as_posix()}/}}{{init}}
\\usetikzlibrary{{positioning}}
\\usetikzlibrary{{3d}}

\\begin{{document}}
\\begin{{tikzpicture}}
\\tikzstyle{{connection}}=[ultra thick,every node/.style={{sloped,allow upside down}},draw=black!70,opacity=0.7]

\\node[anchor=west] at (-1, 3.2, 0) {{\\Large \\bfseries {self._escape_latex(title)}}};

{chr(10).join(node_blocks)}

{chr(10).join(edge_blocks)}

\\end{{tikzpicture}}
\\end{{document}}
"""
        return dedent(tex)

    def _validate_environment(self) -> None:
        if not self.plotneuralnet_root.exists():
            raise PlotNeuralNetRenderError(
                f"Не найдена папка PlotNeuralNet: {self.plotneuralnet_root}"
            )

        if not self.layers_dir.exists():
            raise PlotNeuralNetRenderError(f"Не найдена папка layers: {self.layers_dir}")

        init_tex = self.layers_dir / "init.tex"
        if not init_tex.exists():
            raise PlotNeuralNetRenderError(f"Не найден init.tex: {init_tex}")

        if shutil.which(self.latex_command) is None:
            raise PlotNeuralNetRenderError(
                f"Команда LaTeX не найдена в PATH: {self.latex_command}"
            )

    def _validate_diagram(self, diagram: dict[str, Any]) -> None:
        if not isinstance(diagram, dict):
            raise PlotNeuralNetRenderError("diagram должен быть dict")

        nodes = diagram.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            raise PlotNeuralNetRenderError("diagram['nodes'] должен быть непустым списком")

        for i, node in enumerate(nodes):
            if not isinstance(node, dict):
                raise PlotNeuralNetRenderError(f"Узел #{i} должен быть dict")
            if "id" not in node:
                raise PlotNeuralNetRenderError(f"Узел #{i} не содержит 'id'")
            if "label" not in node:
                raise PlotNeuralNetRenderError(f"Узел #{i} не содержит 'label'")

        edges = diagram.get("edges", [])
        if not isinstance(edges, list):
            raise PlotNeuralNetRenderError("diagram['edges'] должен быть списком")

    def _map_node(self, node: dict[str, Any], index: int) -> dict[str, Any]:
        node_id = self._sanitize_id(str(node.get("id", f"layer_{index}")))

        raw_label = node.get("label", f"Layer {index}")
        label = self._normalize_label(raw_label)
        label = self._shorten_label(label, max_words=3, max_chars=26)

        kind = str(node.get("kind", "")).strip().lower()

        if not kind:
            kind = self._infer_kind_from_label(label)

        block = {
            "id": node_id,
            "label": label,
            "kind": kind,
            "width": 2.0,
            "height": 24,
            "depth": 24,
            "fill": "yellow",
        }

        if kind == "input":
            block.update(width=2.0, height=32, depth=32, fill="green")
        elif kind == "conv":
            block.update(width=3.4, height=26, depth=26, fill="yellow")
        elif kind == "pool":
            block.update(width=1.8, height=20, depth=20, fill="red")
        elif kind == "block":
            block.update(width=3.8, height=24, depth=24, fill="blue")
        elif kind == "fc":
            block.update(width=2.6, height=16, depth=16, fill="cyan")
        elif kind == "output":
            block.update(width=2.0, height=14, depth=14, fill="magenta")

        for key in ("width", "height", "depth", "fill"):
            if key in node:
                block[key] = node[key]

        return block
    def _shorten_label(self, label: str | None, max_words: int = 3, max_chars: int = 26) -> str:
        if label is None:
            return "Block"

        label = str(label).strip()
        if not label:
            return "Block"

        protected = {
            "Input",
            "Conv Stem",
            "CNN Blocks",
            "Transformer Blocks",
            "Fusion",
            "Cls Head",
            "Output",
            "FC"
        }

        if label in protected:
            return label

        if len(label) <= max_chars:
            return label

        words = label.split()
        if len(words) <= max_words:
            return label

        return " ".join(words[:max_words])
    def _infer_kind_from_label(self, label: str) -> str:
        text = label.lower()

        if any(x in text for x in ("input", "вход", "image", "img", "tokens")):
            return "input"

        if any(x in text for x in ("pool", "avgpool", "maxpool", "global pool")):
            return "pool"

        if any(x in text for x in ("fc", "linear", "dense", "mlp", "projection")):
            return "fc"

        if any(x in text for x in (
            "output", "softmax", "classifier", "class", "выход",
            "segmentation", "mask", "detection", "boxes", "logits"
        )):
            return "output"

        if any(x in text for x in (
            "encoder", "decoder", "transformer", "bottleneck", "block",
            "backbone", "neck", "head", "stage", "attention",
            "residual", "resblock", "unet", "fusion", "generator", "discriminator"
        )):
            return "block"

        if any(x in text for x in (
            "conv", "stem", "patch embedding", "embedding"
        )):
            return "conv"

        return "block"

    def _topological_fallback_order(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not edges:
            return nodes

        id_to_node = {n["id"]: n for n in nodes}
        incoming = {n["id"]: 0 for n in nodes}
        outgoing = {n["id"]: [] for n in nodes}

        for edge in edges:
            s = self._sanitize_id(str(edge.get("source", "")))
            t = self._sanitize_id(str(edge.get("target", "")))
            if s in id_to_node and t in id_to_node:
                outgoing[s].append(t)
                incoming[t] += 1

        starts = [nid for nid, deg in incoming.items() if deg == 0]
        if len(starts) != 1:
            return nodes

        ordered: list[dict[str, Any]] = []
        current = starts[0]
        visited = set()

        while current and current not in visited:
            visited.add(current)
            ordered.append(id_to_node[current])
            next_nodes = outgoing.get(current, [])
            if len(next_nodes) != 1:
                break
            current = next_nodes[0]

        if len(ordered) == len(nodes):
            return ordered

        return nodes
    def _normalize_label(self, label: str | None) -> str:
        if label is None:
            return "Block"

        text = str(label).strip()
        if not text:
            return "Block"

        replacements = {
            "Input Layer": "Input",
            "Initial Convolutional Layers": "Conv Stem",
            "Initial Conv Layers": "Conv Stem",
            "Initial Convolution": "Conv Stem",
            "Stem": "Conv Stem",
            "Stacked CNN": "CNN Blocks",
            "CNN Layers": "CNN Blocks",
            "Conv Layers": "CNN Blocks",
            "Transformer Encoder": "Transformer Blocks",
            "Transformer Encoder Blocks": "Transformer Blocks",
            "Transformer Layers": "Transformer Blocks",
            "Feature Fusion": "Fusion",
            "Feature Fusion Module": "Fusion",
            "Feature Fusion Layer": "Fusion",
            "Classification Head": "Cls Head",
            "Classifier Head": "Cls Head",
            "Output Layer": "Output",
            "Output (Softmax)": "Output"
        }

        return replacements.get(text, text)
    def _build_node_tex(self, node: dict[str, Any], index: int, placed_ids: list[str]) -> str:
        name = node["id"]
        caption = self._escape_latex(node["label"])
        fill = node["fill"]
        width = node["width"]
        height = node["height"]
        depth = node["depth"]    
        if index == 0:
            return f"""
\\pic[shift={{(0,0,0)}}] at (0,0,0)
    {{Box={{
        name={name},
        caption={{{caption}}},
        fill={fill},
        height={height},
        width={width},
        depth={depth}
    }}}};
""".strip()

        prev = placed_ids[-1]
        return f"""
\\pic[shift={{(2.2,0,0)}}] at ({prev}-east)
    {{Box={{
        name={name},
        caption={{{caption}}},
        fill={fill},
        height={height},
        width={width},
        depth={depth}
    }}}};
""".strip()

    def _build_edge_tex(self, source: str, target: str) -> str:
        return f"""
\\draw [connection] ({source}-east) -- node {{\\midarrow}} ({target}-west);
""".strip()

    def _compile_tex(self, tex_path: Path) -> None:
        cmd = [
            self.latex_command,
            "-interaction=nonstopmode",
            "-halt-on-error",
            tex_path.name,
        ]

        result = subprocess.run(
            cmd,
            cwd=tex_path.parent,
            capture_output=True,
            text=False,
        )

        if result.returncode != 0:
            out_str = result.stdout.decode('utf-8', errors='replace') if result.stdout else ""
            err_str = result.stderr.decode('utf-8', errors='replace') if result.stderr else ""
            raise PlotNeuralNetRenderError(
                "Ошибка компиляции LaTeX.\n"
                f"STDOUT:\n{out_str}\n\n"
                f"STDERR:\n{err_str}"
            )

    def _sanitize_id(self, value: str) -> str:
        value = value.lower().strip()
        value = re.sub(r"[^a-z0-9_]+", "_", value)
        value = re.sub(r"_+", "_", value)
        return value.strip("_") or "layer"

    def _sanitize_filename(self, value: str) -> str:
        value = value.strip()
        value = re.sub(r"[^\w\-\.]+", "_", value, flags=re.UNICODE)
        value = re.sub(r"_+", "_", value)
        return value.strip("._") or "diagram"

    def _escape_latex(self, text: str) -> str:
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
