from __future__ import annotations

import re
import shutil
import subprocess
import json
from pathlib import Path
from textwrap import dedent
from typing import Any
import graphviz

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

        # Заполняется при вызове build_tex
        self._depth_map: dict[str, int] = {}
        self._layout: str = "linear"

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
        title = self._balance_parentheses(diagram.get("title", ""))
        nodes = diagram.get("nodes", [])
        edges = diagram.get("edges", [])

        mapped_nodes = [self._map_node(node, i) for i, node in enumerate(nodes)]
        self._layout = diagram.get("layout", "linear")
        ordered_nodes = self._compute_layout(mapped_nodes, edges, self._layout)

        node_blocks: list[str] = []
        edge_blocks: list[str] = []

        for i, node in enumerate(ordered_nodes):
            node_blocks.append(self._build_node_tex(node, i))

        valid_ids = {n["id"] for n in ordered_nodes}
        for edge in edges:
            source = self._sanitize_id(str(edge.get("source", "")))
            target = self._sanitize_id(str(edge.get("target", "")))
            if source in valid_ids and target in valid_ids:
                edge_blocks.append(self._build_edge_tex(source, target))

        tex = f"""
\\documentclass[border=15pt, multi, tikz]{{standalone}}
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
\\tikzstyle{{skip}}=[ultra thick,every node/.style={{sloped,allow upside down}},draw=black!40,densely dashed,opacity=0.6]

{chr(10).join(node_blocks)}

{chr(10).join(edge_blocks)}
"""
        if title:
            tex += f"\\node[above=1.5cm, align=center] at (current bounding box.north) {{\\Large\\bfseries\\sffamily {self._escape_latex(title)}}};\n"

        tex += "\\end{tikzpicture}\n\\end{document}\n"
        return dedent(tex)

    # ─────────────────────── validation ───────────────────────

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

    # ─────────────────────── node mapping ───────────────────────

    def _map_node(self, node: dict[str, Any], index: int) -> dict[str, Any]:
        node_id = self._sanitize_id(str(node.get("id", f"layer_{index}")))

        raw_label = node.get("label", f"Layer {index}")
        label = self._normalize_label(raw_label)
        label = self._shorten_label(label, max_words=3, max_chars=26)

        kind = str(node.get("kind", "")).strip().lower()

        if not kind:
            kind = self._infer_kind_from_label(label)

        block: dict[str, Any] = {
            "id": node_id,
            "label": label,
            "kind": kind,
        }

        # ── Размеры блоков уменьшены для компактности ──
        if kind == "input":
            block.update(macro="Box", params={"width": 1.5, "height": 20, "depth": 20, "fill": "green"})
        elif kind in ["conv", "cnn"]:
            block.update(macro="RightBandedBox", params={"width": "{1.5}", "height": 17, "depth": 17, "fill": "yellow", "bandfill": "orange"})
        elif kind in ["pool", "maxpool", "avgpool"]:
            block.update(macro="Box", params={"width": 1.2, "height": 14, "depth": 14, "fill": "red", "opacity": 0.5})
        elif kind in ["fc", "dense", "linear"]:
            block.update(macro="Box", params={"width": 1.5, "height": 8, "depth": 8, "fill": "cyan"})
        elif kind in ["sum", "add", "concat", "mul", "dot"]:
            logo_map = {"sum": "$+$", "add": "$+$", "concat": "©", "mul": "$\\times$", "dot": "$\\circ$"}
            block.update(macro="Ball", params={"radius": 1.5, "fill": "green", "logo": logo_map.get(kind, "$+$")})
        elif kind == "output":
            block.update(macro="Box", params={"width": 1.5, "height": 10, "depth": 10, "fill": "magenta"})
        else:
            block.update(macro="Box", params={"width": 2.5, "height": 16, "depth": 16, "fill": "blue"})

        # Разрешаем переопределять из JSON
        for key in ("width", "height", "depth", "fill"):
            if key in node:
                block["params"][key] = node[key]

        return block

    # ─────────────────────── layout ───────────────────────

    def _compute_layout(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        layout: str = "linear",
    ) -> list[dict[str, Any]]:
        if not nodes:
            return nodes

        # ── Топологическая глубина ──
        out_edges: dict[str, list[str]] = {n["id"]: [] for n in nodes}
        in_edges: dict[str, list[str]] = {n["id"]: [] for n in nodes}
        for e in edges:
            u = self._sanitize_id(str(e.get("source", "")))
            v = self._sanitize_id(str(e.get("target", "")))
            if u in out_edges and v in out_edges:
                out_edges[u].append(v)
                in_edges[v].append(u)

        depth: dict[str, int] = {n["id"]: 0 for n in nodes}
        changed = True
        while changed:
            changed = False
            for u in depth:
                for v in out_edges[u]:
                    if depth[v] < depth[u] + 1:
                        depth[v] = depth[u] + 1
                        changed = True

        self._depth_map = depth

        # ── Graphviz layout ──
        dot = graphviz.Digraph(engine="dot")

        if layout == "u_shape":
            dot.attr(rankdir='LR', ranksep='1.0', nodesep='0.8')
        else:
            dot.attr(rankdir='LR', ranksep='0.8', nodesep='0.6')

        for n in nodes:
            dot.node(n["id"], shape="box", width="1.0", height="1.0")

        valid_ids = {n["id"] for n in nodes}
        for e in edges:
            s_id = self._sanitize_id(str(e.get("source", "")))
            t_id = self._sanitize_id(str(e.get("target", "")))
            if s_id in valid_ids and t_id in valid_ids:
                if depth[t_id] - depth[s_id] > 1:
                    # Skip-connections не влияют на layout engine
                    dot.edge(s_id, t_id, constraint="false")
                else:
                    dot.edge(s_id, t_id)

        try:
            out = dot.pipe(format="json").decode("utf-8")
            data = json.loads(out)
        except Exception:
            for i, n in enumerate(nodes):
                n["x"] = i * 3.0
                n["y"] = 0.0
            return nodes

        pos_map: dict[str, tuple[float, float]] = {}
        for obj in data.get("objects", []):
            name = obj.get("name")
            pos_str = obj.get("pos", "0,0")
            if pos_str and not pos_str.startswith("e,"):
                try:
                    px, py = map(float, pos_str.replace("e,", "").split(","))
                    pos_map[name] = (px / 25.0, py / 25.0)
                except ValueError:
                    pass

        # ── U-морфинг: края ВВЕРХ, центр ВНИЗ ──
        if layout == "u_shape" and pos_map:
            xs = [p[0] for p in pos_map.values()]
            min_x, max_x = min(xs), max(xs)
            center_x = (min_x + max_x) / 2.0

            U_FACTOR = 0.55
            for name, (px, py) in pos_map.items():
                # Минус: чем ближе к центру, тем ниже. Края поднимаются.
                morph_y = py - abs(px - center_x) * U_FACTOR
                pos_map[name] = (px, morph_y)

        for n in nodes:
            x, y = pos_map.get(n["id"], (0.0, 0.0))
            n["x"] = x
            n["y"] = y

        return nodes

    # ─────────────────────── label processing ───────────────────────

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

    def _balance_parentheses(self, text: str) -> str:
        opened = text.count('(')
        closed = text.count(')')
        if opened > closed:
            text += ')' * (opened - closed)
        return text

    def _shorten_label(self, label: str | None, max_words: int = 20, max_chars: int = 45) -> str:
        if not label:
            return "Block"

        label = str(label).strip()
        if not label:
            return "Block"

        protected = {
            "Input", "Conv Stem", "CNN Blocks",
            "Transformer Blocks", "Fusion",
            "Cls Head", "Output", "FC"
        }

        if label in protected:
            return self._balance_parentheses(label)

        if len(label) <= max_chars:
            return self._balance_parentheses(label)

        truncated = label[:max_chars].rsplit(' ', 1)[0]
        return self._balance_parentheses(truncated) + "..."

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

    # ─────────────────────── TeX generation ───────────────────────

    def _build_node_tex(self, node: dict[str, Any], index: int) -> str:
        name = node["id"]
        caption = self._escape_latex(node["label"])
        macro = node.get("macro", "Box")
        params = node.get("params", {})

        params_str = f"name={name}, caption={{{caption}}}"
        for k, v in params.items():
            if k == "xlabel":
                params_str += f", {k}={{{v}}}"
            else:
                params_str += f", {k}={v}"

        x = node.get("x", index * 3.0)
        y = node.get("y", 0.0)

        return f"""\
\\pic[shift={{({x:.2f},{y:.2f},0)}}] at (0,0,0)
    {{{macro}={{
        {params_str}
    }}}};"""

    def _build_edge_tex(self, source: str, target: str) -> str:
        s_depth = self._depth_map.get(source, 0)
        t_depth = self._depth_map.get(target, 0)
        gap = t_depth - s_depth

        if gap <= 1:
            # Обычное прямое соединение (соседние блоки)
            return f"\\draw [connection] ({source}-east) -- node {{\\midarrow}} ({target}-west);"

        # Skip-connection: polyline path СВЕРХУ блоков (не дуга — избегаем TikZ "Dimension too large")
        # Рисуем: source-north → вверх на lift → горизонтально → вниз к target-north
        lift = 1.5 + gap * 0.6  # чем длиннее прыжок, тем выше дуга
        return (
            f"\\draw [skip] ({source}-north) -- "
            f"++(0,{lift:.1f},0) -| "
            f"node[pos=0.25] {{\\midarrow}} "
            f"({target}-north);"
        )

    # ─────────────────────── compilation ───────────────────────

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

    # ─────────────────────── utils ───────────────────────

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
