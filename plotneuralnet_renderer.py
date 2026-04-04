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
        self._skip_edges: set[tuple[str, str]] = set()
        self._merge_edges: set[tuple[str, str]] = set()
        self._node_kind_map: dict[str, str] = {}
        self._node_macro_map: dict[str, str] = {}
        self._incoming_map: dict[str, list[str]] = {}
        self._node_position_map: dict[str, tuple[float, float]] = {}
        self._layout: str = "linear"
        self._warnings: list[str] = []

    def render(self, diagram: dict[str, Any], output_name: str = "diagram") -> dict[str, str]:
        if self.__class__ is PlotNeuralNetRenderer:
            delegated = build_model_renderer(
                diagram=diagram,
                project_root=self.project_root,
                plotneuralnet_root=self.plotneuralnet_root,
                output_root=self.output_root,
                latex_command=self.latex_command,
            )
            if delegated.__class__ is not PlotNeuralNetRenderer:
                return delegated.render(diagram, output_name=output_name)

        self._validate_environment()
        self._validate_diagram(diagram)
        self._warnings = []

        safe_name = self._sanitize_filename(output_name)
        work_dir = self.output_root / safe_name
        work_dir.mkdir(parents=True, exist_ok=True)

        tex_path = work_dir / f"{safe_name}.tex"
        pdf_path = work_dir / f"{safe_name}.pdf"
        preview_path = work_dir / f"{safe_name}.png"

        contract = str(diagram.get("render_contract", "") or "").strip().lower()
        python_backend_mode = str(diagram.get("python_backend", "auto")).strip().lower()
        python_backend_enabled = python_backend_mode not in {"off", "false", "0", "no"}

        if contract == "canonical_unet":
            self._render_canonical_unet_via_python(diagram, safe_name, work_dir, tex_path)
        elif contract == "canonical_resnet":
            self._render_canonical_resnet_via_python(diagram, safe_name, work_dir, tex_path)
        elif contract == "canonical_yolo":
            self._render_canonical_yolo_via_python(diagram, safe_name, work_dir, tex_path)
        elif python_backend_enabled:
            try:
                if self._looks_like_yolo(diagram):
                    self._render_canonical_yolo_via_python(diagram, safe_name, work_dir, tex_path)
                elif self._looks_like_resnet(diagram):
                    self._render_canonical_resnet_via_python(diagram, safe_name, work_dir, tex_path)
                else:
                    self._render_generic_via_python_script(diagram, safe_name, work_dir, tex_path)
            except PlotNeuralNetRenderError as exc:
                self._warnings.append(
                    "Python backend failed, fallback to direct TeX backend: "
                    + str(exc).splitlines()[0]
                )
                tex_content = self.build_tex(diagram)
                tex_path.write_text(tex_content, encoding="utf-8")
        else:
            tex_content = self.build_tex(diagram)
            tex_path.write_text(tex_content, encoding="utf-8")

        self._compile_tex(tex_path)
        generated_preview = self._generate_preview(pdf_path, preview_path)

        if not pdf_path.exists():
            raise PlotNeuralNetRenderError(f"PDF не был создан после компиляции: {pdf_path}")

        result = {
            "status": "ok",
            "tex_path": str(tex_path),
            "pdf_path": str(pdf_path),
            "work_dir": str(work_dir),
            "warnings": list(self._warnings),
        }
        if generated_preview is not None:
            result["preview_path"] = str(generated_preview)
        return result

    def _render_generic_via_python_script(
        self,
        diagram: dict[str, Any],
        safe_name: str,
        work_dir: Path,
        tex_path: Path,
    ) -> None:
        raw_nodes = diagram.get("nodes", [])
        raw_edges = diagram.get("edges", [])
        self._layout = diagram.get("layout", "linear")

        mapped_nodes = self._prepare_nodes(raw_nodes)
        edges = self._normalize_edges(raw_edges, {node["id"] for node in mapped_nodes})
        ordered_nodes = self._compute_layout(mapped_nodes, edges, self._layout)

        node_blocks: list[str] = []
        for i, node in enumerate(ordered_nodes):
            node_blocks.append(self._build_node_tex(node, i))

        self._node_position_map = {
            node["id"]: (float(node.get("x", i * 3.0)), float(node.get("y", 0.0)))
            for i, node in enumerate(ordered_nodes)
        }

        edge_blocks: list[str] = []
        valid_ids = {n["id"] for n in ordered_nodes}
        for edge in edges:
            source = self._sanitize_id(str(edge.get("source", "")))
            target = self._sanitize_id(str(edge.get("target", "")))
            if source in valid_ids and target in valid_ids:
                edge_blocks.append(self._build_edge_tex(source, target))

        extra_blocks = self._build_extra_tex()
        py_script_path = work_dir / f"{safe_name}_gen.py"
        script = self._build_generic_python_script(
            output_tex_name=tex_path.name,
            title=self._balance_parentheses(diagram.get("title", "")),
            node_blocks=node_blocks,
            edge_blocks=edge_blocks,
            extra_blocks=extra_blocks,
            plot_root=self.plotneuralnet_root,
        )
        py_script_path.write_text(script, encoding="utf-8")

        result = subprocess.run(
            ["python3", py_script_path.name],
            cwd=work_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise PlotNeuralNetRenderError(
                "Generic python backend failed.\n"
                f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            )
        if not tex_path.exists():
            raise PlotNeuralNetRenderError(
                f"Python backend did not generate TeX: {tex_path}"
            )

    def _build_generic_python_script(
        self,
        output_tex_name: str,
        title: str,
        node_blocks: list[str],
        edge_blocks: list[str],
        extra_blocks: str,
        plot_root: Path,
    ) -> str:
        blocks: list[str] = []
        blocks.extend(node_blocks)
        blocks.extend(edge_blocks)
        if extra_blocks.strip():
            blocks.append(extra_blocks)
        if title:
            title_tex = self._escape_latex(title)
            blocks.append(
                "\\node[above=1.5cm, align=center] at (current bounding box.north) "
                f"{{\\Large\\bfseries\\sffamily {title_tex}}};"
            )

        arch_entries = ",\n".join(repr(b) for b in blocks)
        return (
            "import sys\n"
            f"sys.path.insert(0, {repr(str(plot_root))})\n"
            "from pycore.tikzeng import to_head, to_cor, to_begin, to_end, to_generate\n\n"
            "arch = [\n"
            f"    to_head({repr(str(plot_root))}),\n"
            "    to_cor(),\n"
            "    to_begin(),\n"
            f"{('    ' + arch_entries.replace('\\n', '\\n    ')) if arch_entries else ''}\n"
            "    to_end(),\n"
            "]\n\n"
            "def main():\n"
            f"    to_generate(arch, {repr(output_tex_name)})\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )

    def _render_canonical_unet_via_python(
        self,
        diagram: dict[str, Any],
        safe_name: str,
        work_dir: Path,
        tex_path: Path,
    ) -> None:
        plot_root = self.plotneuralnet_root
        py_script_path = work_dir / f"{safe_name}_gen.py"
        stage_count = self._infer_unet_stage_count(diagram)

        script = self._build_unet_python_script(
            output_tex_name=tex_path.name,
            plot_root=plot_root,
            stage_count=stage_count,
        )
        py_script_path.write_text(script, encoding="utf-8")

        result = subprocess.run(
            ["python3", py_script_path.name],
            cwd=work_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise PlotNeuralNetRenderError(
                "Ошибка генерации TeX через PlotNeuralNet python backend.\n"
                f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            )
        if not tex_path.exists():
            raise PlotNeuralNetRenderError(
                f"Python backend не сгенерировал TeX: {tex_path}"
            )

    def _looks_like_resnet(self, diagram: dict[str, Any]) -> bool:
        title = str(diagram.get("title", "")).lower()
        if "resnet" in title:
            return True
        for node in diagram.get("nodes", []):
            text = f"{node.get('id', '')} {node.get('label', '')}".lower()
            if "resnet" in text:
                return True
        return False

    def _looks_like_yolo(self, diagram: dict[str, Any]) -> bool:
        title = str(diagram.get("title", "")).lower()
        if "yolo" in title:
            return True
        for node in diagram.get("nodes", []):
            text = f"{node.get('id', '')} {node.get('label', '')}".lower()
            if "yolo" in text:
                return True
        return False

    def _infer_yolo_variant(self, diagram: dict[str, Any]) -> str:
        corpus = [str(diagram.get("title", ""))]
        for node in diagram.get("nodes", []):
            corpus.append(str(node.get("id", "")))
            corpus.append(str(node.get("label", "")))
        text = " ".join(corpus).lower()
        if "yolov3" in text or "yolo v3" in text:
            return "v3"
        if "yolov5" in text or "yolo v5" in text:
            return "v5"
        if "yolov8" in text or "yolo v8" in text:
            return "v8"
        return "v8"

    def _infer_resnet_depth(self, diagram: dict[str, Any]) -> int:
        corpus = [str(diagram.get("title", ""))]
        for node in diagram.get("nodes", []):
            corpus.append(str(node.get("id", "")))
            corpus.append(str(node.get("label", "")))
        text = " ".join(corpus).lower()
        m = re.search(r"resnet[\s_-]?(18|34|50|101|152)", text)
        if not m:
            return 18
        return int(m.group(1))

    def _render_canonical_resnet_via_python(
        self,
        diagram: dict[str, Any],
        safe_name: str,
        work_dir: Path,
        tex_path: Path,
    ) -> None:
        depth = self._infer_resnet_depth(diagram)
        py_script_path = work_dir / f"{safe_name}_gen.py"
        script = self._build_resnet_python_script(
            output_tex_name=tex_path.name,
            plot_root=self.plotneuralnet_root,
            title=self._balance_parentheses(diagram.get("title", "")),
            depth=depth,
        )
        py_script_path.write_text(script, encoding="utf-8")

        result = subprocess.run(
            ["python3", py_script_path.name],
            cwd=work_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise PlotNeuralNetRenderError(
                "Ошибка генерации ResNet TeX через PlotNeuralNet python backend.\n"
                f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            )
        if not tex_path.exists():
            raise PlotNeuralNetRenderError(
                f"ResNet python backend не сгенерировал TeX: {tex_path}"
            )

    def _build_resnet_python_script(
        self,
        output_tex_name: str,
        plot_root: Path,
        title: str,
        depth: int,
    ) -> str:
        if depth == 18:
            stage_blocks = [2, 2, 2, 2]
            bottleneck = False
        elif depth == 34:
            stage_blocks = [3, 4, 6, 3]
            bottleneck = False
        elif depth == 50:
            stage_blocks = [3, 4, 6, 3]
            bottleneck = True
        elif depth == 101:
            stage_blocks = [3, 4, 23, 3]
            bottleneck = True
        elif depth == 152:
            stage_blocks = [3, 8, 36, 3]
            bottleneck = True
        else:
            stage_blocks = [2, 2, 2, 2]
            bottleneck = False

        channels = [64, 128, 256, 512]
        bottleneck_expand = [256, 512, 1024, 2048]
        spatial = [56, 28, 14, 7]
        widths_basic = [2.8, 3.8, 5.0, 6.0]
        widths_bneck = (1.0, 1.7, 2.6)
        offsets = ["(2.0,0,0)", "(2.2,0,0)", "(2.4,0,0)", "(2.4,0,0)"]

        lines: list[str] = [
            "import sys",
            f"sys.path.insert(0, {repr(str(plot_root))})",
            "from pycore.tikzeng import *",
            "",
            "arch = [",
            f"    to_head({repr(str(plot_root))}),",
            "    to_cor(),",
            "    to_begin(),",
            "    to_Conv(name='stem', s_filer=224, n_filer=64, offset='(0,0,0)', to='(0,0,0)', width=3.2, height=64, depth=64, caption='Stem 7x7 s2'),",
            "    to_Pool(name='pool0', offset='(0,0,0)', to='(stem-east)', width=1, height=56, depth=56, opacity=0.55, caption='MaxPool'),",
            "    to_connection('stem', 'pool0'),",
        ]

        prev = "pool0"
        if not bottleneck:
            for stage_idx in range(4):
                block_count = stage_blocks[stage_idx]
                c = channels[stage_idx]
                s = spatial[stage_idx]
                w = widths_basic[stage_idx]
                offset = offsets[stage_idx]
                for block_idx in range(block_count):
                    block_name = f"s{stage_idx + 1}_b{block_idx + 1}"
                    caption = ""
                    if block_idx == 0:
                        caption = f"Stage {stage_idx + 2} x{block_count}"
                    lines.append(
                        "    to_ConvRes("
                        f"name='{block_name}', s_filer={s}, n_filer={c}, "
                        f"offset='{offset if block_idx == 0 else '(0.55,0,0)'}', "
                        f"to='({prev}-east)', width={w:.1f}, height={max(8, s):d}, depth={max(8, s):d}, "
                        "opacity=0.45"
                        f"{', caption=' + repr(caption) if caption else ''}"
                        "),"
                    )
                    lines.append(f"    to_connection('{prev}', '{block_name}'),")
                    prev = block_name
        else:
            for stage_idx in range(4):
                block_count = stage_blocks[stage_idx]
                c_mid = channels[stage_idx]
                c_out = bottleneck_expand[stage_idx]
                s = spatial[stage_idx]
                offset = offsets[stage_idx]
                h = max(8, s)
                d = max(8, s)

                for block_idx in range(block_count):
                    block_name = f"s{stage_idx + 1}_b{block_idx + 1}"
                    conv1 = f"{block_name}_r"
                    conv2 = f"{block_name}_3"
                    conv3 = f"{block_name}_e"
                    sum_name = f"{block_name}_sum"
                    projection_name = f"{block_name}_proj"
                    is_projection = block_idx == 0
                    step = offset if block_idx == 0 else "(0.85,0,0)"
                    caption = ""
                    if block_idx == 0:
                        caption = f"Bneck S{stage_idx + 2} x{block_count}"

                    lines.append(
                        "    to_Conv("
                        f"name='{conv1}', s_filer={s}, n_filer={c_mid}, "
                        f"offset='{step}', to='({prev}-east)', "
                        f"width={widths_bneck[0]:.1f}, height={h}, depth={d}"
                        f"{', caption=' + repr(caption) if caption else ''}"
                        "),"
                    )
                    lines.append(f"    to_connection('{prev}', '{conv1}'),")

                    lines.append(
                        "    to_Conv("
                        f"name='{conv2}', s_filer={s}, n_filer={c_mid}, "
                        f"offset='(0.18,0,0)', to='({conv1}-east)', "
                        f"width={widths_bneck[1]:.1f}, height={h}, depth={d}"
                        "),"
                    )
                    lines.append(f"    to_connection('{conv1}', '{conv2}'),")

                    lines.append(
                        "    to_Conv("
                        f"name='{conv3}', s_filer={s}, n_filer={c_out}, "
                        f"offset='(0.18,0,0)', to='({conv2}-east)', "
                        f"width={widths_bneck[2]:.1f}, height={h}, depth={d}"
                        "),"
                    )
                    lines.append(f"    to_connection('{conv2}', '{conv3}'),")

                    lines.append(
                        "    to_Sum("
                        f"name='{sum_name}', offset='(0.55,0,0)', to='({conv3}-east)', "
                        "radius=1.9, opacity=0.6"
                        "),"
                    )
                    lines.append(f"    to_connection('{conv3}', '{sum_name}'),")

                    if is_projection:
                        lines.append(
                            "    to_Conv("
                            f"name='{projection_name}', s_filer={s}, n_filer={c_out}, "
                            f"offset='(0,1.35,0)', to='({prev}-east)', "
                            "width=0.9, height=8, depth=8, caption='Proj'"
                            "),"
                        )
                        lines.append(f"    to_connection('{prev}', '{projection_name}'),")
                        lines.append(f"    to_connection('{projection_name}', '{sum_name}'),")
                    else:
                        lines.append(f"    to_connection('{prev}', '{sum_name}'),")

                    prev = sum_name

        lines.extend([
            "    to_Pool(name='avg', offset='(1.2,0,0)', to='(%s-east)', width=1, height=6, depth=6, opacity=0.55, caption='Global AvgPool')," % prev,
            "    to_connection('%s', 'avg')," % prev,
            "    to_SoftMax(name='fc', s_filer=1000, offset='(1.2,0,0)', to='(avg-east)', width=1.8, height=8, depth=8, caption='FC-1000'),",
            "    to_connection('avg', 'fc'),",
        ])

        if title:
            title_block = (
                "\\node[above=1.45cm, align=center] at (current bounding box.north) "
                f"{{\\Large\\bfseries\\sffamily {self._escape_latex(title)}}};"
            )
            lines.append(f"    {repr(title_block)},")
        lines.extend([
            "    to_end(),",
            "]",
            "",
            "def main():",
            f"    to_generate(arch, {repr(output_tex_name)})",
            "",
            "if __name__ == '__main__':",
            "    main()",
            "",
        ])
        return "\n".join(lines)

    def _render_canonical_yolo_via_python(
        self,
        diagram: dict[str, Any],
        safe_name: str,
        work_dir: Path,
        tex_path: Path,
    ) -> None:
        variant = self._infer_yolo_variant(diagram)
        py_script_path = work_dir / f"{safe_name}_gen.py"
        script = self._build_yolo_python_script(
            output_tex_name=tex_path.name,
            plot_root=self.plotneuralnet_root,
            title=self._balance_parentheses(diagram.get("title", "")),
            variant=variant,
        )
        py_script_path.write_text(script, encoding="utf-8")

        result = subprocess.run(
            ["python3", py_script_path.name],
            cwd=work_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise PlotNeuralNetRenderError(
                "Ошибка генерации YOLO TeX через PlotNeuralNet python backend.\n"
                f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            )
        if not tex_path.exists():
            raise PlotNeuralNetRenderError(
                f"YOLO python backend не сгенерировал TeX: {tex_path}"
            )

    def _build_yolo_python_script(
        self,
        output_tex_name: str,
        plot_root: Path,
        title: str,
        variant: str = "v8",
    ) -> str:
        if variant == "v3":
            stage_caps = ["Darknet S3", "Darknet S4", "Darknet S5"]
            neck_caps = ["FPN P3", "FPN P4", "FPN P5"]
        elif variant == "v5":
            stage_caps = ["C3 S3", "C3 S4", "C3 S5"]
            neck_caps = ["PAN P3", "PAN P4", "PAN P5"]
        else:
            stage_caps = ["C2f S3", "C2f S4", "C2f S5"]
            neck_caps = ["Neck P3", "Neck P4", "Neck P5"]

        lines: list[str] = [
            "import sys",
            f"sys.path.insert(0, {repr(str(plot_root))})",
            "from pycore.tikzeng import *",
            "",
            "arch = [",
            f"    to_head({repr(str(plot_root))}),",
            "    to_cor(),",
            "    to_begin(),",
            "    '\\\\def\\\\ConvColor{rgb:yellow,7;red,2.2;white,5.5}',",
            "    to_Conv(name='input', s_filer=640, n_filer=3, offset='(0,0,0)', to='(0,0,0)', width=1.2, height=56, depth=56, caption='Input 640'),",
            "    to_Conv(name='stem', s_filer=320, n_filer=32, offset='(1.2,0,0)', to='(input-east)', width=1.4, height=50, depth=50, caption='Stem'),",
            "    to_connection('input', 'stem'),",
            "    to_Conv(name='stage2', s_filer=160, n_filer=64, offset='(1.0,0,0)', to='(stem-east)', width=1.8, height=42, depth=42, caption='Stage2'),",
            "    to_connection('stem', 'stage2'),",
            f"    to_Conv(name='stage3', s_filer=80, n_filer=128, offset='(1.0,0,0)', to='(stage2-east)', width=2.2, height=32, depth=32, caption='{stage_caps[0]}'),",
            "    to_connection('stage2', 'stage3'),",
            f"    to_Conv(name='stage4', s_filer=40, n_filer=256, offset='(1.1,0,0)', to='(stage3-east)', width=2.6, height=24, depth=24, caption='{stage_caps[1]}'),",
            "    to_connection('stage3', 'stage4'),",
            f"    to_Conv(name='stage5', s_filer=20, n_filer=512, offset='(1.2,0,0)', to='(stage4-east)', width=3.0, height=18, depth=18, caption='{stage_caps[2]}'),",
            "    to_connection('stage4', 'stage5'),",
            "    '\\\\def\\\\ConvColor{rgb:blue,4.5;green,1.8;white,6}',",
            f"    to_Conv(name='neck_p5', s_filer=20, n_filer=256, offset='(1.3,0,0)', to='(stage5-east)', width=2.1, height=18, depth=18, caption='{neck_caps[2]}'),",
            "    to_connection('stage5', 'neck_p5'),",
            f"    to_Conv(name='neck_p4', s_filer=40, n_filer=256, offset='(1.3,-3.0,0)', to='(neck_p5-east)', width=2.1, height=24, depth=24, caption='{neck_caps[1]}'),",
            "    to_connection('neck_p5', 'neck_p4'),",
            f"    to_Conv(name='neck_p3', s_filer=80, n_filer=128, offset='(1.3,-3.4,0)', to='(neck_p4-east)', width=2.0, height=30, depth=30, caption='{neck_caps[0]}'),",
            "    to_connection('neck_p4', 'neck_p3'),",
            "    '\\\\def\\\\ConvColor{rgb:yellow,5;red,6.2;white,4.5}',",
            "    to_Conv(name='head_s', s_filer=80, n_filer=255, offset='(2.2,0,0)', to='(neck_p3-east)', width=1.7, height=22, depth=22, caption='Head S'),",
            "    to_connection('neck_p3', 'head_s'),",
            "    to_Conv(name='head_m', s_filer=40, n_filer=255, offset='(2.2,0,0)', to='(neck_p4-east)', width=1.7, height=18, depth=18, caption='Head M'),",
            "    to_connection('neck_p4', 'head_m'),",
            "    to_Conv(name='head_l', s_filer=20, n_filer=255, offset='(2.2,0,0)', to='(neck_p5-east)', width=1.7, height=14, depth=14, caption='Head L'),",
            "    to_connection('neck_p5', 'head_l'),",
            "    '\\\\def\\\\ConvColor{rgb:magenta,3.8;blue,2.2;white,5.8}',",
            "    to_Conv(name='out_s', s_filer=1, n_filer=85, offset='(3.2,0,0)', to='(head_s-east)', width=1.2, height=11, depth=11, caption='Out S'),",
            "    to_connection('head_s', 'out_s'),",
            "    to_Conv(name='out_m', s_filer=1, n_filer=85, offset='(3.2,0,0)', to='(head_m-east)', width=1.2, height=10, depth=10, caption='Out M'),",
            "    to_connection('head_m', 'out_m'),",
            "    to_Conv(name='out_l', s_filer=1, n_filer=85, offset='(3.2,0,0)', to='(head_l-east)', width=1.2, height=9, depth=9, caption='Out L'),",
            "    to_connection('head_l', 'out_l'),",
        ]

        if title:
            title_block = (
                "\\node[above=1.45cm, align=center] at (current bounding box.north) "
                f"{{\\Large\\bfseries\\sffamily {self._escape_latex(title)}}};"
            )
            lines.append(f"    {repr(title_block)},")

        lines.extend([
            "    to_end(),",
            "]",
            "",
            "def main():",
            f"    to_generate(arch, {repr(output_tex_name)})",
            "",
            "if __name__ == '__main__':",
            "    main()",
            "",
        ])
        return "\n".join(lines)

    def _infer_unet_stage_count(self, diagram: dict[str, Any]) -> int:
        nodes = diagram.get("nodes", [])
        max_idx = 0
        enc_seen = 0
        for node in nodes:
            node_id = str(node.get("id", ""))
            label = str(node.get("label", ""))
            text = f"{node_id} {label}".lower()
            if "enc" in text:
                enc_seen += 1
            m = re.search(r"enc[\s_-]?(\d+)", text)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
        if max_idx > 0:
            return max(2, min(6, max_idx))
        if enc_seen > 0:
            return max(2, min(6, enc_seen))
        return 4

    def _build_unet_python_script(
        self,
        output_tex_name: str,
        plot_root: Path,
        stage_count: int,
    ) -> str:
        # Stage templates are aligned with PlotNeuralNet examples and scaled
        # to keep article-like proportions for 2..6 levels.
        channels = [64, 128, 256, 512, 768, 1024][:stage_count]
        spatial = [40, 32, 25, 16, 12, 9][:stage_count]
        widths = [2.5, 3.5, 4.5, 6.0, 6.8, 7.6][:stage_count]
        bottleneck_size = max(8, int(spatial[-1] * 0.5))
        bottleneck_width = min(9.0, widths[-1] + 2.0)
        bottleneck_filters = channels[-1] * 2

        enc_lines: list[str] = []
        dec_lines: list[str] = []
        skip_lines: list[str] = []

        enc_lines.append(
            "    to_ConvConvRelu(name='ccr_b1', s_filer=500, n_filer=(%d,%d), offset='(0,0,0)', to='(0,0,0)', width=(%.1f,%.1f), height=%d, depth=%d),"
            % (channels[0], channels[0], widths[0], widths[0], spatial[0], spatial[0])
        )
        enc_lines.append(
            "    to_Pool(name='pool_b1', offset='(0,0,0)', to='(ccr_b1-east)', width=1, height=%d, depth=%d, opacity=0.6),"
            % (max(6, spatial[0] - int(spatial[0] * 0.22)), max(6, spatial[0] - int(spatial[0] * 0.22)))
        )

        for idx in range(2, stage_count + 1):
            c = channels[idx - 1]
            s = spatial[idx - 1]
            w = widths[idx - 1]
            enc_lines.append(
                "    *block_2ConvPool(name='b%d', botton='pool_b%d', top='pool_b%d', s_filer=%d, n_filer=%d, offset='(1,0,0)', size=(%d,%d,%.1f), opacity=0.6),"
                % (idx, idx - 1, idx, max(64, s * 8), c, s, s, w)
            )

        bottom_id = f"pool_b{stage_count}"
        enc_last = f"ccr_b{stage_count}"
        enc_lines.append(
            "    to_ConvConvRelu(name='ccr_bn', s_filer=%d, n_filer=(%d,%d), offset='(2,0,0)', to='(%s-east)', width=(%.1f,%.1f), height=%d, depth=%d, caption='Bottleneck Conv'),"
            % (max(32, bottleneck_size * 4), bottleneck_filters, bottleneck_filters, bottom_id, bottleneck_width, bottleneck_width, bottleneck_size, bottleneck_size)
        )
        enc_lines.append("    to_connection('%s', 'ccr_bn')," % bottom_id)

        # Decoder path mirrors encoder with stage-aware unconv blocks.
        prev_top = "ccr_bn"
        un_idx = stage_count + 1
        for stage in range(stage_count, 0, -1):
            c = channels[stage - 1]
            s = spatial[stage - 1]
            w = widths[stage - 1]
            name = f"b{un_idx}"
            top = f"end_{name}"
            dec_lines.append(
                "    *block_Unconv(name='%s', botton='%s', top='%s', s_filer=%d, n_filer=%d, offset='(2.1,0,0)', size=(%d,%d,%.1f), opacity=0.6),"
                % (name, prev_top, top, max(64, s * 8), c, s, s, max(2.5, w - 0.3))
            )
            skip_lines.append(
                "    to_skip(of='ccr_b%d', to='ccr_res_%s', pos=1.25),"
                % (stage, name)
            )
            prev_top = top
            un_idx += 1

        softmax_h = spatial[0]
        dec_lines.append(
            "    to_ConvSoftMax(name='soft1', s_filer=%d, offset='(0.75,0,0)', to='(%s-east)', width=1, height=%d, depth=%d, caption='SOFTMAX'),"
            % (max(64, softmax_h * 8), prev_top, softmax_h, softmax_h)
        )
        dec_lines.append("    to_connection('%s', 'soft1')," % prev_top)

        body = "\n".join(enc_lines + dec_lines + skip_lines)
        return (
            "import sys\n"
            "from pathlib import Path\n"
            f"sys.path.insert(0, {repr(str(plot_root))})\n"
            "from pycore.tikzeng import *\n"
            "from pycore.blocks import *\n\n"
            "arch = [\n"
            f"    to_head({repr(str(plot_root))}),\n"
            "    to_cor(),\n"
            "    to_begin(),\n"
            f"{body}\n"
            "    to_end(),\n"
            "]\n\n"
            "def main():\n"
            f"    to_generate(arch, {repr(output_tex_name)})\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )

    def build_tex(self, diagram: dict[str, Any]) -> str:
        title = self._balance_parentheses(diagram.get("title", ""))
        raw_nodes = diagram.get("nodes", [])
        raw_edges = diagram.get("edges", [])

        mapped_nodes = self._prepare_nodes(raw_nodes)
        edges = self._normalize_edges(raw_edges, {node["id"] for node in mapped_nodes})
        self._layout = diagram.get("layout", "linear")
        ordered_nodes = self._compute_layout(mapped_nodes, edges, self._layout)

        node_blocks: list[str] = []
        edge_blocks: list[str] = []

        for i, node in enumerate(ordered_nodes):
            node_blocks.append(self._build_node_tex(node, i))

        self._node_position_map = {
            node["id"]: (float(node.get("x", i * 3.0)), float(node.get("y", 0.0)))
            for i, node in enumerate(ordered_nodes)
        }

        valid_ids = {n["id"] for n in ordered_nodes}
        for edge in edges:
            source = self._sanitize_id(str(edge.get("source", "")))
            target = self._sanitize_id(str(edge.get("target", "")))
            if source in valid_ids and target in valid_ids:
                edge_blocks.append(self._build_edge_tex(source, target))

        extra_blocks = self._build_extra_tex()

        tex = f"""
\\documentclass[border=15pt, multi, tikz]{{standalone}}
\\usepackage[T2A]{{fontenc}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[russian,english]{{babel}}
\\usepackage{{import}}
\\subimport{{{self.layers_dir.as_posix()}/}}{{init}}
\\usetikzlibrary{{positioning}}
\\usetikzlibrary{{3d}}
\\usetikzlibrary{{calc}}

\\begin{{document}}
\\begin{{tikzpicture}}
\\tikzstyle{{connection}}=[ultra thick,every node/.style={{sloped,allow upside down}},draw={{rgb:blue,4;red,1;green,1;black,3}},opacity=0.75]
\\tikzstyle{{skip}}=[ultra thick,every node/.style={{sloped,allow upside down}},draw=black!45,densely dashed,opacity=0.58]

{chr(10).join(node_blocks)}

{chr(10).join(edge_blocks)}
{extra_blocks}
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

    def _prepare_nodes(self, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []
        seen_ids: dict[str, str] = {}

        for index, node in enumerate(nodes):
            mapped = self._map_node(node, index)
            node_id = mapped["id"]
            original_id = str(node.get("id", f"layer_{index}"))

            if node_id in seen_ids and seen_ids[node_id] != original_id:
                raise PlotNeuralNetRenderError(
                    "Коллизия id после sanitize: "
                    f"'{seen_ids[node_id]}' и '{original_id}' -> '{node_id}'"
                )

            seen_ids[node_id] = original_id
            prepared.append(mapped)

        self._node_kind_map = {node["id"]: str(node.get("kind", "block")) for node in prepared}
        self._node_macro_map = {node["id"]: str(node.get("macro", "Box")) for node in prepared}
        return prepared

    def _normalize_edges(
        self,
        edges: list[dict[str, Any]],
        valid_ids: set[str],
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        for edge in edges:
            source = self._sanitize_id(str(edge.get("source", "")))
            target = self._sanitize_id(str(edge.get("target", "")))

            if not source or not target or source == target:
                continue

            if source not in valid_ids or target not in valid_ids:
                self._warnings.append(
                    f"Пропущено ребро с неизвестной вершиной: {source} -> {target}"
                )
                continue

            key = (source, target)
            if key in seen:
                continue

            seen.add(key)
            normalized.append({"source": source, "target": target})

        return normalized

    # ─────────────────────── node mapping ───────────────────────

    def _map_node(self, node: dict[str, Any], index: int) -> dict[str, Any]:
        node_id = self._sanitize_id(str(node.get("id", f"layer_{index}")))

        raw_label = node.get("label", f"Layer {index}")
        label = self._normalize_label(raw_label)

        kind = str(node.get("kind", "")).strip().lower()

        if not kind:
            kind = self._infer_kind_from_label(label)

        label = self._fit_label(label, kind=kind)

        block: dict[str, Any] = {
            "id": node_id,
            "label": label,
            "kind": kind,
        }
        line_count = max(1, len([line for line in str(label).splitlines() if line.strip()]))
        longest_line = max((len(line.strip()) for line in str(label).splitlines() if line.strip()), default=len(str(label)))

        # ── Размеры блоков уменьшены для компактности ──
        if kind == "input":
            base_width = 1.8 if line_count >= 2 or longest_line > 10 else 1.5
            block.update(macro="Box", params={"width": base_width, "height": 20, "depth": 20, "fill": "green!45", "opacity": 0.85})
        elif kind in ["conv", "cnn"]:
            block.update(macro="RightBandedBox", params={"width": "{1.6}", "height": 17, "depth": 17, "fill": "yellow!65", "bandfill": "orange!70", "opacity": 0.92})
        elif kind in ["pool", "maxpool", "avgpool"]:
            block.update(macro="Box", params={"width": 1.2, "height": 14, "depth": 14, "fill": "red!45", "opacity": 0.45})
        elif kind in ["fc", "dense", "linear"]:
            base_width = 1.9 if line_count >= 2 or longest_line > 11 else 1.5
            block.update(macro="Box", params={"width": base_width, "height": 8, "depth": 8, "fill": "violet!55", "opacity": 0.82})
        elif kind in ["sum", "add", "concat", "mul", "dot"]:
            logo_map = {"sum": "$+$", "add": "$+$", "concat": "©", "mul": "$\\times$", "dot": "$\\circ$"}
            block.update(macro="Ball", params={"radius": 1.5, "fill": "green", "logo": logo_map.get(kind, "$+$")})
        elif kind == "output":
            base_width = 2.0 if line_count >= 2 or longest_line > 11 else 1.5
            block.update(macro="Box", params={"width": base_width, "height": 10, "depth": 10, "fill": "magenta!55", "opacity": 0.88})
        else:
            block.update(macro="Box", params={"width": 2.4, "height": 16, "depth": 16, "fill": "black!45", "opacity": 0.45})

        if "width" in block["params"]:
            block["params"]["width"] = self._compute_block_width(
                label=label,
                kind=kind,
                base_width=block["params"].get("width", 2.0),
            )

        # Разрешаем переопределять ключевые параметры из JSON
        for key in ("width", "height", "depth", "fill", "opacity", "bandfill", "xlabel", "zlabel"):
            if key in node:
                block["params"][key] = node[key]

        # Позволяем явно задавать macro/params для paper-like сценариев.
        custom_macro = node.get("macro")
        if isinstance(custom_macro, str) and custom_macro.strip():
            block["macro"] = custom_macro.strip()

        custom_params = node.get("params")
        if isinstance(custom_params, dict):
            block["params"].update(custom_params)

        return block

    # ─────────────────────── layout ───────────────────────

    def _compute_layout(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        layout: str = "linear",
    ) -> list[dict[str, Any]]:
        if layout == "u_shape":
            return self._compute_u_shape_layout(nodes, edges)
        return self._compute_linear_or_special_layout(nodes, edges)

    def _compute_linear_or_special_layout(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        branched = self._try_compute_fanout_layout(nodes, edges)
        if branched is not None:
            return branched

        gan_like = self._try_compute_dual_input_merge_layout(nodes, edges)
        if gan_like is not None:
            return gan_like

        return self._compute_linear_layout(nodes, edges)

    def _compute_linear_layout(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not nodes:
            return nodes

        out_edges: dict[str, list[str]] = {n["id"]: [] for n in nodes}
        in_edges: dict[str, list[str]] = {n["id"]: [] for n in nodes}
        for e in edges:
            u = str(e.get("source", ""))
            v = str(e.get("target", ""))
            if u in out_edges and v in out_edges:
                out_edges[u].append(v)
                in_edges[v].append(u)

        self._incoming_map = in_edges
        depth = self._compute_depth_map(nodes, out_edges, in_edges)
        self._depth_map = depth
        self._skip_edges, self._merge_edges = self._classify_edges(
            nodes=nodes,
            edges=edges,
            out_edges=out_edges,
            in_edges=in_edges,
            depth=depth,
        )

        is_simple_chain = (
            all(len(outs) <= 1 for outs in out_edges.values())
            and all(len(ins) <= 1 for ins in in_edges.values())
        )
        if is_simple_chain and len(nodes) >= 6:
            ordered_ids = self._topological_order(nodes, out_edges, in_edges)
            order_idx = {node_id: idx for idx, node_id in enumerate(ordered_ids)}

            # Адаптивный шаг: длинные подписи/широкие блоки требуют больше интервала.
            max_block_width = 2.0
            max_line_count = 1
            for node in nodes:
                params = node.get("params", {}) or {}
                raw_w = params.get("width", 2.0)
                try:
                    width_val = float(str(raw_w).strip("{}"))
                except (TypeError, ValueError):
                    width_val = 2.0
                max_block_width = max(max_block_width, width_val)
                line_count = max(
                    1,
                    len([ln for ln in str(node.get("label", "")).splitlines() if ln.strip()]),
                )
                max_line_count = max(max_line_count, line_count)

            x_step = max(1.95, min(3.40, 1.10 + max_block_width * 0.62 + (max_line_count - 1) * 0.12))
            x0 = 0.0
            for node in nodes:
                node["x"] = x0 + order_idx.get(node["id"], 0) * x_step
                node["y"] = 0.0
            return nodes
        return self._compute_graphviz_layout(nodes, edges, ranksep="0.58", nodesep="0.42")

    def _compute_u_shape_layout(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        positioned = self._compute_linear_layout(nodes, edges)
        specialized = self._try_compute_encoder_decoder_layout(positioned, edges)
        if specialized is not None:
            return specialized
        return self._apply_u_shape_transform(positioned)

    def _compute_graphviz_layout(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        ranksep: str,
        nodesep: str,
    ) -> list[dict[str, Any]]:
        dot = graphviz.Digraph(engine="dot")
        dot.attr(rankdir='LR', ranksep=ranksep, nodesep=nodesep)

        for n in nodes:
            dot.node(n["id"], shape="box", width="1.0", height="1.0")

        valid_ids = {n["id"] for n in nodes}
        for e in edges:
            s_id = str(e.get("source", ""))
            t_id = str(e.get("target", ""))
            if s_id in valid_ids and t_id in valid_ids:
                if (s_id, t_id) in self._skip_edges:
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
                    pos_map[name] = (px / 25.0, py / 16.0)
                except ValueError:
                    pass

        for n in nodes:
            x, y = pos_map.get(n["id"], (0.0, 0.0))
            n["x"] = x
            n["y"] = y

        return nodes

    def _compute_depth_map(
        self,
        nodes: list[dict[str, Any]],
        out_edges: dict[str, list[str]],
        in_edges: dict[str, list[str]],
    ) -> dict[str, int]:
        node_ids = [node["id"] for node in nodes]
        indegree = {node_id: len(in_edges.get(node_id, [])) for node_id in node_ids}
        queue = [node_id for node_id in node_ids if indegree[node_id] == 0]
        depth = {node_id: 0 for node_id in node_ids}
        visited = 0

        while queue:
            current = queue.pop(0)
            visited += 1
            for nxt in out_edges.get(current, []):
                depth[nxt] = max(depth[nxt], depth[current] + 1)
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)

        if visited != len(node_ids):
            self._warnings.append("Обнаружен цикл или неоднозначная топология; применен fallback depth-map.")
            depth = {node_id: index for index, node_id in enumerate(node_ids)}

        return depth

    def _classify_edges(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        out_edges: dict[str, list[str]],
        in_edges: dict[str, list[str]],
        depth: dict[str, int],
    ) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
        skip_edges: set[tuple[str, str]] = set()
        merge_edges: set[tuple[str, str]] = set()
        merge_kinds = {"sum", "add", "concat", "mul", "dot"}

        def has_alt_path(start: str, target: str) -> bool:
            visited = set()
            stack = [v for v in out_edges.get(start, []) if v != target]
            while stack:
                curr = stack.pop()
                if curr == target:
                    return True
                if curr not in visited:
                    visited.add(curr)
                    stack.extend(out_edges.get(curr, []))
            return False

        for edge in edges:
            source = str(edge.get("source", ""))
            target = str(edge.get("target", ""))
            if source not in depth or target not in depth:
                continue

            target_kind = self._node_kind_map.get(target, "block")
            gap = depth[target] - depth[source]
            is_merge_target = target_kind in merge_kinds and len(in_edges.get(target, [])) > 1

            if is_merge_target:
                merge_edges.add((source, target))

            if not has_alt_path(source, target):
                continue

            if gap >= 2 or is_merge_target:
                skip_edges.add((source, target))

        return skip_edges, merge_edges

    def _apply_u_shape_transform(self, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not nodes:
            return nodes

        depths = [self._depth_map.get(node["id"], 0) for node in nodes]
        if not depths:
            return nodes

        max_depth = max(depths) or 1
        pivot_depth = max_depth / 2.0
        level_height = 1.8
        horizontal_step = 3.2
        level_counters: dict[int, int] = {}

        for node in nodes:
            node_depth = self._depth_map.get(node["id"], 0)
            bucket = int(node_depth)
            lane_index = level_counters.get(bucket, 0)
            level_counters[bucket] = lane_index + 1

            extra_x = lane_index * 0.6
            if node_depth <= pivot_depth:
                x = node_depth * horizontal_step + extra_x
                y = -node_depth * level_height
            else:
                mirrored = max_depth - node_depth
                x = node_depth * horizontal_step + extra_x
                y = -(mirrored * level_height)

            if abs(node_depth - pivot_depth) < 0.6:
                y -= 0.8

            node["x"] = x
            node["y"] = y

        return nodes

    def _topological_order(
        self,
        nodes: list[dict[str, Any]],
        out_edges: dict[str, list[str]],
        in_edges: dict[str, list[str]],
    ) -> list[str]:
        node_ids = [node["id"] for node in nodes]
        indegree = {node_id: len(in_edges.get(node_id, [])) for node_id in node_ids}
        queue = [node_id for node_id in node_ids if indegree[node_id] == 0]
        order: list[str] = []

        while queue:
            current = queue.pop(0)
            order.append(current)
            for nxt in out_edges.get(current, []):
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)

        if len(order) != len(node_ids):
            return node_ids
        return order

    def _try_compute_encoder_decoder_layout(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        if not self._skip_edges:
            return None

        out_edges: dict[str, list[str]] = {n["id"]: [] for n in nodes}
        in_edges: dict[str, list[str]] = {n["id"]: [] for n in nodes}
        for edge in edges:
            source = str(edge.get("source", ""))
            target = str(edge.get("target", ""))
            if source in out_edges and target in out_edges:
                out_edges[source].append(target)
                in_edges[target].append(source)

        merge_kinds = {"sum", "add", "concat", "mul", "dot"}
        merge_nodes = [
            node["id"]
            for node in nodes
            if self._node_kind_map.get(node["id"], "block") in merge_kinds and len(in_edges.get(node["id"], [])) > 1
        ]
        if not merge_nodes:
            return None

        order = self._topological_order(nodes, out_edges, in_edges)
        max_depth = max(self._depth_map.values()) if self._depth_map else 0
        left_step = 3.1
        row_step = 2.2
        positions: dict[str, tuple[float, float]] = {}

        encoder_sources = {source for source, _ in self._skip_edges}
        merge_targets = {target for _, target in self._skip_edges}
        encoder_levels = sorted({self._depth_map.get(node_id, 0) for node_id in encoder_sources})
        level_to_row = {level: idx + 1 for idx, level in enumerate(encoder_levels)}

        bottleneck_candidates = [
            node_id for node_id in order
            if node_id not in merge_targets and self._depth_map.get(node_id, 0) == max_depth
        ]
        bottleneck_id = bottleneck_candidates[0] if bottleneck_candidates else max(order, key=lambda node_id: self._depth_map.get(node_id, 0))
        bottleneck_x = (len(encoder_levels) + 1) * left_step

        for node_id in order:
            depth = self._depth_map.get(node_id, 0)
            kind = self._node_kind_map.get(node_id, "block")

            if node_id == bottleneck_id:
                positions[node_id] = (bottleneck_x, -((len(encoder_levels) + 1) * row_step))
                continue

            if node_id in encoder_sources:
                row = level_to_row.get(depth, max(1, depth))
                positions[node_id] = (row * left_step, -(row * row_step))
                continue

            if node_id in merge_targets:
                skip_inputs = [src for src, tgt in self._skip_edges if tgt == node_id]
                if skip_inputs:
                    aligned_depth = min(self._depth_map.get(src, 0) for src in skip_inputs)
                    aligned_row = level_to_row.get(aligned_depth, max(1, aligned_depth))
                else:
                    aligned_row = max(1, len(encoder_levels) - depth + 1)
                x = bottleneck_x + aligned_row * left_step
                y = -(aligned_row * row_step)
                positions[node_id] = (x, y)
                continue

            parents = in_edges.get(node_id, [])
            if parents:
                parent_positions = [positions[p] for p in parents if p in positions]
                if parent_positions:
                    avg_x = sum(pos[0] for pos in parent_positions) / len(parent_positions)
                    avg_y = sum(pos[1] for pos in parent_positions) / len(parent_positions)
                    offset_x = 0.9 if kind not in merge_kinds else 0.0
                    positions[node_id] = (avg_x + offset_x, avg_y)
                    continue

            positions[node_id] = ((depth + 1) * left_step, -(depth + 1) * row_step)

        if len(positions) != len(nodes):
            return None

        for node in nodes:
            node["x"], node["y"] = positions[node["id"]]
        return nodes

    def _try_compute_fanout_layout(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        out_edges: dict[str, list[str]] = {n["id"]: [] for n in nodes}
        in_edges: dict[str, list[str]] = {n["id"]: [] for n in nodes}
        for edge in edges:
            source = str(edge.get("source", ""))
            target = str(edge.get("target", ""))
            if source in out_edges and target in out_edges:
                out_edges[source].append(target)
                in_edges[target].append(source)

        merge_kinds = {"sum", "add", "concat", "mul", "dot"}
        branch_source = next((node_id for node_id, outs in out_edges.items() if len(outs) >= 3), None)
        if branch_source is None:
            return None

        merge_target = next(
            (
                node_id for node_id, ins in in_edges.items()
                if len(ins) >= 3 and self._node_kind_map.get(node_id, "block") in merge_kinds
            ),
            None,
        )
        if merge_target is None:
            return None

        branch_nodes = [node_id for node_id in out_edges[branch_source] if merge_target in out_edges.get(node_id, [])]
        if len(branch_nodes) < 3:
            return None

        node_ids = {node["id"] for node in nodes}
        remaining = [node_id for node_id in node_ids if node_id not in set(branch_nodes + [branch_source, merge_target])]
        positions: dict[str, tuple[float, float]] = {}
        x0 = 0.0
        x1 = 3.3
        x2 = 6.8
        y_step = 3.6
        branch_nodes = sorted(branch_nodes)
        center = (len(branch_nodes) - 1) / 2.0

        positions[branch_source] = (x0, 0.0)
        for idx, node_id in enumerate(branch_nodes):
            positions[node_id] = (x1, (center - idx) * y_step)
        positions[merge_target] = (x2, 0.0)

        order = self._topological_order(nodes, out_edges, in_edges)
        cursor_x = x2 + 2.6
        for node_id in order:
            if node_id in positions:
                continue
            parents = in_edges.get(node_id, [])
            if parents == [merge_target] or (merge_target in parents):
                positions[node_id] = (cursor_x, 0.0)
                cursor_x += 2.6

        for node_id in order:
            if node_id in positions:
                continue
            parents = [positions[p] for p in in_edges.get(node_id, []) if p in positions]
            if parents:
                avg_x = sum(pos[0] for pos in parents) / len(parents)
                avg_y = sum(pos[1] for pos in parents) / len(parents)
                positions[node_id] = (avg_x + 2.0, avg_y)
            else:
                positions[node_id] = (-2.4, 0.0)

        for node in nodes:
            node["x"], node["y"] = positions[node["id"]]
        return nodes

    def _try_compute_dual_input_merge_layout(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        out_edges: dict[str, list[str]] = {n["id"]: [] for n in nodes}
        in_edges: dict[str, list[str]] = {n["id"]: [] for n in nodes}
        for edge in edges:
            source = str(edge.get("source", ""))
            target = str(edge.get("target", ""))
            if source in out_edges and target in out_edges:
                out_edges[source].append(target)
                in_edges[target].append(source)

        candidate = next(
            (
                node_id for node_id, ins in in_edges.items()
                if len(ins) == 2 and self._node_kind_map.get(node_id, "block") == "block"
            ),
            None,
        )
        if candidate is None:
            return None

        parents = list(in_edges[candidate])
        if len(parents) != 2:
            return None

        input_parents = [node_id for node_id in parents if self._node_kind_map.get(node_id) == "input"]
        artifact_parents = [node_id for node_id in parents if self._node_kind_map.get(node_id) == "output"]
        if len(input_parents) != 1 or len(artifact_parents) != 1:
            return None

        real_id = input_parents[0]
        fake_id = artifact_parents[0]
        generator_parents = in_edges.get(fake_id, [])
        if len(generator_parents) != 1:
            return None

        generator_id = generator_parents[0]
        noise_parents = in_edges.get(generator_id, [])
        if len(noise_parents) != 1:
            return None

        noise_id = noise_parents[0]
        positions = {
            noise_id: (0.0, 1.8),
            generator_id: (3.0, 1.8),
            fake_id: (6.0, 1.8),
            real_id: (6.0, -1.8),
            candidate: (9.4, 0.0),
        }

        out_after = out_edges.get(candidate, [])
        cursor_x = 12.0
        current_y = 0.0
        for node_id in out_after:
            positions[node_id] = (cursor_x, current_y)
            cursor_x += 2.8

        order = self._topological_order(nodes, out_edges, in_edges)
        for node_id in order:
            if node_id in positions:
                continue
            parents_pos = [positions[p] for p in in_edges.get(node_id, []) if p in positions]
            if parents_pos:
                avg_x = sum(pos[0] for pos in parents_pos) / len(parents_pos)
                avg_y = sum(pos[1] for pos in parents_pos) / len(parents_pos)
                positions[node_id] = (avg_x + 2.0, avg_y)
            else:
                positions[node_id] = (-2.0, 0.0)

        for node in nodes:
            node["x"], node["y"] = positions[node["id"]]
        return nodes

    # ─────────────────────── label processing ───────────────────────

    def _normalize_label(self, label: str | None) -> str:
        if label is None:
            return "Unnamed Block"

        text = str(label).strip()
        if not text:
            return "Unnamed Block"

        normalized = re.sub(r"\s+", " ", text).strip()
        alias_patterns: list[tuple[str, str]] = [
            (r"^(input|input layer|image input|token input|вход|входные данные)$", "Input"),
            (r"^(initial )?(convolution(al)?|conv)( layers?)?$", "Conv Stem"),
            (r"^(stem|patch embedding|embedding stem|projection stem)$", "Conv Stem"),
            (r"^(stacked )?(cnn|conv)( layers?| blocks?)$", "CNN Blocks"),
            (r"^(transformer( encoder)?|attention)( layers?| blocks?)$", "Transformer Blocks"),
            (r"^(encoder|decoder) blocks?$", lambda m: m.group(1).title()),
            (r"^encoder stage (\d+)$", r"Enc\1"),
            (r"^decoder stage (\d+)$", r"Dec\1"),
            (r"^enc\s*[_-]?(\d+)$", r"Enc\1"),
            (r"^dec\s*[_-]?(\d+)$", r"Dec\1"),
            (r"^up\s*[_-]?(\d+)$", r"Up\1"),
            (r"^(feature )?fusion( module| layer)?$", "Fusion"),
            (r"^(classification|classifier|projection) head$", "Cls Head"),
            (r"^(segmentation|detection|classification) output$", "Output"),
            (r"^(output|output layer|prediction|logits|softmax)$", "Output"),
            (r"^(concat|concatenate|cat)$", "Concat"),
            (r"^(sum|add)$", "Add"),
            (r"^(upconv|up conv|upsample)$", "UpConv"),
            (r"^latent z$", "Latent Z"),
            (r"^(bottleneck|latent space)$", "Bottleneck"),
            (r"^pool projection$", "Pool Proj"),
            (r"^generator blocks?$", "Generator"),
            (r"^discriminator blocks?$", "Discriminator"),
            (r"^real fake prediction$", "Real/Fake"),
        ]

        for pattern, replacement in alias_patterns:
            if re.match(pattern, normalized, flags=re.IGNORECASE):
                if callable(replacement):
                    return str(replacement(re.match(pattern, normalized, flags=re.IGNORECASE)))
                if "\\" in replacement:
                    return re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
                return replacement

        normalized = re.sub(r"\b(layer|layers|module|modules|stack)\b", "", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\s+", " ", normalized).strip(" -_,")
        return normalized or "Unnamed Block"

    def _balance_parentheses(self, text: str) -> str:
        opened = text.count('(')
        closed = text.count(')')
        if opened > closed:
            text += ')' * (opened - closed)
        return text

    def _shorten_label(self, label: str | None, max_words: int = 20, max_chars: int = 45) -> str:
        if not label:
            return "Unnamed Block"

        label = str(label).strip()
        if not label:
            return "Unnamed Block"

        protected = {
            "Input", "Conv Stem", "CNN Blocks",
            "Transformer Blocks", "Fusion",
            "Cls Head", "Output", "FC",
            "Encoder", "Decoder", "Bottleneck",
            "Concat", "Add", "UpConv"
        }

        if label in protected:
            return self._balance_parentheses(label)

        words = label.split()
        if len(words) > max_words:
            label = " ".join(words[:max_words])

        if len(label) <= max_chars:
            return self._balance_parentheses(label)

        truncated = label[:max_chars].rsplit(' ', 1)[0]
        return self._balance_parentheses(truncated) + "..."

    def _wrap_label_lines(
        self,
        label: str,
        kind: str,
        max_line_chars: int | None = None,
        max_lines: int | None = None,
    ) -> list[str]:
        text = str(label or "").strip()
        if not text:
            return ["Unnamed Block"]

        if max_line_chars is None:
            if kind in {"input", "output", "fc"}:
                max_line_chars = 16
            elif kind in {"conv", "block"}:
                max_line_chars = 16
            else:
                max_line_chars = 14

        if max_lines is None:
            max_lines = 3 if kind in {"conv", "block"} else 2

        words = text.split()
        if len(words) <= 1:
            return [text]

        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if len(candidate) <= max_line_chars:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)

        if len(lines) <= max_lines:
            return lines

        shortened = self._shorten_label(text, max_words=max_lines * 2, max_chars=max_line_chars * max_lines)
        words = shortened.split()
        lines = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if len(candidate) <= max_line_chars:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)

        if len(lines) > max_lines:
            head = lines[: max_lines - 1]
            tail = " ".join(lines[max_lines - 1 :]).strip()
            if len(tail) > max_line_chars:
                tail = self._shorten_label(tail, max_words=3, max_chars=max_line_chars)
            lines = head + [tail]

        return lines

    def _fit_label(self, label: str, kind: str) -> str:
        lines = self._wrap_label_lines(label, kind=kind)
        return "\n".join(lines)

    def _compute_block_width(self, label: str, kind: str, base_width: Any) -> float | str:
        try:
            width_value = float(str(base_width).strip("{}"))
        except (TypeError, ValueError):
            width_value = 2.0

        lines = [line.strip() for line in str(label).splitlines() if line.strip()]
        if not lines:
            lines = [str(label or "").strip()]
        chars = max((len(line) for line in lines), default=0)
        words = max((len(line.split()) for line in lines), default=0)
        line_count = len(lines)
        width_value += max(0.0, min(2.2, (chars - 10) * 0.065))
        width_value += max(0.0, min(1.0, (words - 2) * 0.20))
        width_value += max(0.0, min(0.7, (line_count - 1) * 0.16))

        if kind in {"input", "output", "fc"}:
            width_value = min(width_value, 4.0)
        elif kind in {"conv", "block"}:
            width_value = min(width_value, 5.0)
        else:
            width_value = min(width_value, 3.8)

        if line_count >= 2 and kind in {"input", "output", "fc"}:
            width_value = max(width_value, 3.0)
        if line_count >= 3:
            if kind in {"input", "output", "fc"}:
                width_value = max(width_value, 3.4)
            elif kind in {"conv", "block"}:
                width_value = max(width_value, 3.9)

        if isinstance(base_width, str) and base_width.startswith("{"):
            return "{" + f"{width_value:.2f}" + "}"
        return round(width_value, 2)

    def _infer_kind_from_label(self, label: str) -> str:
        text = label.lower()

        if any(x in text for x in ("input", "вход", "image", "img", "tokens")):
            return "input"

        if any(x in text for x in ("pool", "avgpool", "maxpool", "global pool")):
            return "pool"

        if any(x in text for x in (
            "classifier", "classification head", "cls head", "projection head",
            "fully connected", "dense", "mlp head", "linear head"
        )):
            return "fc"

        if any(x in text for x in ("fc", "linear", "dense", "mlp", "projection")):
            return "fc"

        if any(x in text for x in (
            "output", "softmax", "classifier", "class", "выход",
            "segmentation", "mask", "detection", "boxes", "logits"
        )):
            return "output"

        if any(x in text for x in (
            "conv", "stem", "patch embedding", "embedding", "upconv", "upsample"
        )):
            return "conv"

        if any(x in text for x in (
            "encoder", "decoder", "transformer", "bottleneck", "block",
            "backbone", "neck", "head", "stage", "attention",
            "residual", "resblock", "unet", "fusion", "generator", "discriminator"
        )):
            return "block"

        return "block"

    # ─────────────────────── TeX generation ───────────────────────

    def _build_node_tex(self, node: dict[str, Any], index: int) -> str:
        name = node["id"]
        label = str(node["label"] or "")
        lines = [self._escape_latex(line) for line in label.splitlines() if line.strip()]
        kind = str(node.get("kind", "") or "")
        side_caption = kind == "input" and len(lines) > 1 and not node.get("meta_role")
        if side_caption:
            caption = ""
        elif len(lines) > 1:
            caption = r"\scriptsize\shortstack[c]{" + r" \\ ".join(lines) + "}"
        else:
            if lines and len(lines[0]) > 14:
                caption = r"\scriptsize " + lines[0]
            else:
                caption = lines[0] if lines else ""
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

        tex = f"""\
\\pic[shift={{({x:.2f},{y:.2f},0)}}] at (0,0,0)
    {{{macro}={{
        {params_str}
    }}}};"""

        if side_caption:
            side_text = r"\scriptsize\shortstack[r]{" + r" \\ ".join(lines) + "}"
            tex += f"\n\\node[anchor=north, align=center] at ($({name}-south)+(0.00,-0.62)$) {{{side_text}}};"

        return tex

    def _build_edge_tex(self, source: str, target: str) -> str:
        s_depth = self._depth_map.get(source, 0)
        t_depth = self._depth_map.get(target, 0)
        gap = max(1, t_depth - s_depth)

        if (source, target) not in self._skip_edges:
            return f"\\draw [connection] ({source}-east) -- node {{\\midarrow}} ({target}-west);"

        lift = 1.5 + gap * 0.6  # чем длиннее прыжок, тем выше дуга
        return (
            f"\\draw [skip] ({source}-north) -- "
            f"++(0,{lift:.1f},0) -| "
            f"node[pos=0.25] {{\\midarrow}} "
            f"({target}-north);"
        )

    def _anchor(self, node_id: str, side: str) -> str:
        if self._node_macro_map.get(node_id) in {
            "artifact",
            "latent_source",
            "gan_block",
            "sample_box",
            "decision_box",
        }:
            return f"{node_id}.{side}"
        return f"{node_id}-{side}"

    def _build_extra_tex(self) -> str:
        return ""

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

    def _generate_preview(self, pdf_path: Path, preview_path: Path) -> Path | None:
        if shutil.which("pdftocairo") is None:
            self._warnings.append("pdftocairo не найден; PNG preview не сгенерирован.")
            return None

        cmd = ["pdftocairo", "-png", "-singlefile", "-r", "220", str(pdf_path), str(preview_path.with_suffix(""))]
        result = subprocess.run(cmd, capture_output=True, text=False)
        if result.returncode != 0 or not preview_path.exists():
            self._warnings.append("Не удалось сгенерировать PNG preview из PDF.")
            return None
        return preview_path

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


def _sanitize_renderer_id(value: str) -> str:
    value = str(value or "").lower().strip()
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "layer"


def _build_graph_maps(diagram: dict[str, Any]) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, str]]:
    nodes = diagram.get("nodes", [])
    edges = diagram.get("edges", [])
    node_ids = [_sanitize_renderer_id(node.get("id", "")) for node in nodes]
    out_edges: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    in_edges: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    kind_map = {
        _sanitize_renderer_id(node.get("id", "")): str(node.get("kind", "")).strip().lower()
        for node in nodes
    }

    for edge in edges:
        source = _sanitize_renderer_id(edge.get("source", ""))
        target = _sanitize_renderer_id(edge.get("target", ""))
        if source in out_edges and target in out_edges and target not in out_edges[source]:
            out_edges[source].append(target)
            in_edges[target].append(source)

    return out_edges, in_edges, kind_map


class GenericLinearPlotNeuralNetRenderer(PlotNeuralNetRenderer):
    def _compute_layout(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        layout: str = "linear",
    ) -> list[dict[str, Any]]:
        if layout == "u_shape":
            return self._apply_u_shape_transform(self._compute_linear_layout(nodes, edges))
        return self._compute_linear_layout(nodes, edges)


class EncoderDecoderPlotNeuralNetRenderer(PlotNeuralNetRenderer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._label_map: dict[str, str] = {}
        self._unet_skip_lanes: dict[tuple[str, str], float] = {}

    def _map_node(self, node: dict[str, Any], index: int) -> dict[str, Any]:
        mapped = super()._map_node(node, index)
        kind = mapped.get("kind", "block")
        params = mapped.get("params", {})
        label = str(mapped.get("label", ""))

        if kind == "block":
            params.setdefault("height", 15)
            params.setdefault("depth", 15)
            params["height"] = min(params.get("height", 15), 15)
            params["depth"] = min(params.get("depth", 15), 15)
        elif kind == "conv":
            if label.startswith("Up"):
                mapped["macro"] = "Box"
                params.pop("bandfill", None)
                params["fill"] = "cyan!35"
                params["width"] = min(float(str(params.get("width", 1.6)).strip("{}")), 1.5)
                params["height"] = min(params.get("height", 17), 10)
                params["depth"] = min(params.get("depth", 17), 10)
            else:
                params["height"] = min(params.get("height", 17), 14)
                params["depth"] = min(params.get("depth", 17), 14)
        elif kind == "pool":
            params["height"] = min(params.get("height", 14), 11)
            params["depth"] = min(params.get("depth", 14), 11)
        elif kind == "output" and "segmentation" in label.lower():
            mapped["meta_role"] = "seg_output"
            mapped["display_label"] = label
            mapped["label"] = ""
            params["width"] = max(float(str(params.get("width", 1.8)).strip("{}")), 2.2)
            params["height"] = min(params.get("height", 10), 12)
            params["depth"] = min(params.get("depth", 10), 12)
        elif kind in {"sum", "add", "concat", "mul", "dot"}:
            params["radius"] = min(params.get("radius", 1.5), 1.1)

        return mapped

    def _compute_layout(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        layout: str = "u_shape",
    ) -> list[dict[str, Any]]:
        if not nodes:
            return nodes

        self._label_map = {node["id"]: str(node.get("label", "")) for node in nodes}
        self._unet_skip_lanes = {}

        out_edges: dict[str, list[str]] = {n["id"]: [] for n in nodes}
        in_edges: dict[str, list[str]] = {n["id"]: [] for n in nodes}
        for edge in edges:
            source = str(edge.get("source", ""))
            target = str(edge.get("target", ""))
            if source in out_edges and target in out_edges:
                out_edges[source].append(target)
                in_edges[target].append(source)

        depth = self._compute_depth_map(nodes, out_edges, in_edges)
        self._depth_map = depth
        self._skip_edges, self._merge_edges = self._classify_edges(
            nodes=nodes,
            edges=edges,
            out_edges=out_edges,
            in_edges=in_edges,
            depth=depth,
        )

        source_nodes = [node["id"] for node in nodes if not in_edges[node["id"]]]
        sink_nodes = [node["id"] for node in nodes if not out_edges[node["id"]]]
        if not source_nodes or not sink_nodes:
            return self._apply_u_shape_transform(self._compute_linear_layout(nodes, edges))

        source_id = source_nodes[0]
        sink_id = sink_nodes[0]
        main_path = self._primary_path_without_skips(source_id, sink_id, out_edges)
        if len(main_path) < 3:
            return self._apply_u_shape_transform(self._compute_linear_layout(nodes, edges))

        bottleneck_index = self._find_bottleneck_index(main_path, nodes)
        encoder_part = main_path[:bottleneck_index]
        bottleneck_id = main_path[bottleneck_index]
        decoder_part = main_path[bottleneck_index + 1:]

        skip_pairs = sorted(
            [
                (source, target)
                for source, target in self._skip_edges
                if source in encoder_part and target in decoder_part
            ],
            key=lambda pair: depth.get(pair[0], 0),
        )
        if not skip_pairs:
            return self._apply_u_shape_transform(self._compute_linear_layout(nodes, edges))
        # Keep only true U-Net skip links (encoder -> decoder merge).
        self._skip_edges = set(skip_pairs)

        encoder_rows = {source: idx + 1 for idx, (source, _) in enumerate(skip_pairs)}
        target_rows = {target: encoder_rows[source] for source, target in skip_pairs}
        max_stage = max(encoder_rows.values(), default=1)

        positions: dict[str, tuple[float, float]] = {}
        stage_x_step = 3.0
        op_x_step = 1.6
        merge_x_step = 1.4
        y_step = 3.1

        def is_upconv(node_id: str) -> bool:
            return self._node_kind_map.get(node_id, "") == "conv" and self._label_map.get(node_id, "").lower().startswith("up")

        input_x = 0.0
        encoder_x = [input_x + stage_x_step * idx for idx in range(1, max_stage + 1)]
        pool_x = [x + op_x_step for x in encoder_x]
        bottleneck_x = pool_x[-1] + stage_x_step
        decoder_anchor_x = bottleneck_x + stage_x_step + 1.5

        positions[source_id] = (input_x, -(1.0 * y_step))

        active_stage = 1
        last_encoder_x = input_x
        for node_id in encoder_part:
            if node_id == source_id:
                continue

            kind = self._node_kind_map.get(node_id, "")
            if node_id in encoder_rows:
                active_stage = encoder_rows[node_id]
                x = encoder_x[active_stage - 1]
                y = -(float(active_stage) * y_step)
                last_encoder_x = x
            elif kind == "pool":
                next_stage = min(active_stage + 1, max_stage)
                x = pool_x[active_stage - 1]
                y = -((active_stage + next_stage) / 2.0 * y_step)
            else:
                x = last_encoder_x + 0.8
                y = -(float(active_stage) * y_step)
            positions[node_id] = (x, y)

        bottleneck_level = max_stage + 1.0
        positions[bottleneck_id] = (bottleneck_x, -(bottleneck_level * y_step))

        decoder_stages: dict[int, dict[str, str]] = {}
        current_stage = max_stage
        for node_id in decoder_part:
            kind = self._node_kind_map.get(node_id, "block")
            if is_upconv(node_id):
                decoder_stages.setdefault(current_stage, {})["up"] = node_id
            elif node_id in target_rows:
                current_stage = target_rows[node_id]
                decoder_stages.setdefault(current_stage, {})["merge"] = node_id
            elif kind == "output":
                decoder_stages.setdefault(0, {})["output"] = node_id
            else:
                stage_bucket = current_stage if current_stage in target_rows.values() else max(current_stage, 1)
                slot = decoder_stages.setdefault(stage_bucket, {})
                if "dec" not in slot:
                    slot["dec"] = node_id
                else:
                    slot[f"extra_{len(slot)}"] = node_id

        for stage in range(max_stage, 0, -1):
            row_y = -(float(stage) * y_step)
            column = max_stage - stage
            up_x = decoder_anchor_x + column * (stage_x_step + op_x_step + merge_x_step)
            if stage == max_stage:
                up_x += 0.9
            merge_x = up_x + op_x_step
            dec_x = merge_x + merge_x_step
            slot = decoder_stages.get(stage, {})

            if "up" in slot:
                positions[slot["up"]] = (up_x, -((stage + 0.9) * y_step))
            if "merge" in slot:
                positions[slot["merge"]] = (merge_x, row_y)
            if "dec" in slot:
                positions[slot["dec"]] = (dec_x, row_y)

        # Skip lanes are drawn above each corresponding stage to mimic
        # PlotNeuralNet example-style upper corridors.
        for source, target in skip_pairs:
            stage = encoder_rows.get(source, 1)
            lane_y = -(float(stage) * y_step) + 1.45
            self._unet_skip_lanes[(source, target)] = lane_y

        output_slot = decoder_stages.get(0, {})
        last_stage_x = decoder_anchor_x + (max_stage - 1) * (stage_x_step + op_x_step + merge_x_step)
        if "output" in output_slot:
            positions[output_slot["output"]] = (last_stage_x + stage_x_step + 2.8, -(1.0 * y_step))

        for stage in range(max_stage, 0, -1):
            slot = decoder_stages.get(stage, {})
            if "dec" in slot and stage > 1:
                next_slot = decoder_stages.get(stage - 1, {})
                if "up" not in next_slot:
                    positions.setdefault(slot["dec"], positions.get(slot["dec"], (0.0, 0.0)))

        assigned = set(positions)
        order = self._topological_order(nodes, out_edges, in_edges)
        for node_id in order:
            if node_id in assigned:
                continue
            parents = [positions[parent] for parent in in_edges.get(node_id, []) if parent in positions]
            if parents:
                avg_x = sum(pos[0] for pos in parents) / len(parents)
                avg_y = sum(pos[1] for pos in parents) / len(parents)
                positions[node_id] = (avg_x + 1.0, avg_y)
            else:
                positions[node_id] = (bottleneck_x + stage_x_step, -(bottleneck_level * y_step))

        for node in nodes:
            node["x"], node["y"] = positions.get(node["id"], (0.0, 0.0))
        return nodes

    def _build_node_tex(self, node: dict[str, Any], index: int) -> str:
        tex = super()._build_node_tex(node, index)
        if node.get("meta_role") == "seg_output":
            name = node["id"]
            tex += (
                f"\n\\node[align=center] at ($({name}-south)+(0,-0.95)$) {{\\small Segmentation\\\\Map}};"
            )
        return tex

    def _build_edge_tex(self, source: str, target: str) -> str:
        source_label = self._label_map.get(source, "").lower()
        target_label = self._label_map.get(target, "").lower()
        if target_label.startswith("up") and self._node_kind_map.get(source, "") in {"sum", "add", "concat", "mul", "dot"}:
            return f"\\draw [connection] ({source}-east) -- node {{\\midarrow}} ({target}-west);"

        if (source, target) not in self._skip_edges:
            if target_label.startswith("up"):
                if self._node_kind_map.get(source, "") in {"sum", "add", "concat", "mul", "dot"}:
                    return f"\\draw [connection] ({source}-east) -- node {{\\midarrow}} ({target}-west);"
                if source_label.startswith("bottleneck") or source == "bottleneck":
                    return f"\\draw [connection] ({source}-east) -- node {{\\midarrow}} ({target}-west);"

                source_pos = self._node_position_map.get(source)
                target_pos = self._node_position_map.get(target)
                if source_pos is not None and target_pos is not None:
                    sx, sy = source_pos
                    tx, _ = target_pos
                    elbow_x = min(tx - 0.8, sx + 0.95)
                    return (
                        f"\\draw [connection] ({source}-east) -- ({elbow_x:.2f},{sy:.2f}) "
                        f"|- node[pos=0.55] {{\\midarrow}} ({target}-west);"
                    )
                return f"\\draw [connection] ({source}-east) -- node {{\\midarrow}} ({target}-west);"

            if source_label.startswith("up") and self._node_kind_map.get(target, "") in {"sum", "add", "concat", "mul", "dot"}:
                return f"\\draw [connection] ({source}-east) -- node {{\\midarrow}} ({target}-west);"

            if self._node_kind_map.get(source, "") in {"sum", "add", "concat", "mul", "dot"}:
                return f"\\draw [connection] ({source}-east) -- node {{\\midarrow}} ({target}-west);"

            return super()._build_edge_tex(source, target)

        source_pos = self._node_position_map.get(source)
        target_pos = self._node_position_map.get(target)
        if source_pos is None or target_pos is None:
            return super()._build_edge_tex(source, target)

        sx, sy = source_pos
        tx, ty = target_pos
        lane_y = self._unet_skip_lanes.get((source, target), max(sy, ty) + 1.45)
        start_x = sx + 0.8
        end_x = tx - 0.8
        return (
            f"\\draw [connection] ({source}-east) -- ({start_x:.2f},{sy:.2f}) "
            f"|- node[pos=0.25] {{\\midarrow}} ({end_x:.2f},{lane_y:.2f}) "
            f"-- ({end_x:.2f},{ty + 0.65:.2f}) -- ({target}-north);"
        )

    def _longest_path(
        self,
        source_id: str,
        sink_id: str,
        out_edges: dict[str, list[str]],
    ) -> list[str]:
        memo: dict[str, list[str]] = {}

        def dfs(node_id: str) -> list[str]:
            if node_id in memo:
                return memo[node_id]
            if node_id == sink_id:
                memo[node_id] = [node_id]
                return memo[node_id]

            best: list[str] = []
            for nxt in out_edges.get(node_id, []):
                candidate = dfs(nxt)
                if candidate and len(candidate) > len(best):
                    best = candidate

            memo[node_id] = [node_id] + best if best else [node_id]
            return memo[node_id]

        return dfs(source_id)

    def _primary_path_without_skips(
        self,
        source_id: str,
        sink_id: str,
        out_edges: dict[str, list[str]],
    ) -> list[str]:
        path = [source_id]
        current = source_id
        visited = {source_id}

        while current != sink_id:
            next_nodes = [
                nxt for nxt in out_edges.get(current, [])
                if (current, nxt) not in self._skip_edges
            ]
            if not next_nodes:
                next_nodes = list(out_edges.get(current, []))
            if not next_nodes:
                break

            next_nodes.sort(key=lambda nxt: (self._depth_map.get(nxt, 0), nxt))
            nxt = next_nodes[0]
            if nxt in visited:
                break
            path.append(nxt)
            visited.add(nxt)
            current = nxt

        if path[-1] != sink_id:
            longest = self._longest_path(source_id, sink_id, out_edges)
            if longest:
                return longest
        return path

    def _find_bottleneck_index(
        self,
        main_path: list[str],
        nodes: list[dict[str, Any]],
    ) -> int:
        labels = {node["id"]: str(node.get("label", "")).lower() for node in nodes}
        kinds = {node["id"]: str(node.get("kind", "")).lower() for node in nodes}

        for index, node_id in enumerate(main_path):
            label = labels.get(node_id, "")
            if "bottleneck" in label or "latent" in label:
                return index

        for index, node_id in enumerate(main_path):
            label = labels.get(node_id, "")
            kind = kinds.get(node_id, "")
            if kind == "conv" and "up" in label:
                return max(0, index - 1)

        return max(1, len(main_path) // 2)


class BranchingPlotNeuralNetRenderer(PlotNeuralNetRenderer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._branch_source_id: str | None = None
        self._branch_merge_id: str | None = None
        self._branch_fanout_x: float | None = None
        self._branch_fanin_x: float | None = None

    def _map_node(self, node: dict[str, Any], index: int) -> dict[str, Any]:
        mapped = super()._map_node(node, index)
        kind = mapped.get("kind", "block")
        params = mapped.get("params", {})
        label = str(mapped.get("label", ""))

        if kind in {"conv", "block"}:
            params["height"] = min(params.get("height", 16), 12)
            params["depth"] = min(params.get("depth", 16), 12)
            if "width" in params:
                try:
                    width_value = float(str(params["width"]).strip("{}"))
                    width_value = min(max(width_value, 1.8), 2.8)
                    if isinstance(params["width"], str):
                        params["width"] = "{" + f"{width_value:.2f}" + "}"
                    else:
                        params["width"] = round(width_value, 2)
                except (TypeError, ValueError):
                    pass
        elif kind == "input":
            params["height"] = min(params.get("height", 20), 16)
            params["depth"] = min(params.get("depth", 20), 16)
            params["width"] = max(float(params.get("width", 1.5)), 1.7)
        elif kind in {"sum", "add", "concat", "mul", "dot"}:
            params["radius"] = min(params.get("radius", 1.5), 1.0)

        if label in {"1x1 Conv", "3x3 Conv", "5x5 Conv"} and "width" in params:
            try:
                width_value = float(str(params["width"]).strip("{}"))
                width_value = max(width_value, 2.1)
                if isinstance(params["width"], str):
                    params["width"] = "{" + f"{width_value:.2f}" + "}"
                else:
                    params["width"] = round(width_value, 2)
            except (TypeError, ValueError):
                pass
        if label in {"1x1 Reduce", "Pool Proj"} and "width" in params:
            try:
                width_value = float(str(params["width"]).strip("{}"))
                width_value = max(width_value, 2.5)
                if isinstance(params["width"], str):
                    params["width"] = "{" + f"{width_value:.2f}" + "}"
                else:
                    params["width"] = round(width_value, 2)
            except (TypeError, ValueError):
                pass

        return mapped

    def _compute_layout(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        layout: str = "linear",
    ) -> list[dict[str, Any]]:
        if not nodes:
            return nodes

        out_edges: dict[str, list[str]] = {n["id"]: [] for n in nodes}
        in_edges: dict[str, list[str]] = {n["id"]: [] for n in nodes}
        for edge in edges:
            source = str(edge.get("source", ""))
            target = str(edge.get("target", ""))
            if source in out_edges and target in out_edges:
                out_edges[source].append(target)
                in_edges[target].append(source)

        depth = self._compute_depth_map(nodes, out_edges, in_edges)
        self._depth_map = depth
        self._skip_edges, self._merge_edges = self._classify_edges(
            nodes=nodes,
            edges=edges,
            out_edges=out_edges,
            in_edges=in_edges,
            depth=depth,
        )

        merge_kinds = {"sum", "add", "concat", "mul", "dot"}
        branch_source = next((node_id for node_id, outs in out_edges.items() if len(outs) >= 3), None)
        merge_target = next(
            (
                node_id for node_id, ins in in_edges.items()
                if len(ins) >= 3 and self._node_kind_map.get(node_id, "block") in merge_kinds
            ),
            None,
        )
        if branch_source is None or merge_target is None:
            return self._compute_linear_layout(nodes, edges)
        self._branch_source_id = branch_source
        self._branch_merge_id = merge_target
        self._branch_fanout_x = None
        self._branch_fanin_x = None

        branch_paths: list[list[str]] = []
        for child in out_edges.get(branch_source, []):
            path = [child]
            cursor = child
            visited = {branch_source, child}
            while cursor != merge_target:
                next_nodes = [nxt for nxt in out_edges.get(cursor, []) if nxt not in visited]
                if merge_target in out_edges.get(cursor, []):
                    path.append(merge_target)
                    break
                if len(next_nodes) != 1:
                    break
                cursor = next_nodes[0]
                visited.add(cursor)
                path.append(cursor)
            if path and path[-1] == merge_target:
                branch_paths.append(path)

        if len(branch_paths) < 3:
            return self._compute_linear_layout(nodes, edges)

        positions: dict[str, tuple[float, float]] = {branch_source: (0.0, 0.0)}
        y_step = 4.8
        x_step = 4.2
        center = (len(branch_paths) - 1) / 2.0

        for branch_index, path in enumerate(branch_paths):
            y = (center - branch_index) * y_step
            current_x = x_step
            for node_id in path[:-1]:
                positions[node_id] = (current_x, y)
                current_x += x_step

        merge_x = max(pos[0] for pos in positions.values()) + x_step
        positions[merge_target] = (merge_x, 0.0)
        first_branch_x = min(positions[path[0]][0] for path in branch_paths if path[:-1])
        self._branch_fanout_x = max(1.4, first_branch_x - 1.1)
        self._branch_fanin_x = merge_x - 1.1

        order = self._topological_order(nodes, out_edges, in_edges)
        cursor_x = merge_x + x_step
        for node_id in order:
            if node_id in positions:
                continue
            parents = in_edges.get(node_id, [])
            if parents and all(parent in positions for parent in parents):
                avg_y = sum(positions[parent][1] for parent in parents) / len(parents)
                if merge_target in parents:
                    positions[node_id] = (cursor_x, 0.0)
                    cursor_x += x_step
                else:
                    max_parent_x = max(positions[parent][0] for parent in parents)
                    positions[node_id] = (max_parent_x + x_step, avg_y)
            else:
                positions[node_id] = (0.0, 0.0)

        for node in nodes:
            node["x"], node["y"] = positions.get(node["id"], (0.0, 0.0))
        return nodes

    def _build_edge_tex(self, source: str, target: str) -> str:
        if self._branch_source_id and self._branch_merge_id:
            source_pos = self._node_position_map.get(source)
            target_pos = self._node_position_map.get(target)
            if source_pos is not None and target_pos is not None:
                sx, sy = source_pos
                tx, ty = target_pos
                if source == self._branch_source_id and abs(ty - sy) > 0.2:
                    bend = 0.55 if ty > sy else -0.55
                    return (
                        f"\\draw [connection] ({source}-east) "
                        f".. controls +(+0.95,0) and +(-0.95,{bend:.2f}) .. "
                        f"node[pos=0.50] {{\\midarrow}} ({target}-west);"
                    )
                if target == self._branch_merge_id and abs(ty - sy) > 0.2:
                    bend = -0.55 if sy > ty else 0.55
                    return (
                        f"\\draw [connection] ({source}-east) "
                        f".. controls +(+0.95,{bend:.2f}) and +(-0.95,0) .. "
                        f"node[pos=0.53] {{\\midarrow}} ({target}-west);"
                    )
        return super()._build_edge_tex(source, target)


class DualPathPlotNeuralNetRenderer(PlotNeuralNetRenderer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._gan_noise_id: str | None = None
        self._gan_generator_id: str | None = None
        self._gan_fake_id: str | None = None
        self._gan_real_id: str | None = None
        self._gan_discriminator_id: str | None = None
        self._gan_output_id: str | None = None

    def _map_node(self, node: dict[str, Any], index: int) -> dict[str, Any]:
        mapped = super()._map_node(node, index)
        label = mapped.get("label", "")
        kind = mapped.get("kind", "block")
        if label == "Latent Z":
            mapped["label"] = "Latent z"
            mapped["kind"] = "input"
            mapped["macro"] = "Box"
            mapped["params"] = {"width": 1.0, "height": 12, "depth": 12, "fill": "white"}
        elif label in {"Fake Image", "Real Image"}:
            mapped["label"] = ""
            mapped["macro"] = "Box"
            mapped["params"] = {"width": 1.7, "height": 10, "depth": 10, "fill": "white"}
            mapped["meta_role"] = "fake_sample" if label == "Fake Image" else "real_sample"
        elif label == "Real/Fake":
            mapped["label"] = ""
            mapped["kind"] = "output"
            mapped["macro"] = "Box"
            mapped["params"] = {"width": 1.0, "height": 11, "depth": 11, "fill": "white"}
            mapped["meta_role"] = "decision"
        elif kind == "block":
            lower_label = str(label).lower()
            if "generator" in lower_label:
                mapped["macro"] = "Box"
                mapped["params"] = {"width": 2.5, "height": 18, "depth": 18, "fill": "blue!20"}
            elif "discriminator" in lower_label:
                mapped["macro"] = "Box"
                mapped["params"] = {"width": 2.7, "height": 16, "depth": 16, "fill": "red!20"}

        return mapped

    def _build_node_tex(self, node: dict[str, Any], index: int) -> str:
        tex = super()._build_node_tex(node, index)
        if node.get("meta_role") == "decision":
            name = node["id"]
            tex += (
                f"\n\\node[text=green!45!black] at ($({name}-north)+(0,0.62)$) {{\\scriptsize Real}};"
                f"\n\\node[text=red!70!black] at ($({name}-south)+(0,-0.62)$) {{\\scriptsize Fake}};"
            )
        role = node.get("meta_role")
        if role == "real_sample":
            tex += (
                f"\n\\node at ($({node['id']}-north)+(0,1.28)$) {{\\scriptsize Sample}};"
                f"\n\\node[text=green!45!black] at ($({node['id']}-north)+(0,1.78)$) {{\\scriptsize real}};"
            )
        elif role == "fake_sample":
            tex += (
                f"\n\\node at ($({node['id']}-south)+(0,-0.70)$) {{\\scriptsize Sample}};"
                f"\n\\node[text=red!70!black] at ($({node['id']}-south)+(0,-1.20)$) {{\\scriptsize generated}};"
            )
        return tex

    def _compute_layout(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        layout: str = "linear",
    ) -> list[dict[str, Any]]:
        if not nodes:
            return nodes

        out_edges: dict[str, list[str]] = {n["id"]: [] for n in nodes}
        in_edges: dict[str, list[str]] = {n["id"]: [] for n in nodes}
        for edge in edges:
            source = str(edge.get("source", ""))
            target = str(edge.get("target", ""))
            if source in out_edges and target in out_edges:
                out_edges[source].append(target)
                in_edges[target].append(source)

        depth = self._compute_depth_map(nodes, out_edges, in_edges)
        self._depth_map = depth
        self._skip_edges, self._merge_edges = self._classify_edges(
            nodes=nodes,
            edges=edges,
            out_edges=out_edges,
            in_edges=in_edges,
            depth=depth,
        )

        discriminator_id = next(
            (
                node_id for node_id, ins in in_edges.items()
                if len(ins) == 2 and self._node_kind_map.get(node_id, "block") == "block"
            ),
            None,
        )
        if discriminator_id is None:
            return self._compute_linear_layout(nodes, edges)

        parents = list(in_edges[discriminator_id])
        artifact_parent = next((node_id for node_id in parents if self._node_kind_map.get(node_id) == "output"), None)
        real_parent = next((node_id for node_id in parents if self._node_kind_map.get(node_id) == "input"), None)
        if artifact_parent is None or real_parent is None:
            return self._compute_linear_layout(nodes, edges)

        generator_id = in_edges.get(artifact_parent, [None])[0]
        if generator_id is None:
            return self._compute_linear_layout(nodes, edges)
        noise_id = in_edges.get(generator_id, [None])[0]
        if noise_id is None:
            return self._compute_linear_layout(nodes, edges)

        self._gan_noise_id = noise_id
        self._gan_generator_id = generator_id
        self._gan_fake_id = artifact_parent
        self._gan_real_id = real_parent
        self._gan_discriminator_id = discriminator_id
        discriminator_outputs = out_edges.get(discriminator_id, [])
        self._gan_output_id = discriminator_outputs[0] if discriminator_outputs else None

        positions: dict[str, tuple[float, float]] = {
            noise_id: (0.0, -0.2),
            generator_id: (3.0, -0.2),
            artifact_parent: (6.0, -0.2),
            real_parent: (6.0, 2.9),
            discriminator_id: (9.7, 1.0),
        }

        order = self._topological_order(nodes, out_edges, in_edges)
        cursor_x = 12.7
        for node_id in order:
            if node_id in positions:
                continue
            parents_pos = [positions[parent] for parent in in_edges.get(node_id, []) if parent in positions]
            if parents_pos:
                avg_y = sum(pos[1] for pos in parents_pos) / len(parents_pos)
                max_x = max(pos[0] for pos in parents_pos)
                positions[node_id] = (max_x + 2.6, avg_y)
                if discriminator_id in in_edges.get(node_id, []):
                    positions[node_id] = (cursor_x, 1.0)
                    cursor_x += 2.6
            else:
                positions[node_id] = (0.0, 0.0)

        for node in nodes:
            node["x"], node["y"] = positions.get(node["id"], (0.0, 0.0))
        return nodes

    def _build_edge_tex(self, source: str, target: str) -> str:
        parents = self._incoming_map.get(target, [])
        source_east = self._anchor(source, "east")
        source_west = self._anchor(source, "west")
        target_west = self._anchor(target, "west")
        target_east = self._anchor(target, "east")

        if len(parents) == 2 and self._node_kind_map.get(target, "") == "block":
            source_kind = self._node_kind_map.get(source, "")
            if source_kind in {"input", "output"}:
                return (
                    f"\\draw [connection] ({source_east}) -- ++(0.8,0,0) |- "
                    f"node[pos=0.35] {{\\midarrow}} ({target_west});"
                )
        if self._node_macro_map.get(source) in {"latent_source", "gan_block", "sample_box", "decision_box"}:
            return f"\\draw [connection] ({source_east}) -- node {{\\midarrow}} ({target_west});"
        if self._node_macro_map.get(target) == "decision_box":
            return f"\\draw [connection] ({source_east}) -- ({target_west});"
        if self._node_macro_map.get(source) == "artifact" or self._node_macro_map.get(target) == "artifact":
            return f"\\draw [connection] ({source_east}) -- node {{\\midarrow}} ({target_west});"
        if self._node_macro_map.get(source) == "sample_box" or self._node_macro_map.get(target) == "sample_box":
            return f"\\draw [connection] ({source_east}) -- ({target_west});"
        return super()._build_edge_tex(source, target)

    def _build_extra_tex(self) -> str:
        discriminator_id = self._gan_discriminator_id
        output_id = self._gan_output_id
        generator_id = self._gan_generator_id

        if discriminator_id is None:
            discriminator_id = next(
                (
                    node_id for node_id, parents in self._incoming_map.items()
                    if len(parents) == 2 and self._node_kind_map.get(node_id) == "block"
                ),
                None,
            )
        if discriminator_id is None:
            return ""

        if output_id is None:
            candidate_outputs = [
                node_id for node_id, parents in self._incoming_map.items()
                if discriminator_id in parents
            ]
            output_id = candidate_outputs[0] if candidate_outputs else None
        if output_id is None:
            return ""

        if generator_id is None:
            fake_id = next(
                (
                    node_id for node_id, macro in self._node_macro_map.items()
                    if macro == "sample_box" and self._node_kind_map.get(node_id) == "sample"
                    and any(self._node_kind_map.get(parent) == "block" for parent in self._incoming_map.get(node_id, []))
                ),
                None,
            )
            if fake_id is not None:
                parents = self._incoming_map.get(fake_id, [])
                generator_id = parents[0] if parents else None
        if generator_id is None:
            return ""

        ox, oy = self._node_position_map.get(output_id, (12.5, 1.0))
        loss_x = ox + 2.25
        loss_y = oy
        return (
            f"\\pic[shift={{({loss_x:.2f},{loss_y:.2f},0)}}] at (0,0,0)\n"
            f"    {{Box={{name=ganloss, caption={{Loss}}, width=1.2, height=11, depth=11, fill=gray!15}}}};\n"
            f"\\draw [connection] ({self._anchor(output_id, 'east')}) -- (ganloss-west);"
        )


def pick_model_renderer_class(diagram: dict[str, Any]) -> type[PlotNeuralNetRenderer]:
    layout = str(diagram.get("layout", "linear")).strip().lower()
    if layout == "u_shape":
        return EncoderDecoderPlotNeuralNetRenderer

    out_edges, in_edges, kind_map = _build_graph_maps(diagram)
    merge_kinds = {"sum", "add", "concat", "mul", "dot"}

    if any(len(outs) >= 3 for outs in out_edges.values()) and any(
        len(ins) >= 3 and kind_map.get(node_id, "") in merge_kinds
        for node_id, ins in in_edges.items()
    ):
        return BranchingPlotNeuralNetRenderer

    if any(
        len(ins) == 2 and kind_map.get(node_id, "") == "block"
        for node_id, ins in in_edges.items()
    ):
        return DualPathPlotNeuralNetRenderer

    return GenericLinearPlotNeuralNetRenderer


def build_model_renderer(
    diagram: dict[str, Any],
    project_root: str | Path | None = None,
    plotneuralnet_root: str | Path | None = None,
    output_root: str | Path | None = None,
    latex_command: str = "pdflatex",
) -> PlotNeuralNetRenderer:
    renderer_cls = pick_model_renderer_class(diagram)
    return renderer_cls(
        project_root=project_root,
        plotneuralnet_root=plotneuralnet_root,
        output_root=output_root,
        latex_command=latex_command,
    )
