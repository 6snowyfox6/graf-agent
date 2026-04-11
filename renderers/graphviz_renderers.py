"""Graphviz-based renderers for logical architecture (general) and ML pipelines."""

from __future__ import annotations

from pathlib import Path

from graphviz import Digraph

from pipeline.scoring import get_node_level, get_general_node_sort_key, get_pipeline_lane


# ── general renderer ───────────────────────────────────────────────────────

def chunk_nodes(items: list, size: int) -> list[list]:
    if size <= 0:
        size = 3
    return [items[i:i + size] for i in range(0, len(items), size)]


def apply_default_style(node: dict) -> dict:
    node = dict(node)
    kind = str(node.get("kind", "block")).lower().strip()

    style_map = {
        "input": {
            "shape": "ellipse",
            "style": "filled",
            "fillcolor": "#EAF6EC",
            "color": "#5C8B66",
        },
        "conv": {
            "shape": "box",
            "style": "rounded,filled",
            "fillcolor": "#EAF2FB",
            "color": "#5A7BA3",
        },
        "block": {
            "shape": "box",
            "style": "rounded,filled",
            "fillcolor": "#EEF3F7",
            "color": "#667B8F",
        },
        "output": {
            "shape": "ellipse",
            "style": "filled",
            "fillcolor": "#FDF1E6",
            "color": "#A7774B",
        },
        "pool": {
            "shape": "ellipse",
            "style": "filled",
            "fillcolor": "#F7F0E6",
            "color": "#8D7A5B",
        },
    }

    base = style_map.get(kind, style_map["block"])
    for k, v in base.items():
        node.setdefault(k, v)

    return node


def normalize_edge_attrs(edge: dict, src_level: int, dst_level: int, layout_hint: str) -> dict:
    raw_style = edge.get("style", "solid")

    edge_kwargs = {
        "color": "#555555",
        "style": "solid",
        "penwidth": "1.1",
    }

    raw_color = edge.get("color")
    if isinstance(raw_color, str) and raw_color.strip():
        edge_kwargs["color"] = raw_color.strip()

    if isinstance(raw_style, str) and raw_style.strip():
        edge_kwargs["style"] = raw_style.strip()

    if isinstance(raw_style, dict):
        arrowhead = raw_style.get("arrowhead")
        if isinstance(arrowhead, str) and arrowhead.strip():
            if arrowhead.strip().lower() == "triangle":
                edge_kwargs["arrowhead"] = "normal"
            else:
                edge_kwargs["arrowhead"] = arrowhead.strip()

        line_style = raw_style.get("line_style")
        if isinstance(line_style, str) and line_style.strip():
            edge_kwargs["style"] = line_style.strip()

        color = raw_style.get("color")
        if isinstance(color, str) and color.strip():
            edge_kwargs["color"] = color.strip()

        penwidth = raw_style.get("penwidth")
        if penwidth is not None:
            edge_kwargs["penwidth"] = str(penwidth)

    label = edge.get("label", "")
    if not isinstance(label, str):
        label = ""
    label = label.strip()

    if label in {"", "->", "-->", "=>", "→"}:
        label = ""

    if len(label) > 22:
        label = ""

    if layout_hint == "general":
        label = ""

    if label:
        edge_kwargs["label"] = label

    if abs(dst_level - src_level) > 1:
        edge_kwargs["constraint"] = "false"
        edge_kwargs["color"] = "#777777"
        edge_kwargs["penwidth"] = "0.9"
        edge_kwargs["minlen"] = "2"

    return edge_kwargs


def render_general_diagram(
    diagram: dict,
    output_name: str = "final_diagram",
    output_dir: str | Path = "outputs",
) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dot = Digraph(comment=diagram.get("title", "Diagram"))
    dot.attr(
        rankdir="TB",
        splines="ortho",
        newrank="true",
        overlap="false",
        nodesep="0.24",
        ranksep="0.55",
        pad="0.20",
        bgcolor="white",
    )

    dot.attr(
        "node",
        fontname="Arial",
        fontsize="12",
        margin="0.14,0.08",
        penwidth="1.2",
    )
    dot.attr(
        "edge",
        fontname="Arial",
        fontsize="10",
        arrowsize="0.7",
    )

    styled_nodes = [apply_default_style(n) for n in diagram.get("nodes", [])]

    node_levels = {}
    level_groups: dict[int, list] = {}

    for node in styled_nodes:
        node_id = node["id"]
        label = node.get("label", node_id)
        kind = node.get("kind")
        level = get_node_level(node_id, label, kind)

        node_levels[node_id] = level
        level_groups.setdefault(level, []).append(node)

    for level in level_groups:
        level_groups[level] = sorted(level_groups[level], key=get_general_node_sort_key)

    max_per_row_by_level = {0: 2, 1: 2, 2: 3, 3: 3}
    sorted_levels = sorted(level_groups.keys())

    for level in sorted_levels:
        max_per_row = max_per_row_by_level.get(level, 3)
        rows = chunk_nodes(level_groups[level], max_per_row)

        for row_idx, row in enumerate(rows):
            with dot.subgraph(name=f"cluster_level_{level}_row_{row_idx}") as sub:
                sub.attr(rank="same")
                sub.attr(color="transparent")
                sub.attr(penwidth="0")

                previous_id = None
                for node in row:
                    sub.node(
                        node["id"],
                        node.get("label", node["id"]),
                        shape=node.get("shape", "box"),
                        style=node.get("style", "rounded,filled"),
                        fillcolor=node.get("fillcolor", "#d9e7f7"),
                        color=node.get("color", "#5a84c9"),
                    )

                    if previous_id is not None:
                        sub.edge(
                            previous_id,
                            node["id"],
                            style="invis",
                            weight="10",
                        )
                    previous_id = node["id"]

    for edge in diagram.get("edges", []):
        src = edge.get("source")
        dst = edge.get("target")

        if not src or not dst or src == dst:
            continue

        src_level = node_levels.get(src, 2)
        dst_level = node_levels.get(dst, 2)

        edge_kwargs = normalize_edge_attrs(
            edge=edge,
            src_level=src_level,
            dst_level=dst_level,
            layout_hint=diagram.get("layout_hint", "general"),
        )

        dot.edge(src, dst, **edge_kwargs)

    dot.filename = f"{output_name}.gv"
    dot.directory = str(output_dir)

    png_path = dot.render(format="png", cleanup=True)
    print(f"Схема сохранена: {Path(png_path).name}")

    svg_path = dot.render(format="svg", cleanup=True)
    print(f"Схема сохранена: {Path(svg_path).name}")

    pdf_path = dot.render(format="pdf", cleanup=True)
    print(f"Схема сохранена: {Path(pdf_path).name}")

    return png_path


# ── pipeline renderer ──────────────────────────────────────────────────────

def auto_style_pipeline_node(node: dict) -> dict:
    label = node.get("label", "").lower()
    styled = node.copy()

    if (
        "input" in label or "output" in label or "вход" in label
        or "выход" in label or "результат" in label
    ):
        styled.setdefault("shape", "ellipse")
        styled.setdefault("fillcolor", "#EAF6EC")
        styled.setdefault("color", "#5C8B66")
        styled.setdefault("style", "filled,bold")
    elif (
        "backbone" in label or "encoder" in label or "decoder" in label
        or "transformer" in label or "darknet" in label
        or "resnet" in label or "block" in label or "neck" in label or "модуль" in label
    ):
        styled.setdefault("shape", "box")
        styled.setdefault("fillcolor", "#EAF2FB")
        styled.setdefault("color", "#5A7BA3")
        styled.setdefault("style", "filled,rounded")
    elif (
        "feature" in label or "embedding" in label or "эмбед" in label
        or "map" in label or "representation" in label or "features" in label
        or "database" in label or "бд" in label or "vector" in label
    ):
        styled.setdefault("shape", "cylinder")
        styled.setdefault("fillcolor", "#FDF1E6")
        styled.setdefault("color", "#A7774B")
        styled.setdefault("style", "filled,bold")
    elif (
        "head" in label or "classifier" in label or "detector" in label
        or "segmentation" in label or "classification" in label
        or "классиф" in label or "детектор" in label
    ):
        styled.setdefault("shape", "box")
        styled.setdefault("fillcolor", "#EEF3F7")
        styled.setdefault("color", "#667B8F")
        styled.setdefault("style", "filled,rounded")
    elif (
        "ontology" in label or "онтолог" in label or "kg" in label or "graph" in label
    ):
        styled.setdefault("shape", "cylinder")
        styled.setdefault("fillcolor", "#FDF1E6")
        styled.setdefault("color", "#A7774B")
        styled.setdefault("style", "filled,bold")
    else:
        styled.setdefault("shape", "box")
        styled.setdefault("fillcolor", "#EAF2FB")
        styled.setdefault("color", "#5A7BA3")
        styled.setdefault("style", "filled,rounded")

    return styled


def render_pipeline_diagram(
    diagram: dict,
    output_name: str = "diagram",
    output_dir: str | Path = "outputs",
):
    dot = Digraph(comment=diagram.get("title", "Pipeline"))
    dot.attr(
        rankdir="TB",
        splines="spline",
        overlap="false",
        nodesep="0.35",
        ranksep="0.45",
        bgcolor="white",
    )

    dot.attr(
        "node",
        shape="box",
        style="filled,rounded",
        fillcolor="#d9e8fb",
        color="#4a6fa5",
        fontname="Arial",
        fontsize="13",
        margin="0.12,0.08",
    )

    dot.attr(
        "edge",
        color="#555555",
        fontname="Arial",
        fontsize="10",
        arrowsize="0.7",
    )

    styled_nodes = [auto_style_pipeline_node(node) for node in diagram.get("nodes", [])]

    lanes = diagram.get("lanes", [])
    lane_map: dict[str, str] = {}
    if lanes:
        for lane in lanes:
            lane_name = lane.get("name", "lane")
            for node_id in lane.get("nodes", []):
                lane_map[node_id] = lane_name
    else:
        for node in styled_nodes:
            lane_map[node["id"]] = get_pipeline_lane(node["id"], node["label"])

    lane_order = ["query_pipeline", "document_pipeline"]
    grouped = {name: [] for name in lane_order}
    for node in styled_nodes:
        lane = lane_map.get(node["id"], "document_pipeline")
        if lane not in grouped:
            grouped[lane] = []
        grouped[lane].append(node)

    for lane_name in lane_order:
        if not grouped.get(lane_name):
            continue
        with dot.subgraph() as s:
            s.attr(rank="same")
            for node in grouped[lane_name]:
                s.node(
                    node["id"],
                    node["label"],
                    shape=node.get("shape", "box"),
                    style=node.get("style", "filled,rounded"),
                    fillcolor=node.get("fillcolor", "#d9e8fb"),
                    color=node.get("color", "#4a6fa5"),
                )

    query_order = [
        "user_input", "query_vectorization", "document_retrieval",
        "relevance_ranking", "llm_context", "language_model", "final_response"
    ]
    document_order = [
        "document", "text_extraction", "text_preprocessing",
        "document_segmentation", "embedding_generation", "vector_database"
    ]

    def add_invisible_order_edges(node_order: list[str], existing_ids: set[str]):
        filtered = [n for n in node_order if n in existing_ids]
        for a, b in zip(filtered, filtered[1:]):
            dot.edge(a, b, style="invis", weight="10")

    existing_ids = {node["id"] for node in styled_nodes}
    add_invisible_order_edges(query_order, existing_ids)
    add_invisible_order_edges(document_order, existing_ids)

    for edge in diagram.get("edges", []):
        raw_label = edge.get("label", "")
        label = raw_label.strip() if isinstance(raw_label, str) else ""

        if label in {"→", "->", "-->", "=>", ""}:
            label = ""
        if len(label) > 18:
            label = ""

        edge_kwargs = {
            "color": edge.get("color", "#555555"),
            "style": edge.get("style", "solid"),
            "penwidth": "1.2",
        }

        source_lane = lane_map.get(edge["source"], "document_pipeline")
        target_lane = lane_map.get(edge["target"], "document_pipeline")
        cross_lane = source_lane != target_lane

        if cross_lane:
            edge_kwargs["constraint"] = "false"
            edge_kwargs["color"] = "#666666"
            edge_kwargs["penwidth"] = "1.0"

        if label:
            edge_kwargs["label"] = label

        dot.edge(edge["source"], edge["target"], **edge_kwargs)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dot.filename = f"{output_name}.gv"
    dot.directory = str(out_dir)

    png_path = dot.render(format="png", cleanup=True)
    svg_path = dot.render(format="svg", cleanup=True)
    pdf_path = dot.render(format="pdf", cleanup=True)

    print(f"Схема сохранена: {Path(png_path).name}")
    print(f"Схема сохранена: {Path(svg_path).name}")
    print(f"Схема сохранена: {Path(pdf_path).name}")
    return png_path
