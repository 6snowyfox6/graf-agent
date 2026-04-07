import os
import argparse
import base64
import mimetypes
import json
import re
import time
from pathlib import Path
import requests
from typing import Any

from graphviz import Digraph
from server_manager import ServerManager
from critic_influence import (
    CriticInfluenceAnalyzer,
    compute_change_metrics,
    compute_critic_listening_metrics,
)
from pipeline.render_router import render_diagram as route_render_diagram
from pipeline.json_ops import extract_json as pipeline_extract_json
from pipeline.diagram_cleaning import clean_diagram_labels as pipeline_clean_diagram_labels

# Локальные endpoints llama-cpp-python
# Qwen (генератор) на порту 8000, Gemma (критик/vision) на порту 8001
QWEN_URL = "http://127.0.0.1:8000/v1/chat/completions"
GEMMA_URL = "http://127.0.0.1:8001/v1/chat/completions"

GENERATOR_MODEL = "qwen2.5-7b-instruct"
VISION_MODEL = "gemma-3-4b-it"
CRITIC_MODEL = "gemma-3-4b-it"

MODEL_ENDPOINTS: dict[str, str] = {
    GENERATOR_MODEL: QWEN_URL,
    VISION_MODEL: GEMMA_URL,
    CRITIC_MODEL: GEMMA_URL,
}

def _get_endpoint(model: str) -> str:
    return MODEL_ENDPOINTS.get(model, QWEN_URL)


def save_json_artifact(filename: str, data: Any, base_dir: str | Path = ".") -> Path:
    out_dir = Path(base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / filename
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def load_user_prompt(default_prompt: str, prefer_test_prompt: bool = True) -> str:
    if not prefer_test_prompt:
        return default_prompt
    test_prompt_path = Path("_test_prompt.txt")
    if test_prompt_path.exists():
        text = test_prompt_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return default_prompt




FORBIDDEN_VISIBLE_TOKENS = {
    "input", "output", "block", "conv", "ui", "api", "db", "service"
}


def clean_visible_label(label: str) -> str:
    s = str(label or "").strip()

    # убрать служебные хвосты
    s = re.sub(
        r"\s*\((input|output|block|conv|pool|service|db|ui|api)\)\s*",
        "",
        s,
        flags=re.IGNORECASE,
    )

    # сначала длинные и специальные шаблоны
    replacements = [
        (r"\bweb\s*ui\b", "Веб-интерфейс"),
        (r"\bweb\b", "Веб-интерфейс"),
        (r"\bapi\s*v?\.?\s*(\d+)\b", r"Программный интерфейс \1"),
        (r"\bapi\b", "Программный интерфейс"),
        (r"\bui\b", "Интерфейс"),
        (r"\bdb\s+user\b", "База пользователей"),
        (r"\buser\s+db\b", "База пользователей"),
        (r"\bdb\s+material\b", "База материалов"),
        (r"\bmaterial\s+db\b", "База материалов"),
        (r"\bdb\s+result\b", "База результатов"),
        (r"\bresult\s+db\b", "База результатов"),
        (r"\bdb\b", "База данных"),
        (r"\bnotification\b", "Уведомления"),
        (r"\banalytics\b", "Аналитика"),
        (r"\bauth\b", "Аутентификация"),
        (r"\bdata\b", "Данные"),
        (r"\bservice\b", ""),
        (r"\bblock\b", ""),
        (r"\bconv\b", ""),
        (r"\binput\b", ""),
        (r"\boutput\b", ""),
        (r"\bpool\b", ""),
        (r"\bвнутр\.\b", "внутренний"),
    ]

    for pattern, repl in replacements:
        s = re.sub(pattern, repl, s, flags=re.IGNORECASE)

    # специальные нормализации
    s_low = s.lower().strip()

    if s_low in {"api", "программный интерфейс api"}:
        s = "Программный интерфейс"
    elif s_low in {"web", "веб", "внешний интерфейс"}:
        s = "Веб-интерфейс"
    elif s_low in {"api пользователя", "api админа", "api вход", "api управление"}:
        s = "Программный интерфейс"
    elif s_low in {"пользователи"}:
        s = "Пользователь"
    elif s_low in {"администраторы", "админ"}:
        s = "Администратор"

    # не допускать "Интерфейс Программный интерфейс"
    s = re.sub(
        r"(?i)\bинтерфейс\s+программный интерфейс\b",
        "Программный интерфейс",
        s,
    )

    # не допускать "Программный интерфейс пользователей"
    s = re.sub(
        r"(?i)\bпрограммный интерфейс\s+пользователей\b",
        "Программный интерфейс",
        s,
    )

    # косметика
    s = re.sub(r"\s+", " ", s).strip(" ,;:()[]-")
    s = s.replace("  ", " ")

    if not s:
        s = "Компонент"

    return s


def infer_general_kind_from_label(label: str, node_id: str = "") -> str:
    t = f"{node_id} {label}".lower()

    if any(x in t for x in [
        "пользователь", "админ", "администратор", "модератор",
        "клиент", "оператор", "внешние системы", "внешние сервисы",
        "интеграции"
    ]):
        return "input"

    if any(x in t for x in [
        "интерфейс", "веб", "api", "ui", "портал", "панель"
    ]):
        return "conv"

    if any(x in t for x in [
        "база", "хранилище", "лог", "телеметр", "архив"
    ]):
        return "output"

    return "block"


def dedupe_edges(edges: list[dict]) -> list[dict]:
    seen = set()
    result = []

    for edge in edges or []:
        source = edge.get("source")
        target = edge.get("target")

        if not source or not target or source == target:
            continue

        key = (source, target)
        if key in seen:
            continue

        seen.add(key)
        new_edge = edge.copy()
        label = str(new_edge.get("label", "")).strip()

        if label in {"→", "->", "-->", "=>"}:
            label = ""

        new_edge["label"] = label
        result.append(new_edge)

    return result





def load_diagram_types(folder: str | None = None) -> list:
    if folder is None:
        folder = str(Path(__file__).resolve().parent / "diagram_types")
    path = Path(folder)
    configs = []

    if not path.exists():
        print(f"Папка не найдена: {folder}")
        return configs

    for file in path.glob("*.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                print(f"Пропускаю пустой файл: {file}")
                continue

            configs.append(json.loads(content))

        except json.JSONDecodeError as e:
            print(f"Пропускаю битый JSON-файл: {file}")
            print(f"Ошибка: {e}")
            continue

    return configs


DIAGRAM_CONFIGS = load_diagram_types()

if not DIAGRAM_CONFIGS:
    DIAGRAM_CONFIGS = [
        {
            "name": "general",
            "keywords": [],
            "system_prompt": (
                "Ты генератор диаграмм. "
                "Создавай логически правильные и визуально привлекательные схемы. "
                "Верни только валидный JSON без markdown и пояснений."
            ),
            "extra_rules": [
                "схема должна быть визуально чистой",
                "подписи должны быть короткими",
                "для обычных переходов label должен быть пустым"
            ],
            "layout_hint": "general",
            "renderer": "general"
        }
    ]





def ask_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: tuple[int, int] = (30, 300),
    temperature: float = 0.15,
    response_format: dict[str, Any] | None = None,
) -> str:
    endpoint = _get_endpoint(model)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    if response_format is not None:
        payload["response_format"] = response_format

    response = requests.post(endpoint, json=payload, timeout=timeout)
    if response.status_code >= 400:
        print(f"[LLM ERROR] {response.status_code}: {response.text}")
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def extract_json(text: str) -> dict:
    return pipeline_extract_json(text)


def restore_node_kinds(original: dict, improved: dict) -> dict:
    original_kinds = {
        node["id"]: node.get("kind")
        for node in original.get("nodes", [])
        if "id" in node
    }

    for node in improved.get("nodes", []):
        if "kind" not in node:
            old_kind = original_kinds.get(node.get("id"))
            if old_kind:
                node["kind"] = old_kind

    return improved


def detect_diagram_mode(user_task: str, configs: list[dict]) -> str:
    text = user_task.lower()
    matches = []

    for config in configs:
        keywords = config.get("keywords", [])
        for word in keywords:
            idx = text.find(word.lower())
            if idx != -1:
                matches.append((idx, config["name"]))

    if matches:
        matches.sort(key=lambda x: x[0])
        return matches[0][1]

    # Heuristic fallback: if request clearly asks for NN/ML architecture diagram,
    # force model_architecture mode even when keywords in config were not hit.
    architecture_markers = [
        "архитектур", "architecture", "нейросет", "neural", "модель",
        "model", "3д", "3d", "plotneuralnet", "resnet", "реснет",
        "unet", "юнет", "transformer", "трансформер", "gan", "yolo",
        "encoder", "decoder", "квен", "qwen", "anfis", "анфис",
    ]
    if any(marker in text for marker in architecture_markers):
        known_modes = {str(c.get("name", "")).strip() for c in configs}
        if "model_architecture" in known_modes:
            return "model_architecture"

    return "general"


def get_diagram_config(mode: str, configs: list[dict]) -> dict:
    for config in configs:
        if config["name"] == mode:
            return config

    for config in configs:
        if config["name"] == "general":
            return config

    raise ValueError(f"Не найден конфиг для режима: {mode}")


def get_node_level(node_id: str, label: str, kind: str | None = None) -> int:
    kind = str(kind or "").lower().strip()
    text = f"{node_id} {label}".lower()

    if kind == "input":
        if any(x in text for x in ["интерфейс", "веб", "панель", "портал", "api", "ui"]):
            return 1
        return 0

    if kind == "conv":
        return 1

    if kind == "block":
        return 2

    if kind == "output":
        return 3

    if any(x in text for x in ["пользователь", "модератор", "администратор", "оператор"]):
        return 0
    if any(x in text for x in ["интерфейс", "веб", "панель", "портал", "api", "ui"]):
        return 1
    if any(x in text for x in ["база", "хранилище", "телеметр", "лог", "данн"]):
        return 3

    return 2


def get_pipeline_lane(node_id: str, label: str) -> str:
    text = f"{node_id} {label}".lower()

    if any(x in text for x in [
        "user input", "query", "запрос", "query embedding", "query vectorization",
        "document retrieval", "retrieval", "поиск", "relevance ranking", "ranking",
        "ранжирование", "llm context", "language model", "llm",
        "final response", "answer", "ответ", "context", "контекст"
    ]):
        return "query_pipeline"

    if any(x in text for x in [
        "document", "text extraction", "извлечение текста",
        "text preprocessing", "cleaning", "очистка",
        "document segmentation", "chunking", "чанк",
        "embedding generation", "embeddings",
        "vector db", "vector database", "векторная бд", "index", "индексация"
    ]):
        return "document_pipeline"

    return "document_pipeline"


def auto_style_node(node: dict) -> dict:
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


def get_general_node_sort_key(node: dict) -> tuple[int, int, str]:
    label = str(node.get("label", "")).lower()
    kind = str(node.get("kind", "")).lower()

    if kind == "input":
        if "пользователь" in label:
            return (0, 0, label)
        if "админ" in label or "администратор" in label or "модератор" in label:
            return (0, 1, label)
        if "внеш" in label or "интеграц" in label:
            return (0, 2, label)
        return (0, 9, label)

    if kind == "conv":
        if "веб" in label:
            return (1, 0, label)
        if "программный интерфейс" in label:
            return (1, 1, label)
        return (1, 9, label)

    if kind == "block":
        if "обработ" in label:
            return (2, 0, label)
        if "аутентиф" in label:
            return (2, 1, label)
        if "анализ" in label or "аналит" in label:
            return (2, 2, label)
        if "уведом" in label:
            return (2, 3, label)
        if "логир" in label:
            return (2, 4, label)
        return (2, 9, label)

    if kind == "output":
        if "пользоват" in label:
            return (3, 0, label)
        if "материал" in label:
            return (3, 1, label)
        if "результ" in label:
            return (3, 2, label)
        if "аналит" in label:
            return (3, 3, label)
        if "лог" in label:
            return (3, 4, label)
        if "телеметр" in label:
            return (3, 5, label)
        return (3, 9, label)

    return (9, 9, label)





def score_general_candidate(diagram: dict) -> int:
    score = 100

    nodes = diagram.get("nodes", [])
    edges = diagram.get("edges", [])

    if len(nodes) < 3:
        score -= 30
    if len(nodes) > 12:
        score -= (len(nodes) - 12) * 4

    for node in nodes:
        label = str(node.get("label", ""))
        lower = label.lower()

        if any(tok in lower for tok in FORBIDDEN_VISIBLE_TOKENS):
            score -= 20

        if len(label) > 28:
            score -= 5

        kind = node.get("kind", "")
        if kind == "input" and any(x in lower for x in ["база", "данн", "телеметр", "материал"]):
            score -= 10
        if kind == "output" and any(x in lower for x in ["пользователь", "модератор", "администратор"]):
            score -= 10

    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if not source or not target:
            score -= 10
            continue

        src_node = next((n for n in nodes if n.get("id") == source), None)
        dst_node = next((n for n in nodes if n.get("id") == target), None)
        if not src_node or not dst_node:
            score -= 10
            continue

        src_level = get_node_level(source, src_node.get(
            "label", ""), src_node.get("kind"))
        dst_level = get_node_level(target, dst_node.get(
            "label", ""), dst_node.get("kind"))

        if dst_level < src_level:
            score -= 4
        if abs(dst_level - src_level) > 2:
            score -= 6

    return score


def render_general_diagram(
    diagram: dict,
    output_name: str = "final_diagram",
    output_dir: str | Path = "outputs",
) -> str:
    def chunk_nodes(items, size):
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
    level_groups = {}

    for node in styled_nodes:
        node_id = node["id"]
        label = node.get("label", node_id)
        kind = node.get("kind")
        level = get_node_level(node_id, label, kind)

        node_levels[node_id] = level
        level_groups.setdefault(level, []).append(node)

    for level in level_groups:
        level_groups[level] = sorted(
            level_groups[level], key=get_general_node_sort_key)

    max_per_row_by_level = {
        0: 2,
        1: 2,
        2: 3,
        3: 3,
    }

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


def render_pipeline_diagram(
    diagram: dict,
    output_name: str = "diagram",
    output_dir: str | Path = "outputs",
):
    from graphviz import Digraph

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

    styled_nodes = [auto_style_node(node) for node in diagram.get("nodes", [])]

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
        "user_input",
        "query_vectorization",
        "document_retrieval",
        "relevance_ranking",
        "llm_context",
        "language_model",
        "final_response",
    ]
    document_order = [
        "document",
        "text_extraction",
        "text_preprocessing",
        "document_segmentation",
        "embedding_generation",
        "vector_database",
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


def load_references_for_mode(mode: str, base_folder: str = "references") -> list[dict]:
    refs: list[dict] = []

    base_path = Path(base_folder)
    shared_path = base_path / "shared"
    mode_path = base_path / mode

    folders_to_scan = [shared_path, mode_path]

    for folder in folders_to_scan:
        if not folder.exists() or not folder.is_dir():
            continue

        for file in sorted(folder.iterdir()):
            if not file.is_file():
                continue

            suffix = file.suffix.lower()

            if suffix == ".txt":
                content = file.read_text(encoding="utf-8").strip()
                if content:
                    refs.append({
                        "type": "text",
                        "name": file.stem,
                        "content": content
                    })

            elif suffix == ".json":
                try:
                    raw = file.read_text(encoding="utf-8").strip()
                    if not raw:
                        continue

                    content = json.loads(raw)
                    refs.append({
                        "type": "json",
                        "name": file.stem,
                        "content": content
                    })
                except json.JSONDecodeError:
                    print(f"Пропускаю битый JSON-референс: {file}")

    return refs


def normalize_references(references: list[dict] | None) -> list[dict]:
    normalized: list[dict] = []

    for ref in references or []:
        ref_type = ref.get("type", "text")
        name = ref.get("name", "reference")
        content = ref.get("content", "")

        if ref_type == "json" and isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False, indent=2)

        elif ref_type == "text":
            content = str(content)

        else:
            content = str(content)

        normalized.append({
            "type": ref_type,
            "name": name,
            "content": content
        })

    return normalized


def merge_reference_sources(
    mode: str,
    reference_description: dict | str | None = None,
    base_folder: str = "references"
) -> list[dict]:
    folder_refs = load_references_for_mode(mode, base_folder=base_folder)

    direct_refs: list[dict] = []
    if reference_description:
        if isinstance(reference_description, dict):
            direct_refs.append({
                "type": "json",
                "name": "direct_reference",
                "content": reference_description
            })
        else:
            direct_refs.append({
                "type": "text",
                "name": "direct_reference",
                "content": str(reference_description)
            })

    return normalize_references(folder_refs + direct_refs)


def format_references_for_prompt(references: list[dict]) -> str:
    if not references:
        return "Референсы не заданы."

    parts: list[str] = []

    for i, ref in enumerate(references, start=1):
        ref_type = ref["type"]
        name = ref["name"]
        content = ref["content"]

        if ref_type == "text":
            parts.append(
                f"Референс {i} (text, {name}):\n{content}"
            )

        elif ref_type == "json":
            parts.append(
                f"Референс {i} (json, {name}):\n{content}"
            )

        else:
            parts.append(
                f"Референс {i} ({ref_type}, {name}):\n{content}"
            )

    return "\n\n".join(parts)


def render_diagram(
    diagram: dict,
    output_name: str = "diagram",
    output_dir: str | Path = "outputs",
):
    return route_render_diagram(
        diagram=diagram,
        output_name=output_name,
        output_dir=output_dir,
        render_general=render_general_diagram,
        render_pipeline=render_pipeline_diagram,
    )


def critique_diagram(user_task: str, draft_json: dict, references: list[dict] | None = None) -> dict:
    references = normalize_references(references or [])
    refs_text = format_references_for_prompt(references)

    # Keep critic input compact to fit smaller context windows (e.g. 2048).
    def _compact_diagram_for_critique(diagram: dict) -> dict:
        nodes = []
        for node in diagram.get("nodes", [])[:48]:
            if not isinstance(node, dict):
                continue
            nodes.append({
                "id": str(node.get("id", ""))[:48],
                "label": str(node.get("label", ""))[:72],
                "kind": str(node.get("kind", ""))[:24],
            })

        edges = []
        for edge in diagram.get("edges", [])[:72]:
            if not isinstance(edge, dict):
                continue
            edges.append({
                "source": str(edge.get("source", ""))[:48],
                "target": str(edge.get("target", ""))[:48],
                "label": str(edge.get("label", ""))[:48],
            })

        return {
            "title": str(diagram.get("title", ""))[:120],
            "renderer": str(diagram.get("renderer", ""))[:32],
            "layout_hint": str(diagram.get("layout_hint", ""))[:32],
            "nodes": nodes,
            "edges": edges,
        }

    compact_draft = _compact_diagram_for_critique(draft_json)
    compact_refs = refs_text[:1200] if refs_text else ""

    system_prompt = (
        "Ты критик диаграмм. Верни только валидный JSON по заданной схеме. "
        "Оцени соответствие задаче, смысл, связи и читаемость. "
        "Не меняй JSON-контракт."
    )

    user_prompt = f"""
Референсы:
{compact_refs}

Запрос пользователя:
{user_task[:1200]}

Черновая схема:
{json.dumps(compact_draft, ensure_ascii=False, separators=(",", ":"))}

Верни JSON строго такого вида:
{{
  "score": 0.0,
  "task_fit_score": 0.0,
  "visual_score": 0.0,
  "missing_requirements": ["string"],
  "wrong_interpretations": ["string"],
  "extra_elements": ["string"],
  "visual_problems": ["string"],
  "problems": ["string"],
  "fixes": ["string"]
}}
"""
    raw_answer = ask_llm(
        CRITIC_MODEL,
        system_prompt,
        user_prompt,
        temperature=0.05,
        response_format={"type": "json_object"},
    )
    try:
        return extract_json(raw_answer)
    except Exception as e:
        print(f"[WARN] critique parse failed: {e}")
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        (logs_dir / f"critique_raw_{ts}.txt").write_text(raw_answer, encoding="utf-8")
        # One strict retry before fallback.
        retry_system_prompt = (
            system_prompt
            + " Верни только один валидный JSON-объект без markdown, "
            + "без пояснений, без текста до и после JSON."
        )
        retry_user_prompt = (
            user_prompt
            + "\n\nПовтори ответ СТРОГО как валидный JSON-объект, ничего кроме JSON."
        )
        try:
            retry_answer = ask_llm(
                CRITIC_MODEL,
                retry_system_prompt,
                retry_user_prompt,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return extract_json(retry_answer)
        except Exception as retry_err:
            print(f"[WARN] critique retry parse failed: {retry_err}")
            ts2 = int(time.time())
            (logs_dir / f"critique_retry_raw_{ts2}.txt").write_text(
                retry_answer if "retry_answer" in locals() else "",
                encoding="utf-8",
            )
        return {
            "score": 0.0,
            "task_fit_score": 0.0,
            "visual_score": 0.0,
            "missing_requirements": [],
            "wrong_interpretations": [],
            "extra_elements": [],
            "visual_problems": ["Критик вернул битый или обрезанный JSON"],
            "problems": [
                f"critique parse failed: {e}",
                "critique retry also failed to produce valid JSON",
            ],
            "fixes": ["Повторить критику или использовать черновую схему без изменений"],
        }


def clean_diagram_labels(diagram: dict) -> dict:
    return pipeline_clean_diagram_labels(diagram)

def build_patch_plan(critique_json: dict) -> dict:
    def _collect(items: list, severity: str, prefix: str) -> list[dict]:
        out = []
        for item in items or []:
            text = str(item or "").strip()
            if not text:
                continue
            out.append({
                "issue": text,
                "instruction": f"{prefix}{text}".strip(),
                "severity": severity,
            })
        return out

    must_fix = []
    optional = []

    must_fix += _collect(critique_json.get("missing_requirements", []), "high", "Добавь или восстанови: ")
    must_fix += _collect(critique_json.get("wrong_interpretations", []), "high", "Исправь неверную трактовку: ")
    must_fix += _collect(critique_json.get("fixes", []), "high", "")
    must_fix += _collect(critique_json.get("problems", []), "medium", "Исправь проблему: ")

    optional += _collect(critique_json.get("extra_elements", []), "medium", "Добавь или уточни при уместности: ")
    optional += _collect(critique_json.get("visual_problems", []), "medium", "Исправь визуальную проблему: ")

    def _dedupe(items: list[dict]) -> list[dict]:
        seen = set()
        out = []
        for item in items:
            key = item["issue"].strip().lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    must_fix = _dedupe(must_fix)[:6]
    optional = _dedupe(optional)[:6]

    return {
        "must_fix": must_fix,
        "optional": optional,
        "hard_constraints": [
            "Сохрани совместимость с внутренним JSON-форматом проекта",
            "Не добавляй новые enum-значения kind",
            "Сохрани title, layout_hint, renderer, style, если они корректны",
            "Все видимые label и title должны быть на русском",
            "Для general-диаграмм допустимые kind: input, conv, block, output"
        ],
    }


def compact_diagram_for_verify(diagram: dict, max_nodes: int = 16, max_edges: int = 20) -> dict:
    if not isinstance(diagram, dict):
        return {"title": "", "renderer": "", "layout_hint": "", "nodes": [], "edges": []}

    compact = {
        "title": diagram.get("title", ""),
        "renderer": diagram.get("renderer", ""),
        "layout_hint": diagram.get("layout_hint", ""),
        "nodes_count": len(diagram.get("nodes", []) or []),
        "edges_count": len(diagram.get("edges", []) or []),
        "nodes": [],
        "edges": [],
    }

    for node in (diagram.get("nodes", []) or [])[:max_nodes]:
        if isinstance(node, dict):
            compact["nodes"].append({
                "id": node.get("id", ""),
                "label": node.get("label", ""),
                "kind": node.get("kind", ""),
            })

    for edge in (diagram.get("edges", []) or [])[:max_edges]:
        if isinstance(edge, dict):
            compact["edges"].append({
                "source": edge.get("source", ""),
                "target": edge.get("target", ""),
                "label": edge.get("label", ""),
            })

    return compact


def verify_critique_application(
    user_task: str,
    draft_json: dict,
    patch_plan: dict,
    final_json: dict,
) -> dict:
    draft_nodes = len(draft_json.get("nodes", []) or [])
    final_nodes = len(final_json.get("nodes", []) or [])
    draft_edges = len(draft_json.get("edges", []) or [])
    final_edges = len(final_json.get("edges", []) or [])

    if final_nodes == 0 or final_edges == 0:
        total = len(patch_plan.get("must_fix", []) or [])
        return {
            "items": [
                {
                    "issue": item.get("issue", ""),
                    "status": "ignored",
                    "reason": "Итоговый JSON пустой или почти пустой; проверка провалена."
                }
                for item in (patch_plan.get("must_fix", []) or [])
            ],
            "summary": {
                "fixed": 0,
                "partial": 0,
                "ignored": total
            },
            "invalid_final": True
        }

    if draft_nodes > 0 and final_nodes < max(1, draft_nodes // 3):
        total = len(patch_plan.get("must_fix", []) or [])
        return {
            "items": [
                {
                    "issue": item.get("issue", ""),
                    "status": "ignored",
                    "reason": "Итоговая диаграмма потеряла слишком много узлов."
                }
                for item in (patch_plan.get("must_fix", []) or [])
            ],
            "summary": {
                "fixed": 0,
                "partial": 0,
                "ignored": total
            },
            "invalid_final": True
        }

    final_compact = compact_diagram_for_verify(final_json, max_nodes=18, max_edges=24)
    verify_patch_plan = {
        "must_fix": (patch_plan.get("must_fix", []) or [])[:5]
    }

    system_prompt = (
        "Ты проверяешь, выполнены ли пункты must_fix по итоговому JSON диаграммы. "
        "Оценивай только по final JSON. "
        "Нельзя ссылаться на patch plan как на доказательство. "
        "Если в final JSON нет явных признаков исправления, ставь ignored. "
        "Верни только JSON."
    )

    user_prompt = f"""
Запрос пользователя:
{user_task}

Must-fix:
{json.dumps(verify_patch_plan, ensure_ascii=False, indent=2)}

Final JSON:
{json.dumps(final_compact, ensure_ascii=False, indent=2)}

Верни JSON строго такого вида:
{{
  "items": [
    {{
      "issue": "string",
      "status": "fixed|partial|ignored",
      "reason": "строго укажи, какие node labels или edge связи в final это подтверждают"
    }}
  ],
  "summary": {{
    "fixed": 0,
    "partial": 0,
    "ignored": 0
  }}
}}
"""

    try:
        raw_answer = ask_llm(
            CRITIC_MODEL,
            system_prompt,
            user_prompt,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        payload = extract_json(raw_answer)
    except Exception:
        total = len(verify_patch_plan.get("must_fix", []) or [])
        payload = {
            "items": [
                {
                    "issue": item.get("issue", ""),
                    "status": "ignored",
                    "reason": "LLM verify не смог надёжно проверить итоговый JSON."
                }
                for item in (verify_patch_plan.get("must_fix", []) or [])
            ],
            "summary": {"fixed": 0, "partial": 0, "ignored": total}
        }

    items = payload.get("items", []) or []
    fixed = sum(1 for x in items if str(x.get("status", "")).lower() == "fixed")
    partial = sum(1 for x in items if str(x.get("status", "")).lower() == "partial")
    ignored = sum(1 for x in items if str(x.get("status", "")).lower() == "ignored")

    payload["summary"] = {
        "fixed": fixed,
        "partial": partial,
        "ignored": ignored,
    }
    payload["invalid_final"] = False
    return payload

def build_followup_patch_plan(patch_plan: dict, verify_json: dict) -> dict:
    unresolved = []
    unresolved_map = {
        str(item.get("issue", "")).strip().lower()
        for item in verify_json.get("items", [])
        if str(item.get("status", "")).lower() in {"partial", "ignored"}
    }

    for item in patch_plan.get("must_fix", []):
        issue = str(item.get("issue", "")).strip().lower()
        if issue in unresolved_map:
            unresolved.append(item)

    return {
        "must_fix": unresolved,
        "optional": [],
        "hard_constraints": patch_plan.get("hard_constraints", []),
    }

def improve_diagram(
    user_task: str,
    draft_json: dict,
    critique_json: dict,
    references: list[dict] | None = None,
    patch_plan: dict | None = None,
) -> tuple[dict, dict]:
    references = normalize_references(references or [])
    refs_text = format_references_for_prompt(references)
    patch_plan = patch_plan or build_patch_plan(critique_json)

    system_prompt = (
        "Ты исправляешь JSON диаграммы по patch plan. "
        "Ты не споришь с patch plan и не игнорируешь пункты must_fix. "
        "Сначала исправь все must_fix, потом optional. "
        "Главное правило: сохраняй совместимость с внутренним форматом проекта. "
        "Нельзя придумывать новые поля внутри diagram и новые значения перечислений. "
        "Для general-диаграмм допустимые значения nodes[].kind: input, conv, block, output. "
        "Если хочешь отразить actor/ui/service/database, делай это через существующие kind: "
        "actor -> input, ui -> conv, service -> block, database -> output. "
        "Все видимые подписи и title должны быть только на русском языке. "
        "Верни только валидный JSON."
    )

    user_prompt = f"""
Запрос пользователя:
{user_task}

Референсы:
{refs_text[:1200] if refs_text else ""}

Черновой JSON:
{json.dumps(draft_json, ensure_ascii=False, indent=2)}

Замечания критика:
{json.dumps(critique_json, ensure_ascii=False, indent=2)}

Patch plan:
{json.dumps(patch_plan, ensure_ascii=False, indent=2)}

Верни JSON строго такого вида:
{{
  "diagram": {{
    "title": "string",
    "layout_hint": "string",
    "renderer": "string"
  }},
  "addressed_critique": [
    {{
      "item": "string",
      "status": "fixed",
      "reason": "string"
    }}
  ]
}}
"""

    raw_answer = ask_llm(
        GENERATOR_MODEL,
        system_prompt,
        user_prompt,
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    print("\n=== RAW ANSWER FROM IMPROVER ===")
    print(raw_answer)
    print("=== END RAW ANSWER FROM IMPROVER ===\n")

    try:
        payload = extract_json(raw_answer)
        meta = {
            "addressed_critique": payload.get("addressed_critique", []),
            "raw_payload": payload,
        }
        meta = {
            "addressed_critique": payload.get("addressed_critique", []),
        }
    except Exception as e:
        print(f"[WARN] improve parse failed: {e}")
        retry_system_prompt = (
            system_prompt
            + " Верни только один валидный JSON-объект без markdown, "
            + "без пояснений, без текста до и после JSON."
        )
        retry_user_prompt = (
            user_prompt
            + "\n\nПовтори ответ СТРОГО как валидный JSON-объект, ничего кроме JSON."
        )
        try:
            retry_answer = ask_llm(
                GENERATOR_MODEL,
                retry_system_prompt,
                retry_user_prompt,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            payload = extract_json(retry_answer)
        except Exception:
            print("Ошибка парсинга improve-ответа после retry. Возвращаю draft_json.")
            return draft_json, {"addressed_critique": [], "status": "parse_failed"}

    meta = {
        "addressed_critique": payload.get("addressed_critique", []),
    }

    improved = payload.get("diagram", payload)
    improved = clean_diagram_labels(improved)
    improved = restore_node_kinds(draft_json, improved)

    if not isinstance(improved, dict):
        print("[WARN] improved is not a dict, rollback to draft")
        return draft_json, {"addressed_critique": [], "status": "rollback_non_dict"}

    draft_nodes = len(draft_json.get("nodes", []) or [])
    draft_edges = len(draft_json.get("edges", []) or [])
    new_nodes = len(improved.get("nodes", []) or [])
    new_edges = len(improved.get("edges", []) or [])

    if draft_nodes > 0 and new_nodes == 0:
        print("[WARN] improved lost all nodes, rollback to draft")
        return draft_json, {"addressed_critique": payload.get("addressed_critique", []), "status": "rollback_empty_nodes"}

    if draft_edges > 0 and new_edges == 0:
        print("[WARN] improved lost all edges, rollback to draft")
        return draft_json, {"addressed_critique": payload.get("addressed_critique", []), "status": "rollback_empty_edges"}

    if draft_nodes >= 6 and new_nodes < max(2, draft_nodes // 3):
        print("[WARN] improved lost too many nodes, rollback to draft")
        return draft_json, {"addressed_critique": payload.get("addressed_critique", []), "status": "rollback_too_few_nodes"}

    if draft_edges >= 6 and new_edges < max(2, draft_edges // 3):
        print("[WARN] improved lost too many edges, rollback to draft")
        return draft_json, {"addressed_critique": payload.get("addressed_critique", []), "status": "rollback_too_few_edges"}

    draft_renderer = str(draft_json.get("renderer", "")).lower()
    draft_layout = str(draft_json.get("layout_hint", "")).lower()

    improved.setdefault("renderer", draft_json.get("renderer", "general"))
    improved.setdefault("layout_hint", draft_json.get("layout_hint", "general"))

    is_general = draft_renderer == "general" or draft_layout == "general"
    if is_general:
        improved = normalize_general_diagram(improved, fallback=draft_json)
    else:
        improved["renderer"] = draft_json.get(
            "renderer", improved.get("renderer", "plotneuralnet")
        )
        improved["layout_hint"] = draft_json.get(
            "layout_hint", improved.get("layout_hint", "model_architecture")
        )

    return improved, meta


def image_to_data_url(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/png"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def normalize_general_diagram(diagram: dict, fallback: dict | None = None) -> dict:
    if fallback is None:
        fallback = {}

    result = dict(diagram)

    for key in ["title", "layout_hint", "renderer", "style"]:
        if key not in result and key in fallback:
            result[key] = fallback[key]

    result.setdefault("title", "Архитектура системы")
    result.setdefault("layout_hint", "general")
    result.setdefault("renderer", "general")
    result.setdefault("style", {"direction": "TB"})

    result["title"] = clean_visible_label(
        result.get("title", "Архитектура системы"))

    kind_map = {
        "actor": "input",
        "user": "input",
        "external": "input",

        "ui": "conv",
        "interface": "conv",
        "frontend": "conv",
        "api": "conv",

        "service": "block",
        "module": "block",
        "processor": "block",
        "block": "block",

        "database": "output",
        "db": "output",
        "storage": "output",
        "repository": "output",
        "output": "output",

        "input": "input",
        "conv": "conv",
    }

    fallback_kinds = {}
    for node in fallback.get("nodes", []):
        if "id" in node and "kind" in node:
            fallback_kinds[node["id"]] = node["kind"]

    new_nodes = []
    seen_ids = set()

    for node in result.get("nodes", []):
        node = dict(node)

        node_id = str(node.get("id", "")).strip()
        if not node_id:
            continue
        if node_id in seen_ids:
            continue
        seen_ids.add(node_id)

        node["label"] = clean_visible_label(node.get("label", node_id))

        raw_kind = str(node.get("kind", "")).strip().lower()
        if raw_kind in kind_map:
            node["kind"] = kind_map[raw_kind]
        elif node_id in fallback_kinds:
            node["kind"] = fallback_kinds[node_id]
        else:
            node["kind"] = infer_general_kind_from_label(node["label"])

        new_nodes.append(node)

    result["nodes"] = new_nodes
    result["edges"] = dedupe_edges(result.get("edges", []))

    return result


def analyze_reference_image(image_path: str) -> dict:
    image_data_url = image_to_data_url(image_path)

    system_prompt = (
        "Ты анализируешь PNG-референс диаграммы. "
        "Определи тип диаграммы, визуальный стиль, направление, форму узлов и общую структуру. "
        "Верни только валидный JSON без markdown и пояснений."
    )

    user_text = """
Посмотри на PNG-референс и верни JSON строго такого вида:
{
  "reference_type": "flowchart|chart|diagram|unknown",
  "style": {
    "direction": "TB|LR|unknown",
    "theme": "clean|dense|academic|minimal|unknown",
    "node_shape": "ellipse|box|mixed|unknown",
    "label_style": "short|medium|long|unknown"
  },
  "observations": [
    "string"
  ],
  "guidance_for_generation": [
    "string"
  ]
}
"""

    payload = {
        "model": VISION_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        "temperature": 0.2,
    }

    response = requests.post(_get_endpoint(VISION_MODEL), json=payload, timeout=(30, 180))
    print("STATUS:", response.status_code)
    print("RESPONSE TEXT:")
    print(response.text)
    response.raise_for_status()

    raw_answer = response.json()["choices"][0]["message"]["content"]
    print("\n=== RAW REFERENCE ANALYSIS ===")
    print(raw_answer)
    print("=== END RAW REFERENCE ANALYSIS ===\n")
    return extract_json(raw_answer)


def _is_hybrid_request(user_task: str) -> bool:
    text = str(user_task).lower()
    has_resnet = "resnet" in text or "реснет" in text
    has_qwen = "qwen" in text or "квен" in text
    has_anfis = "anfis" in text or "анфис" in text
    return has_resnet and has_qwen and has_anfis


def _hybrid_coverage(candidate: dict[str, Any]) -> tuple[float, list[str]]:
    title = str(candidate.get("title", "")).lower()
    corpus = [title]
    for node in candidate.get("nodes", []):
        if isinstance(node, dict):
            corpus.append(str(node.get("id", "")).lower())
            corpus.append(str(node.get("label", "")).lower())
            corpus.append(str(node.get("kind", "")).lower())
    text = " ".join(corpus)

    required = {
        "resnet": ("resnet" in text or "реснет" in text),
        "qwen": ("qwen" in text or "квен" in text),
        "anfis": ("anfis" in text or "анфис" in text),
        "fusion": ("fusion" in text or "concat" in text or "слияни" in text),
    }
    missing = [name for name, ok in required.items() if not ok]
    score = float(sum(1 for ok in required.values() if ok))
    # prefer richer yet readable drafts
    score += min(2.0, len(candidate.get("nodes", [])) * 0.08)
    score += min(1.0, len(candidate.get("edges", [])) * 0.05)
    return score, missing


def generate_diagram(user_task: str, reference_description: dict | str | None = None) -> dict:
    configs = DIAGRAM_CONFIGS or load_diagram_types()
    mode = detect_diagram_mode(user_task, configs)
    config = get_diagram_config(mode, configs)

    system_prompt = config["system_prompt"]
    layout_hint = config.get("layout_hint", "general")
    extra_rules = "\n".join(
        f"- {rule}" for rule in config.get("extra_rules", [])
    )

    references = merge_reference_sources(mode, reference_description)
    reference_text = format_references_for_prompt(references)

    custom_format = config.get("json_format", "")
    if custom_format:
        json_format_block = custom_format
    else:
        json_format_block = (
            '{\n'
            '  "title": "string",\n'
            f'  "layout_hint": "{layout_hint}",\n'
            f'  "renderer": "{config.get("renderer", "general")}",\n'
            '  "layout": "linear|u_shape",\n'
            '  "nodes": [\n'
            '    {\n'
            '      "id": "string",\n'
            '      "label": "string",\n'
            '      "kind": "input|conv|pool|block|fc|sum|concat|mul|output"\n'
            '    }\n'
            '  ],\n'
            '  "edges": [\n'
            '    {\n'
            '      "source": "string",\n'
            '      "target": "string"\n'
            '    }\n'
            '  ]\n'
            '}'
        )

    user_prompt = f"""
Референсы:
{reference_text}

Запрос пользователя:
{user_task}

Дополнительные требования:
{extra_rules}

Важно:
- все видимые подписи и title должны быть только на русском языке;
- не используй в label слова input, output, block, conv, ui, api, db, service;
- kind может оставаться служебным внутренним полем;
- не делай лишних связей;
- не делай один центральный хаб без необходимости;
- для general-схемы предпочитай уровни: верхний слой -> интерфейсы -> сервисы -> хранилища.

Верни только валидный JSON без markdown и пояснений.

Формат:
{json_format_block}
"""

    best_candidate = None
    best_score = -10**9
    hybrid_request = _is_hybrid_request(user_task)
    if hybrid_request:
        user_prompt += (
            "\n\nДополнительное жесткое правило для этой задачи:\n"
            "- В диаграмме ОБЯЗАТЕЛЬНО должны быть явные части: ResNet, Qwen, ANFIS и Fusion/Concat.\n"
            "- Если нет хотя бы одной части, ответ считается невалидным.\n"
        )

    for attempt in range(3):
        raw_answer = ask_llm(GENERATOR_MODEL, system_prompt,
                             user_prompt, temperature=0.12)

        print(f"\n=== RAW ANSWER FROM GENERATOR #{attempt + 1} ===")
        print(raw_answer)
        print("=== END RAW ANSWER ===\n")

        try:
            candidate = extract_json(raw_answer)
            candidate.setdefault("renderer", config.get("renderer", "general"))
            candidate.setdefault("layout_hint", layout_hint)

            if layout_hint == "general":
                candidate = normalize_general_diagram(candidate)

            candidate = clean_diagram_labels(candidate)
            if layout_hint == "general":
                score = score_general_candidate(candidate)
            elif hybrid_request:
                score, missing = _hybrid_coverage(candidate)
                if missing:
                    score -= 5.0 + len(missing)
                    print(f"[GEN HYBRID CHECK #{attempt + 1}] missing={missing}")
            else:
                score = float(len(candidate.get("nodes", [])) * 0.1 + len(candidate.get("edges", [])) * 0.05)
            print(f"[GEN SCORE #{attempt + 1}] {score}")

            if score > best_score:
                best_score = score
                best_candidate = candidate

        except Exception:
            print(
                f"Ошибка парсинга JSON в generate_diagram(), попытка {attempt + 1}")

    if best_candidate is not None:
        if hybrid_request:
            _, missing = _hybrid_coverage(best_candidate)
            if missing:
                print(f"[WARN] Hybrid draft incomplete, missing parts: {missing}")
        return best_candidate

    return {
        "type": "flowchart",
        "title": "Ошибка генерации",
        "layout_hint": layout_hint,
        "style": {"direction": "TB", "theme": "clean"},
        "lanes": [],
        "nodes": [{"id": "error", "label": "Ошибка генерации JSON", "kind": "block"}],
        "edges": [],
    }


def main(
    explain_critic_influence: bool = True,
    use_test_prompt: bool = True,
    critic_ab_replay: bool = False,
):

    default_prompt = """Нарисуй 3D архитектурную диаграмму гибридной модели “ResNet18 + Qwen 3.5 + ANFIS” в стиле PlotNeuralNet.

Цель:
Показать полный инференс-пайплайн мультимодальной гибридной системы, где:
1) ResNet-18 извлекает визуальные признаки из изображения,
2) Qwen 3.5 извлекает семантические/текстовые признаки,
3) ANFIS объединяет признаки и выполняет интерпретируемое нечеткое принятие решения.

Требования к структуре:
- renderer: plotneuralnet
- layout_hint: model_architecture
- title: "Гибридная модель ResNet18 + Qwen 3.5 + ANFIS"
- layout: linear

Обязательные блоки (слева направо):
1. "Изображение (224x224x3)" [kind=input]
2. "ResNet-18 Backbone" [kind=conv]
3. "Global Avg Pool" [kind=pool]
4. "Визуальный эмбеддинг (512)" [kind=block]

Параллельная текстовая ветка:
5. "Текстовый запрос / контекст" [kind=input]
6. "Qwen 3.5 Encoder" [kind=block]
7. "Текстовый эмбеддинг" [kind=block]

Слияние и интерпретация:
8. "Fusion (Concat/Projection)" [kind=concat]
9. "ANFIS: Фаззификация" [kind=block]
10. "ANFIS: База правил (IF-THEN)" [kind=block]
11. "ANFIS: Дефаззификация" [kind=block]
12. "Предсказание / класс" [kind=output]

Интерпретируемые выходы (обязательно):
13. "Важность признаков" [kind=output]
14. "Активированные правила ANFIS" [kind=output]
15. "Уверенность решения" [kind=output]

Обязательные связи:
- Изображение -> ResNet-18 -> GAP -> Визуальный эмбеддинг -> Fusion
- Текстовый запрос -> Qwen 3.5 Encoder -> Текстовый эмбеддинг -> Fusion
- Fusion -> ANFIS: Фаззификация -> ANFIS: База правил -> ANFIS: Дефаззификация -> Предсказание
- ANFIS: База правил -> Активированные правила ANFIS
- Fusion -> Важность признаков
- ANFIS: Дефаззификация -> Уверенность решения

Визуальные требования:
- Цвета веток разные: визуальная ветка (сине-голубая), текстовая (фиолетовая), ANFIS (оранжево-золотая), выходы (розово-красные).
- Подписи читаемые, без наложений.
- Стрелки не должны проходить через текст.
- Расстояния между блоками умеренные и одинаковые.
- Не упрощай до абстрактных 2-3 блоков; структура должна быть детальной и правдоподобной.

Формат ответа:
Верни только валидный JSON-объект диаграммы, без markdown и без пояснений.

"""
    user_task = load_user_prompt(default_prompt, prefer_test_prompt=use_test_prompt)
    if use_test_prompt and Path("_test_prompt.txt").exists():
        print("[prompt-source] Используется _test_prompt.txt (если файл не пустой)")
    else:
        print("[prompt-source] Используется default_prompt из main.py")

    # Референсы теперь пустые по умолчанию. Скрипт будет подтягивать 
    # только те файлы, что лежат в папке references/ (если они там есть). 
    references = []

    print("=== ШАГ 1: Генерация черновика ===")
    draft = generate_diagram(user_task, references)
    print(json.dumps(draft, ensure_ascii=False, indent=2))

    print("\n=== ШАГ 2: Критика ===")
    critique = critique_diagram(user_task, draft, references)
    output_filename = f"diagram_{int(time.time())}"
    run_dir = Path("outputs") / output_filename
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json_artifact("critique.json", critique, base_dir=run_dir)
    print(json.dumps(critique, ensure_ascii=False, indent=2))

    patch_plan = build_patch_plan(critique)
    save_json_artifact("patch_plan.json", patch_plan, base_dir=run_dir)

    print("\n=== ШАГ 3: Исправление ===")
    final, improve_meta = improve_diagram(
        user_task,
        draft,
        critique,
        references,
        patch_plan=patch_plan,
    )
    print(json.dumps(final, ensure_ascii=False, indent=2))
    save_json_artifact("improve_meta.json", improve_meta, base_dir=run_dir)

    if not final.get("nodes") or not final.get("edges"):
        print("[WARN] final diagram is empty after improve; rollback to draft before verify")
        final = draft

    verify = verify_critique_application(user_task, draft, patch_plan, final)
    save_json_artifact("verify.json", verify, base_dir=run_dir)

    if (
        not verify.get("invalid_final", False)
        and (
            verify.get("summary", {}).get("ignored", 0) > 0
            or verify.get("summary", {}).get("partial", 0) > 0
        )
    ):
        print("\n=== ШАГ 3.5: Дополнительный improve-pass по проигнорированным пунктам ===")
        followup_patch_plan = build_followup_patch_plan(patch_plan, verify)
        save_json_artifact("followup_patch_plan.json", followup_patch_plan, base_dir=run_dir)

        if followup_patch_plan.get("must_fix"):
            final_retry, improve_meta_retry = improve_diagram(
                user_task,
                final,
                critique,
                references,
                patch_plan=followup_patch_plan,
            )
            verify_retry = verify_critique_application(user_task, draft, patch_plan, final_retry)

            save_json_artifact("improve_meta_retry.json", improve_meta_retry, base_dir=run_dir)
            save_json_artifact("verify_retry.json", verify_retry, base_dir=run_dir)

            old_fixed = verify.get("summary", {}).get("fixed", 0)
            new_fixed = verify_retry.get("summary", {}).get("fixed", 0)
            old_ignored = verify.get("summary", {}).get("ignored", 0)
            new_ignored = verify_retry.get("summary", {}).get("ignored", 0)

            if (new_fixed > old_fixed) or (new_ignored < old_ignored):
                final = final_retry
                verify = verify_retry

    final_clean = clean_diagram_labels(final)
    save_json_artifact("draft.json", draft, base_dir=run_dir)
    save_json_artifact("final.json", final_clean, base_dir=run_dir)

    if critic_ab_replay:
        print("\n=== ШАГ 3.5: A/B replay (с критиком vs без критика) ===")
        neutral_critique = {
            "score": 1.0,
            "task_fit_score": 1.0,
            "visual_score": 1.0,
            "missing_requirements": [],
            "wrong_interpretations": [],
            "extra_elements": [],
            "visual_problems": [],
            "problems": [],
            "fixes": [],
        }
        try:
            counterfactual_final, _ = improve_diagram(
                user_task,
                draft,
                neutral_critique,
                references,
                patch_plan=build_patch_plan(neutral_critique),
            )
            counterfactual_clean = clean_diagram_labels(counterfactual_final)
            save_json_artifact(
                "counterfactual_final_no_critic.json",
                counterfactual_clean,
                base_dir=run_dir,
            )

            factual_change = compute_change_metrics(draft, final_clean)
            counter_change = compute_change_metrics(draft, counterfactual_clean)
            factual_listen = compute_critic_listening_metrics(draft, critique, final_clean, verify=verify)
            counter_listen = compute_critic_listening_metrics(draft, critique, counterfactual_clean)

            ab_report = {
                "run_id": output_filename,
                "factual_change": factual_change,
                "counterfactual_change_no_critic": counter_change,
                "factual_listening": factual_listen,
                "counterfactual_listening_no_critic": counter_listen,
                "critic_effect_delta": {
                    "alignment_gain": float(
                        factual_listen.get("critic_alignment_score", 0.0)
                        - counter_listen.get("critic_alignment_score", 0.0)
                    ),
                    "ignored_rate_reduction": float(
                        counter_listen.get("critic_ignored_rate", 0.0)
                        - factual_listen.get("critic_ignored_rate", 0.0)
                    ),
                    "change_score_delta": float(
                        factual_change.get("change_score", 0.0)
                        - counter_change.get("change_score", 0.0)
                    ),
                },
                "assumption": (
                    "Counterfactual uses same draft and model, but neutral critique. "
                    "Difference estimates practical critic effect, not strict causality."
                ),
            }
            save_json_artifact("critic_ab_replay_report.json", ab_report, base_dir=run_dir)
            (run_dir / "critic_ab_replay_summary.md").write_text(
                "# Critic A/B Replay Summary\n\n"
                f"- run_id: `{output_filename}`\n"
                f"- alignment_gain: `{ab_report['critic_effect_delta']['alignment_gain']}`\n"
                f"- ignored_rate_reduction: `{ab_report['critic_effect_delta']['ignored_rate_reduction']}`\n"
                f"- change_score_delta: `{ab_report['critic_effect_delta']['change_score_delta']}`\n\n"
                "Interpretation:\n"
                "- `alignment_gain > 0` means solution with critique follows critique better.\n"
                "- `ignored_rate_reduction > 0` means critique is less ignored in factual run.\n",
                encoding="utf-8",
            )
            print(f"[critic-ab] report={run_dir / 'critic_ab_replay_report.json'}")
        except Exception as exc:
            save_json_artifact(
                "critic_ab_replay_report.json",
                {"run_id": output_filename, "status": "error_fallback", "reason": str(exc)},
                base_dir=run_dir,
            )
            print(f"[critic-ab] fallback due to error: {exc}")

    if explain_critic_influence:
        print("\n=== ШАГ 4: Анализ влияния критика ===")
        try:
            analyzer = CriticInfluenceAnalyzer(output_root="outputs")
            influence_result = analyzer.analyze_and_save(
                run_id=output_filename,
                run_dir=run_dir,
                draft=draft,
                critique=critique,
                final=final_clean,
                verify=verify,
            )
            print(
                f"[critic-influence] status={influence_result.status}, "
                f"report={run_dir / 'critic_influence_report.json'}"
            )
        except Exception as exc:
            fallback_report = {
                "status": "error_fallback",
                "reason": str(exc),
                "run_id": output_filename,
            }
            save_json_artifact(
                "critic_influence_report.json",
                fallback_report,
                base_dir=run_dir,
            )
            (run_dir / "critic_influence_summary.md").write_text(
                "# Critic Influence Summary\n\n"
                "status: `error_fallback`\n\n"
                f"reason: `{exc}`\n",
                encoding="utf-8",
            )
            print(f"[critic-influence] fallback due to error: {exc}")
    else:
        print("\n=== ШАГ 4: Анализ влияния критика (OFF) ===")

    print("\n=== ШАГ 5: Рендер ===")
    render_diagram(final_clean, output_filename, output_dir=run_dir)

def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Diagram Agent")
    parser.add_argument(
        "--explain-critic-influence",
        choices=["on", "off"],
        default="on",
        help="Enable critic influence analytics and SHAP artifacts (default: on).",
    )
    parser.add_argument(
        "--no-auto-servers",
        action="store_true",
        help="Do not auto-start local model servers via ServerManager.",
    )
    parser.add_argument(
        "--ignore-test-prompt",
        action="store_true",
        help="Ignore _test_prompt.txt and use default prompt from main.py.",
    )
    parser.add_argument(
        "--critic-ab-replay",
        choices=["on", "off"],
        default="off",
        help="Run counterfactual improve pass with neutral critique and save A/B effect report.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()
    explain = args.explain_critic_influence == "on"
    use_test_prompt = not args.ignore_test_prompt
    ab_replay = args.critic_ab_replay == "on"
    if args.no_auto_servers:
        main(
            explain_critic_influence=explain,
            use_test_prompt=use_test_prompt,
            critic_ab_replay=ab_replay,
        )
    else:
        with ServerManager() as manager:
            manager.start_default_servers()
            main(
                explain_critic_influence=explain,
                use_test_prompt=use_test_prompt,
                critic_ab_replay=ab_replay,
            )
            manager.stop_all()
