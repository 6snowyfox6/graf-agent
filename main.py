import os
import base64
import mimetypes
import json
import re
import time
from pathlib import Path
import requests
from typing import Any

from plotneuralnet_renderer import PlotNeuralNetRenderer, build_model_renderer
from infographic_renderer import InfographicRenderer
from graphviz import Digraph
from server_manager import ServerManager

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
    temperature: float = 0.15
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

    response = requests.post(endpoint, json=payload, timeout=timeout)
    if response.status_code >= 400:
        print(f"[LLM ERROR] {response.status_code}: {response.text}")
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def extract_json(text: str) -> dict:
    text = text.strip()

    decoder = json.JSONDecoder()

    # пробуем найти первый JSON-объект в тексте
    for start_idx, ch in enumerate(text):
        if ch != "{":
            continue

        try:
            obj, end_idx = decoder.raw_decode(text[start_idx:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    print("=== BAD JSON TEXT ===")
    print(text)
    print("=== END BAD JSON TEXT ===")
    raise ValueError("Не удалось извлечь JSON-объект из ответа модели")


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
        styled.setdefault("fillcolor", "#d9f2d9")
        styled.setdefault("color", "#4f7f4f")
        styled.setdefault("style", "filled,bold")
    elif (
        "backbone" in label or "encoder" in label or "decoder" in label
        or "transformer" in label or "darknet" in label
        or "resnet" in label or "block" in label or "neck" in label or "модуль" in label
    ):
        styled.setdefault("shape", "box")
        styled.setdefault("fillcolor", "#d9e8fb")
        styled.setdefault("color", "#4a6fa5")
        styled.setdefault("style", "filled,rounded")
    elif (
        "feature" in label or "embedding" in label or "эмбед" in label
        or "map" in label or "representation" in label or "features" in label
        or "database" in label or "бд" in label or "vector" in label
    ):
        styled.setdefault("shape", "cylinder")
        styled.setdefault("fillcolor", "#fbe5d6")
        styled.setdefault("color", "#a65e2e")
        styled.setdefault("style", "filled,bold")
    elif (
        "head" in label or "classifier" in label or "detector" in label
        or "segmentation" in label or "classification" in label
        or "классиф" in label or "детектор" in label
    ):
        styled.setdefault("shape", "box")
        styled.setdefault("fillcolor", "#e8dcf8")
        styled.setdefault("color", "#6b4fa3")
        styled.setdefault("style", "filled,rounded")
    elif (
        "ontology" in label or "онтолог" in label or "kg" in label or "graph" in label
    ):
        styled.setdefault("shape", "cylinder")
        styled.setdefault("fillcolor", "#fbe5d6")
        styled.setdefault("color", "#a65e2e")
        styled.setdefault("style", "filled,bold")
    else:
        styled.setdefault("shape", "box")
        styled.setdefault("fillcolor", "#d9e8fb")
        styled.setdefault("color", "#4a6fa5")
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


def render_general_diagram(diagram: dict, output_name: str = "final_diagram") -> str:
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
                "fillcolor": "#d9ead3",
                "color": "#4f7f4f",
            },
            "conv": {
                "shape": "box",
                "style": "rounded,filled",
                "fillcolor": "#d9e7f7",
                "color": "#5a84c9",
            },
            "block": {
                "shape": "box",
                "style": "rounded,filled",
                "fillcolor": "#d9e7f7",
                "color": "#5a84c9",
            },
            "output": {
                "shape": "ellipse",
                "style": "filled",
                "fillcolor": "#f4cccc",
                "color": "#b45f06",
            },
            "pool": {
                "shape": "ellipse",
                "style": "filled",
                "fillcolor": "#f4cccc",
                "color": "#b45f06",
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

    output_dir = Path("outputs")
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


def render_pipeline_diagram(diagram: dict, output_name: str = "diagram"):
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

    dot.render(output_name, format="png", cleanup=True)
    dot.render(output_name, format="svg", cleanup=True)
    dot.render(output_name, format="pdf", cleanup=True)

    print(f"Схема сохранена: {output_name}.png")
    print(f"Схема сохранена: {output_name}.svg")
    print(f"Схема сохранена: {output_name}.pdf")


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


def render_diagram(diagram: dict, output_name: str = "diagram"):
    renderer = diagram.get("renderer", "")
    layout_hint = diagram.get("layout_hint", "")

    if renderer == "plotneuralnet" or layout_hint == "model_architecture":
        plot_renderer = build_model_renderer(diagram, project_root=".")
        return plot_renderer.render(diagram, output_name=output_name)

    if renderer == "infographic" or layout_hint == "infographic":
        info_renderer = InfographicRenderer()
        return info_renderer.render(diagram, output_name=output_name)

    if renderer == "pipeline" or layout_hint == "pipeline":
        return render_pipeline_diagram(diagram, output_name)

    return render_general_diagram(diagram, output_name)


def critique_diagram(user_task: str, draft_json: dict, references: list[dict] | None = None) -> dict:
    references = normalize_references(references or [])
    refs_text = format_references_for_prompt(references)

    system_prompt = (
        "Ты критик диаграмм. "
        "Оцени соответствие пользовательскому запросу и качество структуры. "
        "Но не критикуй и не предлагай удалять служебные поля внутреннего формата, "
        "если они нужны пайплайну рендера. "
        "К таким полям относятся title, renderer, layout_hint, style, nodes[].kind и другие системные поля. "
        "Ты можешь критиковать только смысл схемы, состав элементов, связи, читаемость и визуальную организацию. "
        "Нельзя предлагать менять JSON-контракт проекта."
    )

    user_prompt = f"""
Референсы:
{refs_text}

Запрос пользователя:
{user_task}

Черновая схема:
{json.dumps(draft_json, ensure_ascii=False, indent=2)}

Сделай проверку в таком порядке:

1. Выдели из запроса пользователя ключевые требования:
   - тип диаграммы
   - обязательные сущности / блоки
   - обязательные связи / поток
   - ограничения по стилю / компактности / читаемости
   - важные подписи / входы / выходы

2. Сравни требования с черновой схемой.

3. Найди:
   - что отсутствует
   - что добавлено лишнее
   - что интерпретировано неверно
   - что мешает читаемости

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
    raw_answer = ask_llm(CRITIC_MODEL, system_prompt, user_prompt)
    return extract_json(raw_answer)


def _balance_parentheses(text: str) -> str:
    opened = text.count('(')
    closed = text.count(')')
    if opened > closed:
        return text + ')' * (opened - closed)
    return text

def _recursive_clean_strings(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _recursive_clean_strings(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_recursive_clean_strings(v) for v in data]
    elif isinstance(data, str):
        return _balance_parentheses(data)
    return data

def clean_diagram_labels(diagram: dict) -> dict:
    cleaned = _recursive_clean_strings(diagram)

    bad_labels = {"→", "->", "-->", "=>", ""}
    if "edges" in cleaned:
        new_edges = []
        for edge in cleaned.get("edges", []):
            new_edge = edge.copy()
            label = str(new_edge.get("label", "")).strip()
            if label in bad_labels:
                new_edge["label"] = ""
            new_edges.append(new_edge)
        cleaned["edges"] = new_edges

    return cleaned


def improve_diagram(
    user_task: str,
    draft_json: dict,
    critique_json: dict,
    references: list[dict] | None = None
) -> dict:
    references = normalize_references(references or [])
    refs_text = format_references_for_prompt(references)

    system_prompt = (
        "Ты исправляешь JSON диаграммы. "
        "Главное правило: сохраняй совместимость с внутренним форматом проекта. "
        "Нельзя придумывать новые поля и новые значения перечислений. "
        "Для general-диаграмм допустимые значения nodes[].kind: input, conv, block, output. "
        "Нельзя заменять их на actor, ui, service, database и другие значения. "
        "Если хочешь отразить акторов, UI, сервисы и базы, делай это через существующие допустимые kind: "
        "actor -> input, ui -> conv, service -> block, database -> output. "
        "Сохраняй существующие корректные поля: title, layout_hint, renderer, style, nodes[].kind. "
        "Не удаляй поля, если они уже есть и корректны. "
        "Исправляй только то, что действительно нужно исправить. "
        "Верни только валидный JSON."
        "Все видимые подписи и title должны быть только на русском языке. "
        "Служебные kind могут оставаться внутренними: input, conv, block, output. "
        "Но label нельзя писать как input, output, block, conv, UI, API, DB. "
    )

    user_prompt = f"""
        Запрос пользователя:
        {user_task}

        Черновой JSON:
        {json.dumps(draft_json, ensure_ascii=False, indent=2)}

        Замечания критика:
        {json.dumps(critique_json, ensure_ascii=False, indent=2)}

        Важно:
        - сохрани title, layout_hint, renderer, style, если они уже есть;
        - все видимые label и title должны быть только на русском языке;
        - служебные kind оставь внутренними, но не показывай их в label;
        - не удаляй существующие корректные поля;
        - для general-диаграмм допустимые kind только: input, conv, block, output;
        - не используй actor, ui, service, database;
        - если критик пишет про actors/services/databases, отрази это через существующие kind:
        input = actors
        conv = UI
        block = services
        output = databases

        Верни только исправленный JSON.
    """
    raw_answer = ask_llm(GENERATOR_MODEL, system_prompt, user_prompt)

    print("\n=== RAW ANSWER FROM IMPROVER ===")
    print(raw_answer)
    print("=== END RAW ANSWER FROM IMPROVER ===\n")

    try:
        improved = extract_json(raw_answer)
        improved = clean_diagram_labels(improved)
        improved = restore_node_kinds(draft_json, improved)
        improved = normalize_general_diagram(improved, fallback=draft_json)
        return improved
    except Exception:
        print("Ошибка парсинга improve-ответа. Возвращаю draft_json.")
        return draft_json


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

    for attempt in range(3):
        raw_answer = ask_llm(GENERATOR_MODEL, system_prompt,
                             user_prompt, temperature=0.12)

        print(f"\n=== RAW ANSWER FROM GENERATOR #{attempt + 1} ===")
        print(raw_answer)
        print("=== END RAW ANSWER ===\n")

        try:
            candidate = extract_json(raw_answer)

            if layout_hint == "general":
                candidate = normalize_general_diagram(candidate)

            candidate = clean_diagram_labels(candidate)

            score = score_general_candidate(candidate) if layout_hint == "general" else 0
            print(f"[GEN SCORE #{attempt + 1}] {score}")

            if score > best_score:
                best_score = score
                best_candidate = candidate

        except Exception:
            print(
                f"Ошибка парсинга JSON в generate_diagram(), попытка {attempt + 1}")

    if best_candidate is not None:
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


def main():

    user_task = """2д схема автосервиса"""

    # Референсы теперь пустые по умолчанию. Скрипт будет подтягивать 
    # только те файлы, что лежат в папке references/ (если они там есть).
    references = []

    print("=== ШАГ 1: Генерация черновика ===")
    draft = generate_diagram(user_task, references)
    print(json.dumps(draft, ensure_ascii=False, indent=2))

    print("\n=== ШАГ 2: Критика ===")
    critique = critique_diagram(user_task, draft, references)
    print(json.dumps(critique, ensure_ascii=False, indent=2))

    print("\n=== ШАГ 3: Исправление ===")
    final = improve_diagram(user_task, draft, critique)
    print(json.dumps(final, ensure_ascii=False, indent=2))

    final_clean = clean_diagram_labels(final)
    
    output_filename = f"diagram_{int(time.time())}"
    render_diagram(final_clean, output_filename)

    pass


if __name__ == "__main__":
    with ServerManager() as manager:
        manager.start_default_servers()
        main()
