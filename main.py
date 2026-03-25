import os
import base64
import mimetypes
import json
from pathlib import Path

import requests

# Point PATH to Graphviz bin directory, not dot.exe itself.
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"
os.environ["PATH"] += os.pathsep + r"C:\Program Files (x86)\Graphviz\bin"

BASE_URL = "http://127.0.0.1:1234/v1/chat/completions"

GENERATOR_MODEL = "qwen3-8b"
VISION_MODEL = "google_gemma-3-4b-it"
CRITIC_MODEL = "google_gemma-3-4b-it"


def load_diagram_types(folder: str = "diagram_types") -> list:
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


def ask_llm(model: str, system_prompt: str, user_prompt: str, timeout: tuple[int, int] = (30, 300)) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }

    response = requests.post(BASE_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def extract_json(text: str) -> dict:
    import re

    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("Не удалось найти JSON в ответе модели")

    text = text[start:end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print("\n=== BAD JSON TEXT ===")
        print(text)
        print("=== END BAD JSON TEXT ===\n")
        raise


def detect_diagram_mode(user_task: str, configs: list[dict]) -> str:
    text = user_task.lower()

    for config in configs:
        keywords = config.get("keywords", [])
        if any(word.lower() in text for word in keywords):
            return config["name"]

    return "general"


def get_diagram_config(mode: str, configs: list[dict]) -> dict:
    for config in configs:
        if config["name"] == mode:
            return config

    for config in configs:
        if config["name"] == "general":
            return config

    raise ValueError(f"Не найден конфиг для режима: {mode}")


def get_node_level(node_id: str, label: str) -> int:
    text = f"{node_id} {label}".lower()

    # 0 — внешние акторы / пользователи
    if any(x in text for x in [
        "user", "student", "teacher", "operator", "admin",
        "пользователь", "студент", "преподаватель", "оператор", "админ"
    ]):
        return 0

    # 1 — интерфейсы / входные точки
    if any(x in text for x in [
        "ui", "web", "frontend", "dashboard", "portal", "interface",
        "веб", "интерфейс", "панель", "портал"
    ]):
        return 1

    # 2 — прикладные сервисы / бизнес-логика
    if any(x in text for x in [
        "service", "auth", "course", "testing", "analytics", "notification",
        "analysis", "processing", "risk", "monitoring", "controller",
        "сервис", "аутенти", "курс", "тест", "аналит", "уведом", "обработ",
        "анализ", "мониторинг", "контрол"
    ]):
        return 2

    # 3 — базы / knowledge / storage
    if any(x in text for x in [
        "database", "db", "storage", "repository", "knowledge", "graph",
        "vector db", "vector database", "база", "бд", "хранилище",
        "репозитор", "граф", "knowledge graph"
    ]):
        return 3

    # fallback для model architecture
    if "input" in text or "вход" in text:
        return 0
    if "backbone" in text or "encoder" in text or "decoder" in text or "block" in text:
        return 2
    if "feature" in text or "embedding" in text or "map" in text:
        return 2
    if "head" in text or "classifier" in text or "detector" in text:
        return 2
    if "output" in text or "результат" in text or "выход" in text:
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


def render_general_diagram(diagram: dict, output_name: str = "diagram"):
    from graphviz import Digraph

    dot = Digraph(comment=diagram.get("title", "Diagram"))
    dot.attr(
        rankdir="TB",
        splines="spline",
        overlap="false",
        nodesep="0.35",
        ranksep="0.5",
        bgcolor="white"
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

    level_groups: dict[int, list[dict]] = {}
    for node in styled_nodes:
        level = get_node_level(node["id"], node["label"])
        level_groups.setdefault(level, []).append(node)

    for level in sorted(level_groups.keys()):
        with dot.subgraph() as s:
            s.attr(rank="same")
            for node in level_groups[level]:
                s.node(
                    node["id"],
                    node["label"],
                    shape=node.get("shape", "box"),
                    style=node.get("style", "filled,rounded"),
                    fillcolor=node.get("fillcolor", "#d9e8fb"),
                    color=node.get("color", "#4a6fa5"),
                )

    layout_hint = diagram.get("layout_hint", "")
    for edge in diagram.get("edges", []):
        raw_label = edge.get("label", "")
        label = raw_label.strip() if isinstance(raw_label, str) else ""

        if label in {"→", "->", "-->", "=>", ""}:
            label = ""

        if layout_hint == "model_architecture":
            keep_labels = {"yes", "no", "да", "нет"}
            if label.lower() not in keep_labels:
                label = ""

        if len(label) > 18:
            label = ""

        edge_kwargs = {
            "color": edge.get("color", "#555555"),
            "style": edge.get("style", "solid"),
            "penwidth": "1.2",
        }

        edge_text = f"{edge.get('source', '')} {edge.get('target', '')} {label}".lower(
        )
        is_feedback = (
            "обрат" in edge_text
            or "feedback" in edge_text
            or edge.get("style") == "dashed"
        )

        if is_feedback:
            edge_kwargs["style"] = "dashed"
            edge_kwargs["color"] = "#999999"
            edge_kwargs["constraint"] = "false"
            edge_kwargs["penwidth"] = "1.0"
            edge_kwargs["arrowhead"] = "normal"
            label = ""

        if label:
            edge_kwargs["label"] = label

        dot.edge(edge["source"], edge["target"], **edge_kwargs)

    dot.render(output_name, format="png", cleanup=True)
    dot.render(output_name, format="svg", cleanup=True)
    dot.render(output_name, format="pdf", cleanup=True)

    print(f"Схема сохранена: {output_name}.png")
    print(f"Схема сохранена: {output_name}.svg")
    print(f"Схема сохранена: {output_name}.pdf")


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


def render_diagram(diagram: dict, output_name: str = "diagram"):
    layout_hint = diagram.get("layout_hint", "")
    if layout_hint == "pipeline":
        render_pipeline_diagram(diagram, output_name)
    else:
        render_general_diagram(diagram, output_name)


def critique_diagram(user_task: str, draft_json: dict) -> dict:
    system_prompt = (
        "Ты критик диаграмм. "
        "Проверяй не только логику и полноту, но и визуальное качество схемы. "
        "Оцени, насколько схема будет понятной, аккуратной и удобной для чтения. "
        "Проверяй, нет ли слишком длинных подписей, перегруженности, лишних шагов и запутанных ветвлений. "
        "Схема должна быть визуально чистой, компактной и хорошо восприниматься человеком. "
        "Если можно сделать схему проще, чище или красивее — укажи это. "
        "Верни только valid JSON с полями: score, problems, fixes."
    )

    user_prompt = f"""
Запрос пользователя:
{user_task}

Черновая схема:
{json.dumps(draft_json, ensure_ascii=False, indent=2)}

Оцени:
- логическую полноту
- понятность структуры
- визуальную читаемость
- длину подписей
- простоту схемы
- отсутствие перегруженности
- удобство восприятия при рендере

Верни JSON строго такого вида:
{{
  "score": 0.0,
  "problems": ["string"],
  "fixes": ["string"]
}}
"""

    raw_answer = ask_llm(CRITIC_MODEL, system_prompt, user_prompt)
    return extract_json(raw_answer)


def improve_diagram(user_task: str, draft_json: dict, critique_json: dict) -> dict:
    system_prompt = (
        "Ты генератор диаграмм. "
        "Исправь схему по замечаниям критика. "
        "Сохрани структуру схемы, но внеси только необходимые улучшения. "
        "Не переписывай схему полностью без необходимости. "
        "Верни только валидный JSON. "
        "Не используй markdown. "
        "Не пиши пояснения. "
        "Не добавляй текст до и после JSON."
    )

    user_prompt = f"""
Исправь схему по замечаниям критика.

Черновой JSON:
{json.dumps(draft_json, ensure_ascii=False)}

Замечания критика:
{json.dumps(critique_json, ensure_ascii=False)}

Верни только исправленный JSON.
"""

    raw_answer = ask_llm(GENERATOR_MODEL, system_prompt, user_prompt)

    print("\n=== RAW ANSWER FROM IMPROVER ===")
    print(raw_answer)
    print("=== END RAW ANSWER FROM IMPROVER ===\n")

    try:
        return extract_json(raw_answer)
    except Exception:
        print("Ошибка парсинга improve-ответа. Возвращаю draft_json.")
        return draft_json


def clean_diagram_labels(diagram: dict) -> dict:
    cleaned = {
        "type": diagram.get("type", ""),
        "title": diagram.get("title", ""),
        "layout_hint": diagram.get("layout_hint", "general"),
        "style": diagram.get("style", {}),
        "lanes": diagram.get("lanes", []),
        "nodes": diagram.get("nodes", []),
        "edges": [],
    }

    bad_labels = {"→", "->", "-->", "=>", ""}
    for edge in diagram.get("edges", []):
        new_edge = edge.copy()
        label = str(new_edge.get("label", "")).strip()
        if label in bad_labels:
            new_edge["label"] = ""
        cleaned["edges"].append(new_edge)

    return cleaned


def image_to_data_url(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/png"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


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

    response = requests.post(BASE_URL, json=payload, timeout=(30, 180))
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
        f"- {rule}" for rule in config.get("extra_rules", []))

    if not reference_description:
        reference_text = "Референс не задан."
    elif isinstance(reference_description, dict):
        reference_text = json.dumps(
            reference_description, ensure_ascii=False, indent=2)
    else:
        reference_text = str(reference_description)

    user_prompt = f"""
Референс:
{reference_text}

Запрос пользователя:
{user_task}

Дополнительные требования:
{extra_rules}

Верни JSON строго такого вида:
{{
  "type": "flowchart",
  "title": "string",
  "layout_hint": "{layout_hint}",
  "style": {{
    "direction": "TB",
    "theme": "clean"
  }},
  "lanes": [
    {{
      "name": "string",
      "nodes": ["string"]
    }}
  ],
  "nodes": [
    {{
      "id": "string",
      "label": "string",
      "shape": "string",
      "fillcolor": "string",
      "color": "string",
      "style": "string"
    }}
  ],
  "edges": [
    {{
      "source": "string",
      "target": "string",
      "label": "string",
      "color": "string",
      "style": "string"
    }}
  ]
}}
"""

    raw_answer = ask_llm(GENERATOR_MODEL, system_prompt, user_prompt)

    print("\n=== RAW ANSWER FROM GENERATOR ===")
    print(raw_answer)
    print("=== END RAW ANSWER ===\n")

    try:
        return extract_json(raw_answer)
    except Exception:
        print("Ошибка парсинга JSON в generate_diagram()")
        return {
            "type": "flowchart",
            "title": "Parse Error Fallback",
            "layout_hint": layout_hint,
            "style": {"direction": "TB", "theme": "clean"},
            "lanes": [],
            "nodes": [{"id": "error", "label": "Ошибка генерации JSON"}],
            "edges": [],
        }


def main():
    user_task = """Построй общую архитектурную диаграмму цифровой образовательной платформы.

Нужна универсальная архитектурная схема, а не pipeline и не архитектура нейросети.
Схема должна быть визуально чистой, компактной и подходящей для вставки в документ.
Используй короткие и понятные подписи узлов.
Не перегружай схему лишними деталями.
Для обычных переходов label должен быть пустой строкой.

Платформа включает:
1. Студент
2. Преподаватель
3. Веб-интерфейс
4. Сервис аутентификации
5. Сервис курсов
6. Сервис тестирования
7. Сервис аналитики
8. Базу данных пользователей
9. Базу данных учебных материалов
10. Сервис уведомлений

Дополнительно нужно отразить:
- студент и преподаватель взаимодействуют через веб-интерфейс
- веб-интерфейс использует сервис аутентификации
- сервис курсов и сервис тестирования работают с учебными материалами
- сервис аналитики собирает данные об активности
- сервис уведомлений отправляет сообщения пользователям

Сделай схему как общую архитектурную диаграмму системы.
Верни результат в виде структуры диаграммы."""

    image_path = "reference.png"
    reference_description = {}
    if Path(image_path).exists():
        try:
            reference_description = analyze_reference_image(image_path)
            print("=== ОПИСАНИЕ РЕФЕРЕНСА ===")
            print(json.dumps(reference_description, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"Не удалось проанализировать PNG-референс: {e}")

    print("=== ШАГ 1: Генерация черновика ===")
    draft = generate_diagram(user_task, reference_description)
    print(json.dumps(draft, ensure_ascii=False, indent=2))

    print("\n=== ШАГ 2: Критика ===")
    critique = critique_diagram(user_task, draft)
    print(json.dumps(critique, ensure_ascii=False, indent=2))

    print("\n=== ШАГ 3: Исправление ===")
    final = improve_diagram(user_task, draft, critique)
    print(json.dumps(final, ensure_ascii=False, indent=2))

    final_clean = clean_diagram_labels(final)
    render_diagram(final_clean, "final_diagram")


if __name__ == "__main__":
    main()
