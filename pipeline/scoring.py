"""Diagram scoring, node levelling, and sorting utilities."""

from __future__ import annotations

from pipeline.label_utils import FORBIDDEN_VISIBLE_TOKENS


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

        src_level = get_node_level(source, src_node.get("label", ""), src_node.get("kind"))
        dst_level = get_node_level(target, dst_node.get("label", ""), dst_node.get("kind"))

        if dst_level < src_level:
            score -= 4
        if abs(dst_level - src_level) > 2:
            score -= 6

    return score


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
