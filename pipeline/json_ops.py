from __future__ import annotations

import json
import re
import ast
from typing import Any


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()

    # убрать markdown fences
    text = re.sub(r"^\s*```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*```\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)

    decoder = json.JSONDecoder()

    # пробуем найти первый полноценный JSON-объект в тексте
    for start_idx, ch in enumerate(text):
        if ch != "{":
            continue

        try:
            obj, _ = decoder.raw_decode(text[start_idx:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    # Relaxed fallback: extract first balanced {...} chunk and try python-literal parse.
    candidate = _extract_first_balanced_object(text)
    if candidate:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        try:
            obj = ast.literal_eval(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    print("=== BAD JSON TEXT ===")
    print(text)
    print("=== END BAD JSON TEXT ===")
    raise ValueError("Не удалось извлечь JSON-объект из ответа модели")


def _extract_first_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    quote = ""
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                in_str = False
            continue

        if ch in {"'", '"'}:
            in_str = True
            quote = ch
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1].strip()

    return None


def extract_all_json_objects(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []

    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue

        try:
            obj, consumed = decoder.raw_decode(text[i:])
            if isinstance(obj, dict):
                objects.append(obj)
            i += consumed
        except json.JSONDecodeError:
            i += 1

    return objects


def coerce_improver_output_to_diagram(
    raw_answer: str,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    try:
        obj = extract_json(raw_answer)
        if isinstance(obj, dict) and isinstance(obj.get("nodes"), list):
            obj["renderer"] = fallback.get("renderer", "general")
            obj["layout_hint"] = fallback.get("layout_hint", "general")
            return obj
    except Exception:
        pass

    objects = extract_all_json_objects(raw_answer)
    if not objects:
        raise ValueError("Не удалось извлечь ни одного JSON-объекта из ответа improver")

    title = fallback.get("title", "Диаграмма")
    renderer = fallback.get("renderer", "general")
    layout_hint = fallback.get("layout_hint", "general")
    style = fallback.get("style", {"direction": "TB"})
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_edges: set[tuple[str, str]] = set()

    for obj in objects:
        if not isinstance(obj, dict):
            continue

        if "title" in obj and obj["title"]:
            title = obj["title"]

        # renderer/layout_hint больше НЕ перезаписываем из ответа модели
        if isinstance(obj.get("style"), dict):
            style = obj["style"]

        if isinstance(obj.get("nodes"), list):
            for node in obj["nodes"]:
                if not isinstance(node, dict):
                    continue
                node_id = str(node.get("id", "")).strip()
                if not node_id or node_id in seen_ids:
                    continue
                seen_ids.add(node_id)
                nodes.append(node)

        if isinstance(obj.get("edges"), list):
            for edge in obj["edges"]:
                if not isinstance(edge, dict):
                    continue
                src = str(edge.get("source", "")).strip()
                dst = str(edge.get("target", "")).strip()
                key = (src, dst)
                if not src or not dst or src == dst or key in seen_edges:
                    continue
                seen_edges.add(key)
                edges.append(edge)

    if not edges and isinstance(fallback.get("edges"), list):
        edges = fallback["edges"]

    if not nodes:
        raise ValueError("Improver не вернул пригодные nodes")

    return {
        "title": title,
        "renderer": renderer,
        "layout_hint": layout_hint,
        "style": style,
        "nodes": nodes,
        "edges": edges,
    }
