from __future__ import annotations

from typing import Any


def _balance_parentheses(text: str) -> str:
    opened = text.count("(")
    closed = text.count(")")
    if opened > closed:
        return text + ")" * (opened - closed)
    return text


def _recursive_clean_strings(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _recursive_clean_strings(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_recursive_clean_strings(v) for v in data]
    if isinstance(data, str):
        return _balance_parentheses(data)
    return data


def clean_diagram_labels(diagram: dict[str, Any]) -> dict[str, Any]:
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

