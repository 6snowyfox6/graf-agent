"""Normalize general layout diagrams to enforce the schema."""

from __future__ import annotations

from pipeline.label_utils import clean_visible_label, infer_general_kind_from_label, dedupe_edges


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
