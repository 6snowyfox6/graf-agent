"""Improve-pass: apply critique patch plan to draft diagram."""

from __future__ import annotations

import json

import requests

from config import GENERATOR_MODEL
from llm_client import ask_llm
from pipeline.json_ops import extract_json
from pipeline.critic import build_patch_plan
from pipeline.diagram_cleaning import clean_diagram_labels
from pipeline.references import normalize_references, format_references_for_prompt


def restore_node_kinds(original: dict, improved: dict) -> dict:
    """Copy missing ``kind`` values from the original draft into *improved*."""
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


def improve_diagram(
    user_task: str,
    draft_json: dict,
    critique_json: dict,
    references: list[dict] | None = None,
    patch_plan: dict | None = None,
    *,
    normalize_general_diagram_fn=None,
) -> tuple[dict, dict]:
    """Return *(improved_diagram, meta)*.

    Parameters
    ----------
    normalize_general_diagram_fn:
        Optional callable ``(diagram, fallback) -> diagram`` used to
        normalise general-layout diagrams.  Injected from ``main.py``
        to avoid a circular import.
    """
    references = normalize_references(references or [])
    refs_text = format_references_for_prompt(references)
    patch_plan = patch_plan or build_patch_plan(critique_json)

    system_prompt = (
        "Ты исправляешь JSON диаграммы по patch plan. "
        "Ты не споришь с patch plan и не игнорируешь пункты must_fix. "
        "Сначала исправь все must_fix, потом optional. "
        "Верни ПОЛНЫЙ diagram, а не частичный. "
        "diagram обязательно должен содержать title, layout_hint, renderer, nodes, edges. "
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
    "renderer": "string",
    "layout": "linear|u_shape",
    "nodes": [
      {{
        "id": "string",
        "label": "string",
        "kind": "input|conv|pool|block|fc|sum|concat|mul|output"
      }}
    ],
    "edges": [
      {{
        "source": "string",
        "target": "string",
        "label": "string"
      }}
    ]
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

    try:
        raw_answer = ask_llm(
            GENERATOR_MODEL,
            system_prompt,
            user_prompt,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    except requests.exceptions.RequestException as req_err:
        print(f"[WARN] improve request failed: {req_err}")
        return draft_json, {"addressed_critique": [], "status": "llm_request_failed"}

    print("\n=== RAW ANSWER FROM IMPROVER ===")
    print(raw_answer)
    print("=== END RAW ANSWER FROM IMPROVER ===\n")

    try:
        payload = extract_json(raw_answer)
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
    # Some models put nodes/edges at top-level while keeping diagram header partial.
    # Merge them back to avoid false rollbacks.
    if isinstance(improved, dict) and isinstance(payload, dict):
        if not improved.get("nodes") and isinstance(payload.get("nodes"), list):
            improved["nodes"] = payload.get("nodes", [])
        if not improved.get("edges") and isinstance(payload.get("edges"), list):
            improved["edges"] = payload.get("edges", [])
        if "layout" not in improved and "layout" in payload:
            improved["layout"] = payload.get("layout")
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

    improved.setdefault("renderer", draft_json.get("renderer", "general"))
    improved.setdefault("layout_hint", draft_json.get("layout_hint", "general"))

    draft_renderer = str(draft_json.get("renderer", "")).lower()
    draft_layout = str(draft_json.get("layout_hint", "")).lower()

    is_general = draft_renderer == "general" or draft_layout == "general"
    if is_general and normalize_general_diagram_fn is not None:
        improved = normalize_general_diagram_fn(improved, fallback=draft_json)
    elif not is_general:
        improved["renderer"] = draft_json.get(
            "renderer", improved.get("renderer", "plotneuralnet")
        )
        improved["layout_hint"] = draft_json.get(
            "layout_hint", improved.get("layout_hint", "model_architecture")
        )

    return improved, meta
