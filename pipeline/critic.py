"""Critique pipeline: critique_diagram, build_patch_plan, verify, followup."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

from config import CRITIC_MODEL
from llm_client import ask_llm
from pipeline.json_ops import extract_json
from pipeline.references import normalize_references, format_references_for_prompt


# ── helpers ────────────────────────────────────────────────────────────────

def is_error_draft(diagram: dict) -> bool:
    title = str(diagram.get("title", "")).lower()
    if "ошибка генерации" in title:
        return True
    nodes = diagram.get("nodes", []) or []
    edges = diagram.get("edges", []) or []
    if len(nodes) == 1 and len(edges) == 0:
        node0 = nodes[0] if isinstance(nodes[0], dict) else {}
        if str(node0.get("id", "")).lower() == "error":
            return True
    return False


def build_critique_fallback(reason: str) -> dict:
    return {
        "score": 0.0,
        "task_fit_score": 0.0,
        "visual_score": 0.0,
        "missing_requirements": [],
        "wrong_interpretations": [],
        "extra_elements": [],
        "visual_problems": ["Критика недоступна (таймаут/ошибка LLM)."],
        "problems": [reason],
        "fixes": ["Использовать черновую схему без изменений и повторить запуск позже."],
    }


# ── critique ───────────────────────────────────────────────────────────────

def critique_diagram(
    user_task: str,
    draft_json: dict,
    references: list[dict] | None = None,
) -> dict:
    if is_error_draft(draft_json):
        return build_critique_fallback(
            "Пропущен вызов критика: черновик уже аварийный (Ошибка генерации JSON)."
        )

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
    try:
        raw_answer = ask_llm(
            CRITIC_MODEL,
            system_prompt,
            user_prompt,
            temperature=0.05,
            response_format={"type": "json_object"},
        )
    except requests.exceptions.RequestException as req_err:
        print(f"[WARN] critique request failed: {req_err}")
        return build_critique_fallback(f"critique request failed: {req_err}")
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


# ── patch plan ─────────────────────────────────────────────────────────────

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

    must_fix: list[dict] = []
    optional: list[dict] = []

    must_fix += _collect(critique_json.get("missing_requirements", []), "high", "Добавь или восстанови: ")
    must_fix += _collect(critique_json.get("wrong_interpretations", []), "high", "Исправь неверную трактовку: ")
    must_fix += _collect(critique_json.get("fixes", []), "high", "")
    must_fix += _collect(critique_json.get("problems", []), "medium", "Исправь проблему: ")

    optional += _collect(critique_json.get("extra_elements", []), "medium", "Добавь или уточни при уместности: ")
    optional += _collect(critique_json.get("visual_problems", []), "medium", "Исправь визуальную проблему: ")

    def _dedupe(items: list[dict]) -> list[dict]:
        seen: set[str] = set()
        out: list[dict] = []
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


# ── verify ─────────────────────────────────────────────────────────────────

def compact_diagram_for_verify(
    diagram: dict,
    max_nodes: int = 16,
    max_edges: int = 20,
) -> dict:
    if not isinstance(diagram, dict):
        return {"title": "", "renderer": "", "layout_hint": "", "nodes": [], "edges": []}

    compact: dict[str, Any] = {
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
            "summary": {"fixed": 0, "partial": 0, "ignored": total},
            "invalid_final": True,
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
            "summary": {"fixed": 0, "partial": 0, "ignored": total},
            "invalid_final": True,
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
            "summary": {"fixed": 0, "partial": 0, "ignored": total},
        }

    items = payload.get("items", []) or []
    fixed = sum(1 for x in items if str(x.get("status", "")).lower() == "fixed")
    partial = sum(1 for x in items if str(x.get("status", "")).lower() == "partial")
    ignored = sum(1 for x in items if str(x.get("status", "")).lower() == "ignored")

    payload["summary"] = {"fixed": fixed, "partial": partial, "ignored": ignored}
    payload["invalid_final"] = False
    return payload


def build_followup_patch_plan(patch_plan: dict, verify_json: dict) -> dict:
    unresolved_map = {
        str(item.get("issue", "")).strip().lower()
        for item in verify_json.get("items", [])
        if str(item.get("status", "")).lower() in {"partial", "ignored"}
    }

    unresolved = [
        item
        for item in patch_plan.get("must_fix", [])
        if str(item.get("issue", "")).strip().lower() in unresolved_map
    ]

    return {
        "must_fix": unresolved,
        "optional": [],
        "hard_constraints": patch_plan.get("hard_constraints", []),
    }
