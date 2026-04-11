"""General diagram generation module (initial draft)."""

from __future__ import annotations

import json
from pathlib import Path

import requests

from config import (
    GENERATOR_MODEL,
    GENERATOR_MAX_TOKENS_ARCHITECTURE,
    GENERATOR_MAX_TOKENS_GENERAL,
)
from llm_client import ask_llm
from pipeline.json_ops import extract_json
from pipeline.diagram_cleaning import clean_diagram_labels
from pipeline.scoring import score_general_candidate
from pipeline.references import merge_reference_sources, format_references_for_prompt


def load_diagram_types(folder: str | None = None) -> list:
    if folder is None:
        folder = str(Path(__file__).resolve().parent.parent / "diagram_types")
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

    attendance_markers = [
        "архитектур", "architecture", "нейросет", "neural", "модель",
        "model", "3д", "3d", "plotneuralnet", "resnet", "реснет",
        "unet", "юнет", "transformer", "трансформер", "gan", "yolo",
        "encoder", "decoder", "квен", "qwen", "anfis", "анфис",
    ]
    if any(marker in text for marker in attendance_markers):
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


def _detect_render_contract(user_task: str, draft: dict, layout_hint: str) -> str:
    if layout_hint != "model_architecture":
        return ""
    text = user_task.lower()
    if "plotneuralnet" in text or "3d" in text or draft.get("renderer") == "plotneuralnet":
        return "plotneuralnet"
    return "mermaid_class"


def _hybrid_coverage(diagram: dict) -> tuple[float, list[str]]:
    render_contract = str(diagram.get("render_contract", "")).lower()
    if not render_contract:
        return score_general_candidate(diagram), []

    nodes = diagram.get("nodes", [])
    if render_contract == "plotneuralnet":
        missing = []
        has_conv = any("conv" in str(n.get("kind", "")).lower() for n in nodes)
        has_pool = any("pool" in str(n.get("kind", "")).lower() for n in nodes)
        if not has_conv:
            missing.append("ConvBlocks")
        if not has_pool:
            missing.append("PoolBlocks")
        return score_general_candidate(diagram), missing
    return score_general_candidate(diagram), []


def generate_diagram(
    user_task: str,
    references: list[dict] | None = None,
    *,
    normalize_general_diagram_fn=None,
) -> dict:
    mode = detect_diagram_mode(user_task, DIAGRAM_CONFIGS)
    config = get_diagram_config(mode, DIAGRAM_CONFIGS)
    layout_hint = config.get("layout_hint", "general")

    all_refs = merge_reference_sources(mode, references)
    refs_text = format_references_for_prompt(all_refs)

    user_prompt = f"Запрос пользователя:\n{user_task}"
    if refs_text:
        user_prompt = f"Извлеченные референсы и стандарты:\n{refs_text}\n\n{user_prompt}"

    system_prompt = config.get("system_prompt", "Ты генератор диаграмм.")
    extra_rules = config.get("extra_rules", [])

    if extra_rules:
        rules_text = "\n".join(f"- {r}" for r in extra_rules)
        system_prompt += f"\n\nОбязательные внутренние правила проекта:\n{rules_text}"

    hybrid_request = layout_hint == "model_architecture" and "hybrid_contract" in config.get("extra_rules", [])

    if layout_hint == "general":
        system_prompt += (
            "\n\nВерни JSON строго такого вида:\n"
            "{\n"
            '  "title": "Краткое название диаграммы",\n'
            '  "nodes": [\n'
            '    {"id": "n1", "label": "Подпись узла", "kind": "input|conv|block|output"}\n'
            "  ],\n"
            '  "edges": [\n'
            '    {"source": "n1", "target": "n2", "label": "Действие (опционально, если не обычный->переход)"}\n'
            "  ]\n"
            "}\n"
            "Используй kind: input (внешние/пользователи), conv (интерфейсы/API), "
            "block (логика), output (базы/результаты)."
        )
    elif layout_hint == "model_architecture":
        system_prompt += (
            "\n\nВерни JSON строго такого вида:\n"
            "{\n"
            '  "title": "Краткое название",\n'
            '  "renderer": "plotneuralnet",\n'
            '  "layout": "linear|u_shape",\n'
            '  "render_contract": "plotneuralnet",\n'
            '  "nodes": [\n'
            '    {"id": "node1", "label": "Conv2D 64", "kind": "input|conv|pool|block|fc|sum|concat|mul|output"}\n'
            "  ],\n"
            '  "edges": [\n'
            '    {"source": "node1", "target": "node2", "label": "forward"}\n'
            "  ]\n"
            "}\n"
            "Внимание: если от тебя просят PlotNeuralNet, `render_contract` должен быть `plotneuralnet`, а виды (kind) строго:\n"
            "- input: входной тензор/изображение\n"
            "- conv: блок сверток (+bn+relu)\n"
            "- pool: пулинг/downsample\n"
            "- block: bottleneck или сложный блок\n"
            "- fc: полносвязный слой\n"
            "- sum: skip-connection сложение\n"
            "- concat: skip-connection конкатенация\n"
            "- mul: перемножение (внимание)\n"
            "- output: результат/маска\n"
            "- ВАЖНО: Если ты используешь sum/concat/mul, к ним обязательно должно идти больше одной стрелки (edge)!\n"
            "- Если нет хотя бы одной части, ответ считается невалидным.\n"
        )

    gen_max_tokens = GENERATOR_MAX_TOKENS_ARCHITECTURE if layout_hint == "model_architecture" else GENERATOR_MAX_TOKENS_GENERAL
    max_attempts = 1 if layout_hint == "model_architecture" else 3

    best_candidate = None
    best_score = -9999.0

    for attempt in range(max_attempts):
        try:
            raw_answer = ask_llm(
                GENERATOR_MODEL,
                system_prompt,
                user_prompt,
                temperature=0.12,
                max_tokens=gen_max_tokens,
                response_format={"type": "json_object"},
            )
        except requests.exceptions.ReadTimeout:
            print(f"[WARN] generator timeout on attempt #{attempt + 1}")
            if best_candidate is not None:
                break
            continue
        except requests.exceptions.RequestException as e:
            print(f"[WARN] generator request failed on attempt #{attempt + 1}: {e}")
            if best_candidate is not None:
                break
            continue

        print(f"\n=== RAW ANSWER FROM GENERATOR #{attempt + 1} ===")
        print(raw_answer)
        print("=== END RAW ANSWER ===\n")

        try:
            candidate = extract_json(raw_answer)
            candidate.setdefault("renderer", config.get("renderer", "general"))
            candidate.setdefault("layout_hint", layout_hint)

            if layout_hint == "model_architecture":
                contract = _detect_render_contract(user_task, candidate, layout_hint)
                if contract:
                    candidate.setdefault("render_contract", contract)

            if layout_hint == "general" and normalize_general_diagram_fn is not None:
                candidate = normalize_general_diagram_fn(candidate)

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
            print(f"Ошибка парсинга JSON в generate_diagram(), попытка {attempt + 1}")

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
