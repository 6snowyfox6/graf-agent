from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _count_list_field(obj: dict[str, Any], key: str) -> int:
    value = obj.get(key, [])
    if isinstance(value, list):
        return len(value)
    return 0


def _text_len_list_field(obj: dict[str, Any], key: str) -> int:
    value = obj.get(key, [])
    if not isinstance(value, list):
        return 0
    total = 0
    for item in value:
        total += len(str(item))
    return total


def _list_items_as_text(obj: dict[str, Any], key: str) -> list[str]:
    value = obj.get(key, [])
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _edge_set(diagram: dict[str, Any]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for edge in diagram.get("edges", []):
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if source and target:
            out.add((source, target))
    return out


def _node_maps(diagram: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    labels: dict[str, str] = {}
    kinds: dict[str, str] = {}
    for node in diagram.get("nodes", []):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id", "")).strip()
        if not node_id:
            continue
        labels[node_id] = str(node.get("label", ""))
        kinds[node_id] = str(node.get("kind", ""))
    return labels, kinds


def extract_critique_features(critique: dict[str, Any]) -> dict[str, float]:
    score = _safe_float(critique.get("score"), 0.0)
    task_fit = _safe_float(critique.get("task_fit_score"), 0.0)
    visual = _safe_float(critique.get("visual_score"), 0.0)

    missing_items = _list_items_as_text(critique, "missing_requirements")
    wrong_items = _list_items_as_text(critique, "wrong_interpretations")
    extra_items = _list_items_as_text(critique, "extra_elements")
    visual_items = _list_items_as_text(critique, "visual_problems")
    problem_items = _list_items_as_text(critique, "problems")
    fix_items = _list_items_as_text(critique, "fixes")
    all_items_text = " ".join(missing_items + wrong_items + extra_items + visual_items + problem_items + fix_items)
    all_tokens = _tokenize(all_items_text)

    action_markers = {
        "add", "remove", "rename", "connect", "merge", "split", "replace", "fix",
        "добав", "удал", "переимен", "соедин", "объедин", "раздел", "замен", "исправ",
    }
    architecture_markers = {
        "resnet", "qwen", "anfis", "transformer", "encoder", "decoder", "unet", "yolo", "gan",
        "реснет", "квен", "анфис", "трансформер", "энкодер", "декодер", "юнет",
    }
    visual_markers = {
        "label", "color", "palette", "style", "layout", "arrow", "overlap",
        "подпись", "цвет", "палитр", "стиль", "лейаут", "стрелк", "налез",
    }
    structural_markers = {
        "node", "edge", "branch", "fusion", "concat", "skip", "output", "input",
        "узел", "ребро", "ветк", "слиян", "выход", "вход",
    }

    action_hits = len([t for t in all_tokens if t in action_markers])
    architecture_hits = len([t for t in all_tokens if t in architecture_markers])
    visual_hits = len([t for t in all_tokens if t in visual_markers])
    structural_hits = len([t for t in all_tokens if t in structural_markers])
    unique_token_count = len(all_tokens)
    all_text_len = max(1, len(all_items_text))
    specificity_score = min(1.0, unique_token_count / 40.0) * min(1.0, all_text_len / 800.0)

    features: dict[str, float] = {
        "score": score,
        "task_fit_score": task_fit,
        "visual_score": visual,
        "severity_score": max(0.0, 1.0 - score),
        "severity_task_fit": max(0.0, 1.0 - task_fit),
        "severity_visual": max(0.0, 1.0 - visual),
        "missing_requirements_count": float(_count_list_field(critique, "missing_requirements")),
        "wrong_interpretations_count": float(_count_list_field(critique, "wrong_interpretations")),
        "extra_elements_count": float(_count_list_field(critique, "extra_elements")),
        "visual_problems_count": float(_count_list_field(critique, "visual_problems")),
        "problems_count": float(_count_list_field(critique, "problems")),
        "fixes_count": float(_count_list_field(critique, "fixes")),
        "missing_requirements_text_len": float(_text_len_list_field(critique, "missing_requirements")),
        "visual_problems_text_len": float(_text_len_list_field(critique, "visual_problems")),
        "problems_text_len": float(_text_len_list_field(critique, "problems")),
        "fixes_text_len": float(_text_len_list_field(critique, "fixes")),
        "semantic_action_hits": float(action_hits),
        "semantic_architecture_hits": float(architecture_hits),
        "semantic_visual_hits": float(visual_hits),
        "semantic_structural_hits": float(structural_hits),
        "semantic_unique_tokens": float(unique_token_count),
        "critic_specificity_score": float(specificity_score),
    }
    return features


def compute_change_metrics(draft: dict[str, Any], final: dict[str, Any]) -> dict[str, float]:
    draft_labels, draft_kinds = _node_maps(draft)
    final_labels, final_kinds = _node_maps(final)
    draft_ids = set(draft_labels.keys())
    final_ids = set(final_labels.keys())

    common_ids = draft_ids & final_ids
    added_nodes = final_ids - draft_ids
    removed_nodes = draft_ids - final_ids

    changed_labels = 0
    changed_kinds = 0
    for node_id in common_ids:
        if draft_labels.get(node_id) != final_labels.get(node_id):
            changed_labels += 1
        if draft_kinds.get(node_id) != final_kinds.get(node_id):
            changed_kinds += 1

    draft_edges = _edge_set(draft)
    final_edges = _edge_set(final)
    added_edges = final_edges - draft_edges
    removed_edges = draft_edges - final_edges

    node_delta_abs = abs(len(final_ids) - len(draft_ids))
    edge_delta_abs = abs(len(final_edges) - len(draft_edges))
    semantic_delta = changed_labels + changed_kinds

    # Интегральная метрика объема изменений.
    change_score = (
        node_delta_abs * 1.5
        + edge_delta_abs * 1.2
        + len(added_nodes) * 1.4
        + len(removed_nodes) * 1.4
        + len(added_edges) * 1.1
        + len(removed_edges) * 1.1
        + semantic_delta * 1.8
    )

    return {
        "draft_nodes_count": float(len(draft_ids)),
        "final_nodes_count": float(len(final_ids)),
        "draft_edges_count": float(len(draft_edges)),
        "final_edges_count": float(len(final_edges)),
        "delta_nodes": float(len(final_ids) - len(draft_ids)),
        "delta_edges": float(len(final_edges) - len(draft_edges)),
        "delta_nodes_abs": float(node_delta_abs),
        "delta_edges_abs": float(edge_delta_abs),
        "added_nodes_count": float(len(added_nodes)),
        "removed_nodes_count": float(len(removed_nodes)),
        "added_edges_count": float(len(added_edges)),
        "removed_edges_count": float(len(removed_edges)),
        "changed_labels_count": float(changed_labels),
        "changed_kinds_count": float(changed_kinds),
        "change_score": float(change_score),
    }


def _tokenize(text: str) -> set[str]:
    text = str(text or "").lower()
    parts = re.split(r"[^a-zа-я0-9]+", text, flags=re.IGNORECASE)
    return {p for p in parts if len(p) >= 3}


def _collect_change_corpus(draft: dict[str, Any], final: dict[str, Any]) -> str:
    draft_labels, draft_kinds = _node_maps(draft)
    final_labels, final_kinds = _node_maps(final)
    draft_ids = set(draft_labels.keys())
    final_ids = set(final_labels.keys())
    common_ids = draft_ids & final_ids

    chunks: list[str] = []
    for node_id in sorted(final_ids - draft_ids):
        chunks.append(final_labels.get(node_id, ""))
        chunks.append(final_kinds.get(node_id, ""))
    for node_id in sorted(draft_ids - final_ids):
        chunks.append(draft_labels.get(node_id, ""))
        chunks.append(draft_kinds.get(node_id, ""))
    for node_id in sorted(common_ids):
        if draft_labels.get(node_id) != final_labels.get(node_id):
            chunks.append(draft_labels.get(node_id, ""))
            chunks.append(final_labels.get(node_id, ""))
        if draft_kinds.get(node_id) != final_kinds.get(node_id):
            chunks.append(draft_kinds.get(node_id, ""))
            chunks.append(final_kinds.get(node_id, ""))

    draft_edges = _edge_set(draft)
    final_edges = _edge_set(final)
    for s, t in sorted(final_edges - draft_edges):
        chunks.append(f"{s} {t}")
    for s, t in sorted(draft_edges - final_edges):
        chunks.append(f"{s} {t}")

    return " ".join(chunks)


def _collect_change_entries(draft: dict[str, Any], final: dict[str, Any]) -> list[dict[str, str]]:
    draft_labels, draft_kinds = _node_maps(draft)
    final_labels, final_kinds = _node_maps(final)
    draft_ids = set(draft_labels.keys())
    final_ids = set(final_labels.keys())
    common_ids = draft_ids & final_ids

    entries: list[dict[str, str]] = []
    for node_id in sorted(final_ids - draft_ids):
        label = final_labels.get(node_id, "")
        kind = final_kinds.get(node_id, "")
        entries.append(
            {
                "type": "added_node",
                "text": f"added node {node_id} {label} {kind}".strip(),
            }
        )
    for node_id in sorted(draft_ids - final_ids):
        label = draft_labels.get(node_id, "")
        kind = draft_kinds.get(node_id, "")
        entries.append(
            {
                "type": "removed_node",
                "text": f"removed node {node_id} {label} {kind}".strip(),
            }
        )
    for node_id in sorted(common_ids):
        d_label = draft_labels.get(node_id, "")
        f_label = final_labels.get(node_id, "")
        d_kind = draft_kinds.get(node_id, "")
        f_kind = final_kinds.get(node_id, "")
        if d_label != f_label:
            entries.append(
                {
                    "type": "changed_label",
                    "text": f"changed label {node_id}: {d_label} -> {f_label}",
                }
            )
        if d_kind != f_kind:
            entries.append(
                {
                    "type": "changed_kind",
                    "text": f"changed kind {node_id}: {d_kind} -> {f_kind}",
                }
            )

    draft_edges = _edge_set(draft)
    final_edges = _edge_set(final)
    for s, t in sorted(final_edges - draft_edges):
        entries.append({"type": "added_edge", "text": f"added edge {s} -> {t}"})
    for s, t in sorted(draft_edges - final_edges):
        entries.append({"type": "removed_edge", "text": f"removed edge {s} -> {t}"})
    return entries


def _item_matches_change(item: str, change_tokens: set[str]) -> bool:
    item_tokens = _tokenize(item)
    if not item_tokens:
        return False
    inter = item_tokens & change_tokens
    # strict enough to avoid random collisions for long phrases
    if len(item_tokens) >= 4:
        return len(inter) >= 2
    return len(inter) >= 1


def compute_critic_listening_metrics(
    draft: dict[str, Any],
    critique: dict[str, Any],
    final: dict[str, Any],
    verify: dict[str, Any] | None = None,
) -> dict[str, float]:
    if verify and verify.get("items"):
        items = verify.get("items", [])
        total = len(items)
        fixed = sum(1 for x in items if str(x.get("status", "")).lower() == "fixed")
        partial = sum(1 for x in items if str(x.get("status", "")).lower() == "partial")
        ignored = sum(1 for x in items if str(x.get("status", "")).lower() == "ignored")

        fixes_coverage = float((fixed + 0.5 * partial) / max(1, total))
        problems_addressed = fixes_coverage
        contradiction_rate = float(ignored / max(1, total))
        ignored_rate = float(ignored / max(1, total))
        precision_proxy = fixes_coverage
        recall_proxy = fixes_coverage
        listening_f1 = fixes_coverage
        listening_confidence = min(1.0, 0.4 + 0.6 * (total / 8.0))
        alignment_score = max(
            0.0,
            min(
                1.0,
                0.75 * (fixed / max(1, total))
                + 0.15 * (partial / max(1, total))
                + 0.10 * (1.0 - contradiction_rate),
            ),
        )

        return {
            "fixes_coverage": float(fixes_coverage),
            "problems_addressed_rate": float(problems_addressed),
            "critic_ignored_rate": float(ignored_rate),
            "contradiction_rate": float(contradiction_rate),
            "critic_alignment_score": float(alignment_score),
            "critic_precision_proxy": float(precision_proxy),
            "critic_recall_proxy": float(recall_proxy),
            "critic_listening_f1": float(listening_f1),
            "critic_listening_confidence": float(listening_confidence),
            "verify_based": 1.0,
        }

    change_corpus = _collect_change_corpus(draft, final)
    change_tokens = _tokenize(change_corpus)

    fixes = [str(x) for x in critique.get("fixes", []) if str(x).strip()]
    problems = [str(x) for x in critique.get("problems", []) if str(x).strip()]
    missing = [str(x) for x in critique.get("missing_requirements", []) if str(x).strip()]
    wrong = [str(x) for x in critique.get("wrong_interpretations", []) if str(x).strip()]
    extras = [str(x) for x in critique.get("extra_elements", []) if str(x).strip()]

    def _match_count(items: list[str]) -> int:
        if not items:
            return 0
        return sum(1 for item in items if _item_matches_change(item, change_tokens))

    def _match_rate(items: list[str]) -> float:
        if not items:
            return 1.0
        matched = _match_count(items)
        return float(matched / max(1, len(items)))

    fixes_coverage = _match_rate(fixes)
    problems_addressed = _match_rate(problems + missing + wrong)

    final_text = " ".join(str(n.get("label", "")) for n in final.get("nodes", []) if isinstance(n, dict))
    final_tokens = _tokenize(final_text)
    if extras:
        contrad = sum(1 for item in extras if (_tokenize(item) & final_tokens))
        contradiction_rate = float(contrad / max(1, len(extras)))
    else:
        contradiction_rate = 0.0

    ignored_rate = max(0.0, 1.0 - fixes_coverage)

    critique_tokens = _tokenize(" ".join(fixes + problems + missing + wrong + extras))
    if change_tokens:
        precision_proxy = float(len(change_tokens & critique_tokens) / max(1, len(change_tokens)))
    else:
        precision_proxy = 0.0

    actionable_items = fixes + problems + missing + wrong
    matched_actionable = _match_count(actionable_items)
    recall_proxy = float(matched_actionable / max(1, len(actionable_items))) if actionable_items else 1.0
    if precision_proxy + recall_proxy > 1e-9:
        listening_f1 = float(2.0 * precision_proxy * recall_proxy / (precision_proxy + recall_proxy))
    else:
        listening_f1 = 0.0

    evidence_scale = min(1.0, (len(actionable_items) + len(change_tokens)) / 20.0)
    listening_confidence = max(0.0, min(1.0, evidence_scale * (1.0 - 0.5 * contradiction_rate)))

    alignment_score = max(
        0.0,
        min(
            1.0,
            0.55 * fixes_coverage + 0.35 * problems_addressed + 0.10 * (1.0 - contradiction_rate),
        ),
    )

    return {
        "fixes_coverage": float(fixes_coverage),
        "problems_addressed_rate": float(problems_addressed),
        "critic_ignored_rate": float(ignored_rate),
        "contradiction_rate": float(contradiction_rate),
        "critic_alignment_score": float(alignment_score),
        "critic_precision_proxy": float(precision_proxy),
        "critic_recall_proxy": float(recall_proxy),
        "critic_listening_f1": float(listening_f1),
        "critic_listening_confidence": float(listening_confidence),
        "verify_based": 0.0,
    }


def _best_match(item: str, change_entries: list[dict[str, str]]) -> dict[str, Any]:
    item_tokens = _tokenize(item)
    best: dict[str, Any] = {
        "matched": False,
        "best_change": "",
        "best_change_type": "",
        "overlap_tokens": [],
        "overlap_score": 0.0,
    }
    if not item_tokens:
        return best

    best_score = 0.0
    best_overlap: set[str] = set()
    best_text = ""
    best_type = ""
    for entry in change_entries:
        text = str(entry.get("text", ""))
        entry_tokens = _tokenize(text)
        if not entry_tokens:
            continue
        overlap = item_tokens & entry_tokens
        if not overlap:
            continue
        score = len(overlap) / max(1, len(item_tokens))
        if len(item_tokens) >= 4 and len(overlap) < 2:
            score *= 0.6
        if score > best_score:
            best_score = score
            best_overlap = overlap
            best_text = text
            best_type = str(entry.get("type", ""))

    if best_score > 0.0:
        best["matched"] = True
        best["best_change"] = best_text
        best["best_change_type"] = best_type
        best["overlap_tokens"] = sorted(best_overlap)
        best["overlap_score"] = float(min(1.0, best_score))
    return best


def compute_critic_traceability(
    draft: dict[str, Any],
    critique: dict[str, Any],
    final: dict[str, Any],
) -> dict[str, Any]:
    change_entries = _collect_change_entries(draft, final)
    fixes = _list_items_as_text(critique, "fixes")
    problems = _list_items_as_text(critique, "problems")
    missing = _list_items_as_text(critique, "missing_requirements")
    wrong = _list_items_as_text(critique, "wrong_interpretations")

    def _link(items: list[str], kind: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in items:
            match = _best_match(item, change_entries)
            out.append(
                {
                    "kind": kind,
                    "critique_item": item,
                    **match,
                }
            )
        return out

    linked = (
        _link(fixes, "fix")
        + _link(problems, "problem")
        + _link(missing, "missing_requirement")
        + _link(wrong, "wrong_interpretation")
    )
    matched = [x for x in linked if bool(x.get("matched"))]
    unmatched = [x for x in linked if not bool(x.get("matched"))]

    matched_sorted = sorted(matched, key=lambda x: float(x.get("overlap_score", 0.0)), reverse=True)
    unmatched_sorted = sorted(unmatched, key=lambda x: (str(x.get("kind", "")), str(x.get("critique_item", ""))))

    return {
        "total_actionable_items": len(linked),
        "matched_items": len(matched_sorted),
        "unmatched_items": len(unmatched_sorted),
        "match_rate": float(len(matched_sorted) / max(1, len(linked))) if linked else 1.0,
        "top_matches": matched_sorted[:8],
        "unmatched_actionable_items": unmatched_sorted[:8],
        "change_entries_count": len(change_entries),
        "change_entries_preview": change_entries[:10],
    }


def _as_dataset_row(run_record: dict[str, Any]) -> dict[str, Any]:
    row = {
        "run_id": run_record.get("run_id"),
        "ts": run_record.get("ts"),
    }
    for k, v in run_record.get("features", {}).items():
        row[f"f__{k}"] = v
    for k, v in run_record.get("targets", {}).items():
        row[f"t__{k}"] = v
    return row


@dataclass
class InfluenceResult:
    status: str
    report: dict[str, Any]
    summary_md: str


class CriticInfluenceAnalyzer:
    def __init__(self, output_root: str | Path = "outputs") -> None:
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.history_path = self.output_root / "_critic_influence_history.jsonl"
        self.min_samples = 8
        self.target_name = "change_score"

    def _load_history(self) -> list[dict[str, Any]]:
        if not self.history_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.history_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except json.JSONDecodeError:
                continue
        return rows

    def _append_history(self, run_record: dict[str, Any]) -> None:
        line = json.dumps(_as_dataset_row(run_record), ensure_ascii=False)
        with self.history_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _build_local_contributions_fallback(
        self,
        feature_names: list[str],
        feature_row: list[float],
    ) -> dict[str, float]:
        # Если SHAP недоступен, даем простую интерпретацию:
        # вклад ~= центрированное значение признака.
        mean_abs = sum(abs(x) for x in feature_row) / max(1, len(feature_row))
        denom = mean_abs if mean_abs > 1e-9 else 1.0
        out: dict[str, float] = {}
        for name, value in zip(feature_names, feature_row):
            out[name] = value / denom
        return out

    def _fit_and_explain(
        self,
        history_rows: list[dict[str, Any]],
        current_feature_map: dict[str, float],
    ) -> tuple[str, dict[str, Any]]:
        feature_names = sorted(current_feature_map.keys())
        if len(history_rows) < self.min_samples:
            return (
                "insufficient_data",
                {
                    "reason": (
                        f"Need at least {self.min_samples} samples, have {len(history_rows)}."
                    ),
                    "local_contributions": self._build_local_contributions_fallback(
                        feature_names, [current_feature_map[n] for n in feature_names]
                    ),
                    "aggregate_importance": {},
                },
            )

        # Используем sklearn как основной путь.
        try:
            from sklearn.ensemble import RandomForestRegressor  # type: ignore
        except Exception as exc:
            return (
                "degraded_no_sklearn",
                {
                    "reason": f"sklearn unavailable: {exc}",
                    "local_contributions": self._build_local_contributions_fallback(
                        feature_names, [current_feature_map[n] for n in feature_names]
                    ),
                    "aggregate_importance": {},
                },
            )

        # Готовим матрицу.
        X: list[list[float]] = []
        y: list[float] = []
        for row in history_rows:
            try:
                x_row = [float(row.get(f"f__{name}", 0.0)) for name in feature_names]
                target = float(row.get(f"t__{self.target_name}", 0.0))
            except (TypeError, ValueError):
                continue
            X.append(x_row)
            y.append(target)

        if len(X) < self.min_samples:
            return (
                "insufficient_data",
                {
                    "reason": (
                        f"Need at least {self.min_samples} valid rows, have {len(X)}."
                    ),
                    "local_contributions": self._build_local_contributions_fallback(
                        feature_names, [current_feature_map[n] for n in feature_names]
                    ),
                    "aggregate_importance": {},
                },
            )

        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=1,
        )
        model.fit(X, y)

        current_row = [current_feature_map[name] for name in feature_names]
        pred = float(model.predict([current_row])[0])
        aggregate_importance = {
            name: float(imp)
            for name, imp in zip(feature_names, model.feature_importances_)
        }

        # SHAP как предпочтительный путь.
        try:
            import numpy as np  # type: ignore
            import shap  # type: ignore

            def _to_2d_matrix(values: Any) -> Any:
                """Normalize SHAP output to (n_samples, n_features)."""
                arr: Any
                if isinstance(values, list):
                    if not values:
                        return np.zeros((0, len(feature_names)), dtype=float)
                    # For regressors many SHAP versions return [array(...)].
                    arr = np.asarray(values[0], dtype=float)
                else:
                    arr = np.asarray(values, dtype=float)

                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                elif arr.ndim == 3:
                    # Rare format: keep first output dimension.
                    arr = arr[..., 0]

                if arr.shape[-1] != len(feature_names):
                    if arr.shape[0] == len(feature_names):
                        arr = arr.T
                    else:
                        raise ValueError(
                            f"Unexpected SHAP shape {arr.shape}, "
                            f"expected last dim={len(feature_names)}"
                        )
                return arr

            X_np = np.asarray(X, dtype=float)
            current_np = np.asarray([current_row], dtype=float)

            explainer = shap.TreeExplainer(model)
            shap_values_raw = explainer.shap_values(X_np)
            current_raw = explainer.shap_values(current_np)

            shap_matrix = _to_2d_matrix(shap_values_raw)
            current_matrix = _to_2d_matrix(current_raw)
            current_shap = current_matrix[0]

            local_contrib = {
                name: float(val) for name, val in zip(feature_names, current_shap)
            }
            mean_abs = np.abs(shap_matrix).mean(axis=0)
            shap_global = {
                feature_names[i]: float(mean_abs[i])
                for i in range(len(feature_names))
            }

            return (
                "ok_shap",
                {
                    "predicted_change_score": pred,
                    "local_contributions": local_contrib,
                    "aggregate_importance": shap_global,
                    "model_feature_importance": aggregate_importance,
                    "shap_enabled": True,
                    "shap_matrix": shap_matrix,
                    "X": X_np,
                    "feature_names": feature_names,
                },
            )
        except Exception as exc:
            # Fallback: explain by feature importance * centered feature.
            centered = self._build_local_contributions_fallback(feature_names, current_row)
            local_contrib = {
                name: float(centered[name] * aggregate_importance.get(name, 0.0))
                for name in feature_names
            }
            return (
                "degraded_no_shap",
                {
                    "reason": f"shap unavailable: {exc}",
                    "predicted_change_score": pred,
                    "local_contributions": local_contrib,
                    "aggregate_importance": aggregate_importance,
                    "shap_enabled": False,
                },
            )

    def _write_plots_if_possible(
        self,
        run_dir: Path,
        explain_payload: dict[str, Any],
    ) -> list[str]:
        produced: list[str] = []
        if not explain_payload.get("shap_enabled"):
            return produced
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore
            import shap  # type: ignore
        except Exception:
            return produced

        X = explain_payload.get("X")
        feature_names = explain_payload.get("feature_names")
        shap_matrix = explain_payload.get("shap_matrix")
        if not isinstance(feature_names, list) or X is None or shap_matrix is None:
            return produced

        try:
            import numpy as np  # type: ignore

            X_plot = np.asarray(X, dtype=float)
            shap_plot = np.asarray(shap_matrix, dtype=float)

            # Beeswarm
            plt.figure(figsize=(12, 7))
            shap.summary_plot(
                shap_plot,
                X_plot,
                feature_names=feature_names,
                show=False,
                plot_type="dot",
            )
            plt.gca().set_xlabel("mean(|SHAP value|)", fontsize=13, labelpad=10)
            beeswarm = run_dir / "critic_shap_beeswarm.png"
            plt.gcf().subplots_adjust(left=0.35, right=0.98, bottom=0.20, top=0.95)
            plt.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
            plt.savefig(beeswarm, dpi=170, bbox_inches="tight", pad_inches=0.25)
            plt.close()
            produced.append(beeswarm.name)

            # Bar
            plt.figure(figsize=(12, 7))
            shap.summary_plot(
                shap_plot,
                X_plot,
                feature_names=feature_names,
                show=False,
                plot_type="bar",
            )
            plt.gca().set_xlabel("mean(|SHAP value|)", fontsize=13, labelpad=10)
            bar = run_dir / "critic_shap_bar.png"
            plt.gcf().subplots_adjust(left=0.35, right=0.98, bottom=0.22, top=0.95)
            plt.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
            plt.savefig(bar, dpi=170, bbox_inches="tight", pad_inches=0.25)
            plt.close()
            produced.append(bar.name)
        except Exception:
            return produced

        return produced

    def _write_local_contrib_plot(
        self,
        run_dir: Path,
        local_contributions: dict[str, Any],
    ) -> str | None:
        if not isinstance(local_contributions, dict) or not local_contributions:
            return None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            return None

        try:
            rows: list[tuple[str, float]] = []
            for k, v in local_contributions.items():
                try:
                    rows.append((str(k), float(v)))
                except (TypeError, ValueError):
                    continue
            if not rows:
                return None

            rows = sorted(rows, key=lambda kv: abs(kv[1]), reverse=True)[:12]
            names = [r[0] for r in rows][::-1]
            vals = [r[1] for r in rows][::-1]
            colors = ["#2E8B57" if v >= 0 else "#B22222" for v in vals]

            plt.figure(figsize=(12, 7))
            plt.barh(names, vals, color=colors)
            plt.axvline(0, color="#333333", linewidth=1)
            plt.xlabel("Local contribution to predicted change_score", fontsize=12)
            plt.title("Critic Influence (Current Run)", fontsize=14)
            plt.gcf().subplots_adjust(left=0.35, right=0.98, bottom=0.14, top=0.92)
            plt.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))

            out = run_dir / "critic_shap_local_bar.png"
            plt.savefig(out, dpi=170, bbox_inches="tight", pad_inches=0.25)
            plt.close()
            return out.name
        except Exception:
            return None

    def _write_traceability_plot(
        self,
        run_dir: Path,
        traceability: dict[str, Any],
    ) -> str | None:
        if not isinstance(traceability, dict) or not traceability:
            return None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            return None

        try:
            matched = int(traceability.get("matched_items", 0))
            unmatched = int(traceability.get("unmatched_items", 0))
            top_matches = traceability.get("top_matches", [])
            if not isinstance(top_matches, list):
                top_matches = []

            fig = plt.figure(figsize=(12, 7))
            gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.45])
            ax0 = fig.add_subplot(gs[0, 0])
            ax1 = fig.add_subplot(gs[0, 1])

            ax0.bar(["Matched", "Unmatched"], [matched, unmatched], color=["#2E8B57", "#B22222"])
            ax0.set_title("Critique Item Coverage", fontsize=13)
            ax0.set_ylabel("Count", fontsize=11)
            for idx, v in enumerate([matched, unmatched]):
                ax0.text(idx, v + 0.05, str(v), ha="center", va="bottom", fontsize=10)

            rows = top_matches[:5]
            if rows:
                labels: list[str] = []
                scores: list[float] = []
                for item in rows:
                    crit = str(item.get("critique_item", "")).strip()
                    chg = str(item.get("best_change", "")).strip()
                    score = float(item.get("overlap_score", 0.0))
                    compact = f"{crit[:28]} -> {chg[:28]}"
                    labels.append(compact)
                    scores.append(score)
                ax1.barh(labels[::-1], scores[::-1], color="#3B82F6")
                ax1.set_xlim(0.0, 1.0)
                ax1.set_xlabel("Overlap score", fontsize=11)
                ax1.set_title("Top fix -> change links", fontsize=13)
            else:
                ax1.axis("off")
                ax1.text(0.5, 0.5, "No matches for top links", ha="center", va="center", fontsize=12)

            fig.suptitle("Critic Traceability (Current Run)", fontsize=14)
            fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.90, wspace=0.28)
            out = run_dir / "critic_traceability.png"
            fig.savefig(out, dpi=170, bbox_inches="tight", pad_inches=0.2)
            plt.close(fig)
            return out.name
        except Exception:
            return None

    def _make_summary_md(self, report: dict[str, Any]) -> str:
        lines: list[str] = []
        lines.append("# Critic Influence Summary")
        lines.append("")
        lines.append(f"- status: `{report.get('status')}`")
        lines.append(f"- run_id: `{report.get('run_id')}`")
        lines.append(f"- history_size: `{report.get('history_size')}`")
        lines.append("")
        lines.append("## Change Metrics")
        for k, v in report.get("targets", {}).items():
            lines.append(f"- `{k}`: `{v}`")
        lines.append("")
        lines.append("## Critic Listening Metrics")
        listening = report.get("critic_listening", {})
        if isinstance(listening, dict) and listening:
            for k, v in listening.items():
                lines.append(f"- `{k}`: `{v}`")
        else:
            lines.append("- no listening metrics available")
        lines.append("")
        lines.append("## Listening Verdict")
        if isinstance(listening, dict) and listening:
            f1 = float(listening.get("critic_listening_f1", 0.0))
            conf = float(listening.get("critic_listening_confidence", 0.0))
            align = float(listening.get("critic_alignment_score", 0.0))
            if f1 >= 0.66 and align >= 0.66:
                verdict = "generator mostly follows critic feedback"
            elif f1 >= 0.40 and align >= 0.45:
                verdict = "generator partially follows critic feedback"
            else:
                verdict = "generator weakly follows critic feedback"
            lines.append(f"- verdict: **{verdict}**")
            lines.append(f"- confidence: `{conf:.3f}`")
        else:
            lines.append("- verdict unavailable (no listening metrics)")
        lines.append("")
        lines.append("## Fix-to-Change Traceability")
        trace = report.get("critic_traceability", {})
        if isinstance(trace, dict) and trace:
            lines.append(f"- `match_rate`: `{float(trace.get('match_rate', 0.0)):.3f}`")
            lines.append(f"- `matched_items`: `{trace.get('matched_items', 0)}`")
            lines.append(f"- `unmatched_items`: `{trace.get('unmatched_items', 0)}`")
            top_matches = trace.get("top_matches", [])
            if isinstance(top_matches, list) and top_matches:
                lines.append("- top matches:")
                for item in top_matches[:4]:
                    crit = str(item.get("critique_item", "")).strip()
                    chg = str(item.get("best_change", "")).strip()
                    score = float(item.get("overlap_score", 0.0))
                    lines.append(f"  - `{crit}` -> `{chg}` (score={score:.2f})")
            top_unmatched = trace.get("unmatched_actionable_items", [])
            if isinstance(top_unmatched, list) and top_unmatched:
                lines.append("- unmatched items:")
                for item in top_unmatched[:4]:
                    crit = str(item.get("critique_item", "")).strip()
                    lines.append(f"  - `{crit}`")
        else:
            lines.append("- traceability unavailable")
        lines.append("")
        lines.append("## Top Local Contributions")
        local = report.get("local_contributions", {})
        if isinstance(local, dict) and local:
            top = sorted(local.items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:8]
            for k, v in top:
                sign = "increase" if float(v) >= 0 else "decrease"
                lines.append(f"- `{k}`: `{float(v):.4f}` ({sign} expected change)")
        else:
            lines.append("- no local contributions available")
        lines.append("")
        lines.append("## Top Aggregate Importance")
        agg = report.get("aggregate_importance", {})
        if isinstance(agg, dict) and agg:
            top = sorted(agg.items(), key=lambda kv: float(kv[1]), reverse=True)[:8]
            for k, v in top:
                lines.append(f"- `{k}`: `{float(v):.4f}`")
        else:
            lines.append("- no aggregate importance available")
        lines.append("")
        if report.get("assumptions"):
            lines.append("## Assumptions")
            for item in report.get("assumptions", []):
                lines.append(f"- {item}")
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    def analyze_and_save(
        self,
        run_id: str,
        run_dir: str | Path,
        draft: dict[str, Any],
        critique: dict[str, Any],
        final: dict[str, Any],
        verify: dict[str, Any] | None = None,
    ) -> InfluenceResult:
        run_path = Path(run_dir)
        run_path.mkdir(parents=True, exist_ok=True)

        features = extract_critique_features(critique)
        targets = compute_change_metrics(draft, final)
        listening = compute_critic_listening_metrics(draft, critique, final, verify=verify)
        traceability = compute_critic_traceability(draft, critique, final)
        targets = {
            **targets,
            "critic_alignment_score": float(listening.get("critic_alignment_score", 0.0)),
            "critic_listening_f1": float(listening.get("critic_listening_f1", 0.0)),
        }
        record = {
            "run_id": run_id,
            "ts": int(time.time()),
            "features": features,
            "targets": targets,
            "meta": {
                "draft_nodes": len(draft.get("nodes", [])),
                "final_nodes": len(final.get("nodes", [])),
            },
        }

        self._append_history(record)
        history = self._load_history()
        status, explain_payload = self._fit_and_explain(history, features)
        plot_files = self._write_plots_if_possible(run_path, explain_payload)
        local_plot = self._write_local_contrib_plot(
            run_path,
            explain_payload.get("local_contributions", {}),
        )
        if local_plot:
            plot_files.append(local_plot)
        trace_plot = self._write_traceability_plot(run_path, traceability)
        if trace_plot:
            plot_files.append(trace_plot)

        report = {
            "status": status,
            "run_id": run_id,
            "history_size": len(history),
            "features": features,
            "targets": targets,
            "critic_listening": listening,
            "critic_verify": verify or {},
            "critic_traceability": traceability,
            "predicted_change_score": explain_payload.get("predicted_change_score"),
            "local_contributions": explain_payload.get("local_contributions", {}),
            "aggregate_importance": explain_payload.get("aggregate_importance", {}),
            "plot_files": plot_files,
            "fallback_reason": explain_payload.get("reason"),
            "assumptions": [
                "Surrogate target is structural change_score between draft and final.",
                "Critic influence is inferred statistically, not causally identified.",
                "SHAP may be unavailable; fallback explanation is used when needed.",
            ],
        }
        summary_md = self._make_summary_md(report)

        (run_path / "critic_influence_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (run_path / "critic_influence_summary.md").write_text(
            summary_md,
            encoding="utf-8",
        )
        return InfluenceResult(status=status, report=report, summary_md=summary_md)
