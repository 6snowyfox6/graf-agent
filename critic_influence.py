from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    ) -> InfluenceResult:
        run_path = Path(run_dir)
        run_path.mkdir(parents=True, exist_ok=True)

        features = extract_critique_features(critique)
        targets = compute_change_metrics(draft, final)
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

        report = {
            "status": status,
            "run_id": run_id,
            "history_size": len(history),
            "features": features,
            "targets": targets,
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
