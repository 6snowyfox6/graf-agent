from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from critic_influence import (
    CriticInfluenceAnalyzer,
    compute_critic_listening_metrics,
    compute_change_metrics,
    extract_critique_features,
)


class CriticInfluenceTests(unittest.TestCase):
    def test_extract_critique_features_counts(self) -> None:
        critique = {
            "score": 0.4,
            "task_fit_score": 0.6,
            "visual_score": 0.5,
            "missing_requirements": ["a", "b"],
            "wrong_interpretations": ["x"],
            "extra_elements": [],
            "visual_problems": ["vp1", "vp2", "vp3"],
            "problems": ["p1"],
            "fixes": ["f1", "f2"],
        }
        f = extract_critique_features(critique)
        self.assertEqual(f["missing_requirements_count"], 2.0)
        self.assertEqual(f["visual_problems_count"], 3.0)
        self.assertAlmostEqual(f["severity_score"], 0.6, places=6)
        self.assertIn("semantic_action_hits", f)
        self.assertIn("critic_specificity_score", f)

    def test_compute_change_metrics(self) -> None:
        draft = {
            "nodes": [
                {"id": "a", "label": "A", "kind": "block"},
                {"id": "b", "label": "B", "kind": "block"},
            ],
            "edges": [{"source": "a", "target": "b"}],
        }
        final = {
            "nodes": [
                {"id": "a", "label": "A2", "kind": "block"},
                {"id": "c", "label": "C", "kind": "output"},
            ],
            "edges": [{"source": "a", "target": "c"}],
        }
        m = compute_change_metrics(draft, final)
        self.assertEqual(m["changed_labels_count"], 1.0)
        self.assertEqual(m["added_nodes_count"], 1.0)
        self.assertEqual(m["removed_nodes_count"], 1.0)
        self.assertGreater(m["change_score"], 0.0)

    def test_analyzer_writes_fallback_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            run_dir = out / "run_1"
            analyzer = CriticInfluenceAnalyzer(output_root=out)

            draft = {"nodes": [{"id": "a", "label": "A", "kind": "block"}], "edges": []}
            final = {"nodes": [{"id": "a", "label": "A1", "kind": "block"}], "edges": []}
            critique = {
                "score": 0.5,
                "task_fit_score": 0.5,
                "visual_score": 0.5,
                "missing_requirements": ["x"],
                "wrong_interpretations": [],
                "extra_elements": [],
                "visual_problems": [],
                "problems": [],
                "fixes": ["do x"],
            }

            result = analyzer.analyze_and_save(
                run_id="run_1",
                run_dir=run_dir,
                draft=draft,
                critique=critique,
                final=final,
            )
            self.assertIn(result.status, {"insufficient_data", "degraded_no_sklearn", "degraded_no_shap", "ok_shap"})
            report_path = run_dir / "critic_influence_report.json"
            summary_path = run_dir / "critic_influence_summary.md"
            self.assertTrue(report_path.exists())
            self.assertTrue(summary_path.exists())
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn("features", payload)
            self.assertIn("targets", payload)
            self.assertIn("critic_traceability", payload)

    def test_critic_listening_metrics_range(self) -> None:
        draft = {
            "nodes": [
                {"id": "a", "label": "Input", "kind": "input"},
                {"id": "b", "label": "ResNet Backbone", "kind": "block"},
            ],
            "edges": [{"source": "a", "target": "b"}],
        }
        final = {
            "nodes": [
                {"id": "a", "label": "Input", "kind": "input"},
                {"id": "b", "label": "ResNet Backbone", "kind": "block"},
                {"id": "c", "label": "Qwen Encoder", "kind": "block"},
                {"id": "d", "label": "ANFIS Fusion", "kind": "concat"},
            ],
            "edges": [
                {"source": "a", "target": "b"},
                {"source": "b", "target": "d"},
                {"source": "c", "target": "d"},
            ],
        }
        critique = {
            "fixes": ["Добавить Qwen Encoder", "Добавить ANFIS Fusion"],
            "problems": ["Нет мультимодального слияния"],
            "missing_requirements": ["Нет Qwen ветки"],
            "wrong_interpretations": [],
            "extra_elements": [],
        }
        m = compute_critic_listening_metrics(draft, critique, final)
        for k in [
            "fixes_coverage",
            "problems_addressed_rate",
            "critic_ignored_rate",
            "contradiction_rate",
            "critic_alignment_score",
            "critic_precision_proxy",
            "critic_recall_proxy",
            "critic_listening_f1",
            "critic_listening_confidence",
        ]:
            self.assertGreaterEqual(m[k], 0.0)
            self.assertLessEqual(m[k], 1.0)
        self.assertGreater(m["fixes_coverage"], 0.0)
        self.assertGreaterEqual(m["actionable_items_count"], 1.0)
        self.assertGreaterEqual(m["actionable_items_matched_count"], 1.0)

    def test_traceability_contains_matches(self) -> None:
        from critic_influence import compute_critic_traceability

        draft = {
            "nodes": [
                {"id": "a", "label": "Input", "kind": "input"},
                {"id": "b", "label": "Backbone", "kind": "block"},
            ],
            "edges": [{"source": "a", "target": "b"}],
        }
        final = {
            "nodes": [
                {"id": "a", "label": "Input", "kind": "input"},
                {"id": "b", "label": "Backbone", "kind": "block"},
                {"id": "c", "label": "Qwen Encoder", "kind": "block"},
            ],
            "edges": [
                {"source": "a", "target": "b"},
                {"source": "c", "target": "b"},
            ],
        }
        critique = {
            "fixes": ["add qwen encoder"],
            "problems": ["missing fusion branch"],
            "missing_requirements": [],
            "wrong_interpretations": [],
        }
        t = compute_critic_traceability(draft, critique, final)
        self.assertIn("match_rate", t)
        self.assertGreaterEqual(float(t["match_rate"]), 0.0)
        self.assertLessEqual(float(t["match_rate"]), 1.0)
        self.assertGreaterEqual(int(t["matched_items"]), 1)


if __name__ == "__main__":
    unittest.main()
