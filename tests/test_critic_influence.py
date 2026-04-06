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


if __name__ == "__main__":
    unittest.main()
