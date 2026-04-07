from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as app_main


class MainOrchestrationTests(unittest.TestCase):
    def test_main_writes_run_artifacts_with_shap_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prev = os.getcwd()
            os.chdir(tmp)
            try:
                draft = {
                    "title": "Draft",
                    "renderer": "general",
                    "layout_hint": "general",
                    "nodes": [{"id": "a", "label": "A", "kind": "input"}],
                    "edges": [],
                }
                critique = {
                    "score": 0.5,
                    "task_fit_score": 0.5,
                    "visual_score": 0.5,
                    "missing_requirements": ["x"],
                    "wrong_interpretations": [],
                    "extra_elements": [],
                    "visual_problems": [],
                    "problems": ["p1"],
                    "fixes": ["f1"],
                }
                final = {
                    "title": "Final",
                    "renderer": "general",
                    "layout_hint": "general",
                    "nodes": [{"id": "a", "label": "A2", "kind": "input"}],
                    "edges": [],
                }

                with (
                    patch.object(app_main, "generate_diagram", return_value=draft),
                    patch.object(app_main, "critique_diagram", return_value=critique),
                    patch.object(app_main, "improve_diagram", return_value=final),
                    patch.object(app_main, "render_diagram", return_value=None),
                    patch.object(app_main.time, "time", return_value=1776000000),
                ):
                    app_main.main(explain_critic_influence=True)

                run_dir = Path("outputs") / "diagram_1776000000"
                self.assertTrue((run_dir / "draft.json").exists())
                self.assertTrue((run_dir / "critique.json").exists())
                self.assertTrue((run_dir / "final.json").exists())
                self.assertTrue((run_dir / "critic_influence_report.json").exists())
                self.assertTrue((run_dir / "critic_influence_summary.md").exists())

                report = json.loads((run_dir / "critic_influence_report.json").read_text(encoding="utf-8"))
                self.assertIn("status", report)
                self.assertIn("features", report)
                self.assertIn("targets", report)
            finally:
                os.chdir(prev)

    def test_main_skips_shap_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prev = os.getcwd()
            os.chdir(tmp)
            try:
                draft = {
                    "title": "Draft",
                    "renderer": "general",
                    "layout_hint": "general",
                    "nodes": [{"id": "a", "label": "A", "kind": "input"}],
                    "edges": [],
                }
                critique = {
                    "score": 0.5,
                    "task_fit_score": 0.5,
                    "visual_score": 0.5,
                    "missing_requirements": [],
                    "wrong_interpretations": [],
                    "extra_elements": [],
                    "visual_problems": [],
                    "problems": [],
                    "fixes": [],
                }

                with (
                    patch.object(app_main, "generate_diagram", return_value=draft),
                    patch.object(app_main, "critique_diagram", return_value=critique),
                    patch.object(app_main, "improve_diagram", return_value=draft),
                    patch.object(app_main, "render_diagram", return_value=None),
                    patch.object(app_main.time, "time", return_value=1776000001),
                ):
                    app_main.main(explain_critic_influence=False)

                run_dir = Path("outputs") / "diagram_1776000001"
                self.assertTrue((run_dir / "draft.json").exists())
                self.assertTrue((run_dir / "critique.json").exists())
                self.assertTrue((run_dir / "final.json").exists())
                self.assertFalse((run_dir / "critic_influence_report.json").exists())
                self.assertFalse((run_dir / "critic_influence_summary.md").exists())
            finally:
                os.chdir(prev)

    def test_main_writes_ab_replay_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prev = os.getcwd()
            os.chdir(tmp)
            try:
                draft = {
                    "title": "Draft",
                    "renderer": "general",
                    "layout_hint": "general",
                    "nodes": [{"id": "a", "label": "A", "kind": "input"}],
                    "edges": [],
                }
                critique = {
                    "score": 0.4,
                    "task_fit_score": 0.4,
                    "visual_score": 0.4,
                    "missing_requirements": ["qwen branch"],
                    "wrong_interpretations": [],
                    "extra_elements": [],
                    "visual_problems": [],
                    "problems": ["no fusion"],
                    "fixes": ["add qwen", "add anfis"],
                }
                final = {
                    "title": "Final",
                    "renderer": "general",
                    "layout_hint": "general",
                    "nodes": [{"id": "a", "label": "A2", "kind": "input"}],
                    "edges": [],
                }

                with (
                    patch.object(app_main, "generate_diagram", return_value=draft),
                    patch.object(app_main, "critique_diagram", return_value=critique),
                    patch.object(app_main, "improve_diagram", return_value=final),
                    patch.object(app_main, "render_diagram", return_value=None),
                    patch.object(app_main.time, "time", return_value=1776000002),
                ):
                    app_main.main(
                        explain_critic_influence=False,
                        critic_ab_replay=True,
                    )

                run_dir = Path("outputs") / "diagram_1776000002"
                self.assertTrue((run_dir / "counterfactual_final_no_critic.json").exists())
                self.assertTrue((run_dir / "critic_ab_replay_report.json").exists())
                self.assertTrue((run_dir / "critic_ab_replay_summary.md").exists())
                payload = json.loads((run_dir / "critic_ab_replay_report.json").read_text(encoding="utf-8"))
                self.assertIn("critic_effect_delta", payload)
            finally:
                os.chdir(prev)


if __name__ == "__main__":
    unittest.main()
