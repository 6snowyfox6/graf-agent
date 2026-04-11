"""Main entrypoint for Graf Agent pipeline."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

from server_manager import ServerManager
from critic_influence import (
    CriticInfluenceAnalyzer,
    compute_change_metrics,
    compute_critic_listening_metrics,
)

# Internal pipeline modules
from pipeline.diagram_cleaning import clean_diagram_labels
from pipeline.normalizer import normalize_general_diagram
from pipeline.generator import generate_diagram
from pipeline.critic import (
    critique_diagram,
    is_error_draft,
    build_patch_plan,
    verify_critique_application,
    build_followup_patch_plan,
)
from pipeline.improver import improve_diagram
from pipeline.render_router import render_diagram as route_render_diagram
from renderers.graphviz_renderers import render_general_diagram, render_pipeline_diagram


def save_json_artifact(filename: str, data: Any, base_dir: str | Path = ".") -> Path:
    out_dir = Path(base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / filename
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def render_diagram(diagram: dict, run_id: str, output_dir: str | Path = "outputs") -> str:
    """Wrapper that injects the actual renderer functions to the router."""
    return route_render_diagram(
        diagram=diagram,
        run_id=run_id,
        output_dir=output_dir,
        render_general=render_general_diagram,
        render_pipeline=render_pipeline_diagram,
    )


def load_user_prompt(default_prompt: str, prefer_test_prompt: bool = True) -> str:
    if not prefer_test_prompt:
        return default_prompt
    test_prompt_path = Path("_test_prompt.txt")
    if test_prompt_path.exists():
        text = test_prompt_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return default_prompt


def run_orchestration(
    explain_critic_influence: bool = True,
    use_test_prompt: bool = True,
    critic_ab_replay: bool = False,
):
    default_prompt = """Нарисуй 3D архитектурную диаграмму Canonical U-Net для сегментации медицинских изображений в стиле PlotNeuralNet.

Жесткие требования:
- renderer: plotneuralnet
- layout_hint: model_architecture
- layout: u_shape
- title: "Canonical U-Net"

Обязательные блоки энкодера (слева, сверху вниз):
1) Input Image (572x572x1) [kind=input]
2) Enc1 (Conv 64 x2) [kind=conv]
3) Pool1 [kind=pool]
4) Enc2 (Conv 128 x2) [kind=conv]
5) Pool2 [kind=pool]
6) Enc3 (Conv 256 x2) [kind=conv]
7) Pool3 [kind=pool]
8) Enc4 (Conv 512 x2) [kind=conv]
9) Pool4 [kind=pool]

Bottleneck:
10) Bottleneck (Conv 1024 x2) [kind=block]

Обязательные блоки декодера (справа, снизу вверх):
11) Up4 (UpConv 512) [kind=conv]
12) Concat4 (skip Enc4 + Up4) [kind=concat]
13) Dec4 (Conv 512 x2) [kind=conv]
14) Up3 (UpConv 256) [kind=conv]
15) Concat3 (skip Enc3 + Up3) [kind=concat]
16) Dec3 (Conv 256 x2) [kind=conv]
17) Up2 (UpConv 128) [kind=conv]
18) Concat2 (skip Enc2 + Up2) [kind=concat]
19) Dec2 (Conv 128 x2) [kind=conv]
20) Up1 (UpConv 64) [kind=conv]
21) Concat1 (skip Enc1 + Up1) [kind=concat]
22) Dec1 (Conv 64 x2) [kind=conv]
23) Segmentation Map (572x572x1) [kind=output]

Обязательные связи:
- Прямая цепочка через все уровни энкодер -> bottleneck -> декодер -> output.
- Skip connections: Enc4->Concat4, Enc3->Concat3, Enc2->Concat2, Enc1->Concat1.
- Concat узлы должны получать два входа: от up-блока и от соответствующего encoder-блока.

Визуальные требования:
- U-образная геометрия без наложения текста на блоки.
- Skip-стрелки аккуратные, не пересекают подписи.
- Короткие читаемые подписи.
- Декодер симметричен энкодеру по уровням каналов 64/128/256/512.
- Никаких 2D flowchart-элементов, только 3D PlotNeuralNet блоки.

Верни только валидный JSON диаграммы без markdown и пояснений.
"""

    user_task = load_user_prompt(default_prompt, prefer_test_prompt=use_test_prompt)
    if use_test_prompt and Path("_test_prompt.txt").exists():
        print("[prompt-source] Используется _test_prompt.txt (если файл не пустой)")
    else:
        print("[prompt-source] Используется default_prompt из main.py")

    references = []  # Can supply dicts of typed references if available

    print("=== ШАГ 1: Генерация черновика ===")
    draft = generate_diagram(
        user_task,
        references,
        normalize_general_diagram_fn=normalize_general_diagram
    )
    print(json.dumps(draft, ensure_ascii=False, indent=2))
    draft_is_error = is_error_draft(draft)

    output_filename = f"diagram_{int(time.time())}"
    run_dir = Path("outputs") / output_filename
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== ШАГ 2: Критика ===")
    critique = critique_diagram(user_task, draft, references)
    save_json_artifact("critique.json", critique, base_dir=run_dir)
    print(json.dumps(critique, ensure_ascii=False, indent=2))

    patch_plan = build_patch_plan(critique)
    save_json_artifact("patch_plan.json", patch_plan, base_dir=run_dir)

    print("\n=== ШАГ 3: Исправление ===")
    if draft_is_error:
        print("[WARN] Пропускаю improve: черновик аварийный, сохраняю как final.")
        final = draft
        improve_meta = {"addressed_critique": [], "status": "skipped_error_draft"}
    else:
        final, improve_meta = improve_diagram(
            user_task,
            draft,
            critique,
            references,
            patch_plan=patch_plan,
            normalize_general_diagram_fn=normalize_general_diagram,
        )
    print(json.dumps(final, ensure_ascii=False, indent=2))
    save_json_artifact("improve_meta.json", improve_meta, base_dir=run_dir)

    if not final.get("nodes") or not final.get("edges"):
        print("[WARN] final diagram is empty after improve; rollback to draft before verify")
        final = draft

    # Verify critique applications
    if draft_is_error:
        verify = {
            "items": [],
            "summary": {"fixed": 0, "partial": 0, "ignored": 0},
            "invalid_final": True,
            "reason": "verify skipped: draft is error fallback",
        }
        save_json_artifact("verify.json", verify, base_dir=run_dir)
    else:
        verify = verify_critique_application(user_task, draft, patch_plan, final)
        save_json_artifact("verify.json", verify, base_dir=run_dir)

        if (
            not verify.get("invalid_final", False)
            and (verify.get("summary", {}).get("ignored", 0) > 0 or verify.get("summary", {}).get("partial", 0) > 0)
        ):
            print("\n=== ШАГ 3.5: Дополнительный improve-pass по проигнорированным пунктам ===")
            followup_patch_plan = build_followup_patch_plan(patch_plan, verify)
            save_json_artifact("followup_patch_plan.json", followup_patch_plan, base_dir=run_dir)

            if followup_patch_plan.get("must_fix"):
                final_retry, improve_meta_retry = improve_diagram(
                    user_task,
                    final,
                    critique,
                    references,
                    patch_plan=followup_patch_plan,
                    normalize_general_diagram_fn=normalize_general_diagram,
                )
                verify_retry = verify_critique_application(user_task, draft, patch_plan, final_retry)

                save_json_artifact("improve_meta_retry.json", improve_meta_retry, base_dir=run_dir)
                save_json_artifact("verify_retry.json", verify_retry, base_dir=run_dir)

                old_fixed = verify.get("summary", {}).get("fixed", 0)
                new_fixed = verify_retry.get("summary", {}).get("fixed", 0)
                old_ignored = verify.get("summary", {}).get("ignored", 0)
                new_ignored = verify_retry.get("summary", {}).get("ignored", 0)

                if (new_fixed > old_fixed) or (new_ignored < old_ignored):
                    final = final_retry
                    verify = verify_retry

    final_clean = clean_diagram_labels(final)
    save_json_artifact("draft.json", draft, base_dir=run_dir)
    save_json_artifact("final.json", final_clean, base_dir=run_dir)

    if critic_ab_replay:
        print("\n=== ШАГ 3.5: A/B replay (с критиком vs без критика) ===")
        neutral_critique = {
            "score": 1.0, "task_fit_score": 1.0, "visual_score": 1.0,
            "missing_requirements": [], "wrong_interpretations": [], "extra_elements": [],
            "visual_problems": [], "problems": [], "fixes": [],
        }
        try:
            counterfactual_final, _ = improve_diagram(
                user_task,
                draft,
                neutral_critique,
                references,
                patch_plan=build_patch_plan(neutral_critique),
                normalize_general_diagram_fn=normalize_general_diagram,
            )
            counterfactual_clean = clean_diagram_labels(counterfactual_final)
            save_json_artifact("counterfactual_final_no_critic.json", counterfactual_clean, base_dir=run_dir)

            factual_change = compute_change_metrics(draft, final_clean)
            counter_change = compute_change_metrics(draft, counterfactual_clean)
            factual_listen = compute_critic_listening_metrics(draft, critique, final_clean, verify=verify)
            counter_listen = compute_critic_listening_metrics(draft, critique, counterfactual_clean)

            ab_report = {
                "run_id": output_filename,
                "factual_change": factual_change,
                "counterfactual_change_no_critic": counter_change,
                "factual_listening": factual_listen,
                "counterfactual_listening_no_critic": counter_listen,
                "critic_effect_delta": {
                    "alignment_gain": float(factual_listen.get("critic_alignment_score", 0.0) - counter_listen.get("critic_alignment_score", 0.0)),
                    "ignored_rate_reduction": float(counter_listen.get("critic_ignored_rate", 0.0) - factual_listen.get("critic_ignored_rate", 0.0)),
                    "change_score_delta": float(factual_change.get("change_score", 0.0) - counter_change.get("change_score", 0.0)),
                },
                "assumption": "Counterfactual uses same draft and model, but neutral critique.",
            }
            save_json_artifact("critic_ab_replay_report.json", ab_report, base_dir=run_dir)
            print(f"[critic-ab] report={run_dir / 'critic_ab_replay_report.json'}")
        except Exception as exc:
            save_json_artifact(
                "critic_ab_replay_report.json",
                {"run_id": output_filename, "status": "error_fallback", "reason": str(exc)},
                base_dir=run_dir,
            )
            print(f"[critic-ab] fallback due to error: {exc}")

    if explain_critic_influence:
        print("\n=== ШАГ 4: Анализ влияния критика ===")
        try:
            analyzer = CriticInfluenceAnalyzer(output_root="outputs")
            influence_result = analyzer.analyze_and_save(
                run_id=output_filename,
                run_dir=run_dir,
                draft=draft,
                critique=critique,
                final=final_clean,
                verify=verify,
            )
            print(f"[critic-influence] status={influence_result.status}, report={run_dir / 'critic_influence_report.json'}")
        except Exception as exc:
            fallback_report = {"status": "error_fallback", "reason": str(exc), "run_id": output_filename}
            save_json_artifact("critic_influence_report.json", fallback_report, base_dir=run_dir)
            print(f"[critic-influence] fallback due to error: {exc}")
    else:
        print("\n=== ШАГ 4: Анализ влияния критика (OFF) ===")

    print("\n=== ШАГ 5: Рендер ===")
    render_diagram(final_clean, run_id=output_filename, output_dir=run_dir)


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Diagram Agent")
    parser.add_argument(
        "--explain-critic-influence",
        choices=["on", "off"],
        default="on",
        help="Enable critic influence analytics and SHAP artifacts (default: on).",
    )
    parser.add_argument(
        "--no-auto-servers",
        action="store_true",
        help="Do not auto-start local model servers via ServerManager.",
    )
    parser.add_argument(
        "--ignore-test-prompt",
        action="store_true",
        help="Ignore _test_prompt.txt and use default prompt from main.",
    )
    parser.add_argument(
        "--critic-ab-replay",
        choices=["on", "off"],
        default="off",
        help="Run counterfactual improve pass with neutral critique and save A/B effect report.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()
    explain = args.explain_critic_influence == "on"
    use_test_prompt = not args.ignore_test_prompt
    ab_replay = args.critic_ab_replay == "on"

    if args.no_auto_servers:
        run_orchestration(
            explain_critic_influence=explain,
            use_test_prompt=use_test_prompt,
            critic_ab_replay=ab_replay,
        )
    else:
        with ServerManager() as manager:
            manager.start_default_servers()
            run_orchestration(
                explain_critic_influence=explain,
                use_test_prompt=use_test_prompt,
                critic_ab_replay=ab_replay,
            )
            manager.stop_all()
