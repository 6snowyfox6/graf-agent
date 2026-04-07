from __future__ import annotations

import contextlib
import json
import time
import traceback
from pathlib import Path
from typing import Any, Generator

from critic_influence import CriticInfluenceAnalyzer
from main import (
    analyze_reference_image,
    clean_diagram_labels,
    critique_diagram,
    generate_diagram,
    improve_diagram,
    render_diagram,
    save_json_artifact,
)
from server_manager import ServerManager


STAGE_ORDER = [
    "prepare",
    "servers",
    "reference",
    "draft",
    "critique",
    "improve",
    "influence",
    "render",
    "done",
]

STAGE_TITLES = {
    "prepare": "Подготовка",
    "servers": "Запуск серверов",
    "reference": "Анализ референса",
    "draft": "Генерация черновика",
    "critique": "Критика",
    "improve": "Исправление",
    "influence": "Анализ влияния критика",
    "render": "Рендер",
    "done": "Готово",
    "error": "Ошибка",
}


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _append_log(log_lines: list[str], message: str) -> str:
    line = f"[{_now()}] {message}"
    log_lines.append(line)
    return "\n".join(log_lines)


def _collect_artifacts(run_dir: Path, output_name: str, render_result: str | Path | None = None) -> tuple[str | None, list[str]]:
    preview_candidates: list[Path] = []
    if render_result:
        preview_candidates.append(Path(render_result))

    preview_candidates.extend(
        [
            run_dir / f"{output_name}.png",
            run_dir / "final_diagram.png",
            run_dir / f"{output_name}.jpg",
            run_dir / f"{output_name}.jpeg",
        ]
    )

    preview_path: str | None = None
    for candidate in preview_candidates:
        if candidate.exists() and candidate.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            preview_path = str(candidate)
            break

    preferred_names = [
        "draft.json",
        "critique.json",
        "final.json",
        "reference_analysis.json",
        "critic_influence_report.json",
        "critic_influence_summary.md",
        f"{output_name}.png",
        f"{output_name}.svg",
        f"{output_name}.pdf",
        "final_diagram.png",
        "final_diagram.svg",
        "final_diagram.pdf",
        "error.txt",
    ]

    files: list[Path] = []
    seen: set[str] = set()

    for name in preferred_names:
        path = run_dir / name
        if path.exists() and path.is_file() and str(path) not in seen:
            files.append(path)
            seen.add(str(path))

    for path in sorted(run_dir.iterdir()):
        if path.is_file() and str(path) not in seen:
            files.append(path)
            seen.add(str(path))

    return preview_path, [str(p) for p in files]


def _make_event(
    *,
    stage: str,
    progress: float,
    message: str,
    log_text: str,
    run_dir: Path,
    preview_path: str | None = None,
    files: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "stage": stage,
        "stage_title": STAGE_TITLES.get(stage, stage),
        "progress": max(0.0, min(1.0, float(progress))),
        "message": message,
        "log_text": log_text,
        "run_dir": str(run_dir),
        "preview_path": preview_path,
        "files": files or [],
    }
    if extra:
        payload.update(extra)
    return payload


def run_pipeline_stream(
    user_prompt: str,
    reference_image_path: str | None = None,
    *,
    explain_critic_influence: bool = True,
    auto_start_servers: bool = True,
) -> Generator[dict[str, Any], None, None]:
    prompt = _safe_text(user_prompt)
    if not prompt:
        raise ValueError("Промпт пустой. Введи описание диаграммы.")

    run_id = f"diagram_{int(time.time())}"
    run_dir = Path("outputs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    preview_path: str | None = None
    files: list[str] = []

    log_text = _append_log(log_lines, f"Создан каталог запуска: {run_dir}")
    yield _make_event(
        stage="prepare",
        progress=0.03,
        message="Подготавливаю запуск.",
        log_text=log_text,
        run_dir=run_dir,
    )

    manager: ServerManager | None = None
    references: list[dict[str, Any]] = []

    try:
        with contextlib.ExitStack() as stack:
            if auto_start_servers:
                manager = stack.enter_context(ServerManager())

                log_text = _append_log(log_lines, "Запускаю сервер генератора (порт 8000).")
                yield _make_event(
                    stage="servers",
                    progress=0.08,
                    message="Запускаю сервер генератора.",
                    log_text=log_text,
                    run_dir=run_dir,
                )
                manager.start_script("scripts/start_server.sh", port=8000, startup_timeout=180)

                log_text = _append_log(log_lines, "Сервер генератора готов. Запускаю сервер критика/vision (порт 8001).")
                yield _make_event(
                    stage="servers",
                    progress=0.14,
                    message="Запускаю сервер критика и vision-модели.",
                    log_text=log_text,
                    run_dir=run_dir,
                )
                manager.start_script("scripts/start_server_gemma.sh", port=8001, startup_timeout=180)

                log_text = _append_log(log_lines, "Оба локальных сервера готовы.")
                yield _make_event(
                    stage="servers",
                    progress=0.18,
                    message="Серверы готовы.",
                    log_text=log_text,
                    run_dir=run_dir,
                )
            else:
                log_text = _append_log(log_lines, "Автозапуск серверов отключён. Использую уже поднятые endpoints.")
                yield _make_event(
                    stage="servers",
                    progress=0.08,
                    message="Автозапуск серверов отключён.",
                    log_text=log_text,
                    run_dir=run_dir,
                )

            if reference_image_path:
                log_text = _append_log(log_lines, f"Анализирую PNG-референс: {reference_image_path}")
                yield _make_event(
                    stage="reference",
                    progress=0.24,
                    message="Анализирую референс.",
                    log_text=log_text,
                    run_dir=run_dir,
                )
                try:
                    reference_analysis = analyze_reference_image(reference_image_path)
                    save_json_artifact("reference_analysis.json", reference_analysis, base_dir=run_dir)
                    references = [reference_analysis]
                    log_text = _append_log(log_lines, "Референс успешно проанализирован.")
                except Exception as exc:  # noqa: BLE001
                    log_text = _append_log(log_lines, f"Не удалось проанализировать референс: {exc}. Продолжаю без него.")
                yield _make_event(
                    stage="reference",
                    progress=0.28,
                    message="Анализ референса завершён.",
                    log_text=log_text,
                    run_dir=run_dir,
                )

            log_text = _append_log(log_lines, "Шаг 1/4: генерирую черновик диаграммы.")
            yield _make_event(
                stage="draft",
                progress=0.34,
                message="Генерация черновика.",
                log_text=log_text,
                run_dir=run_dir,
            )
            draft = generate_diagram(prompt, references)
            save_json_artifact("draft.json", draft, base_dir=run_dir)
            log_text = _append_log(
                log_lines,
                f"Черновик готов: {len(draft.get('nodes', []))} узлов, {len(draft.get('edges', []))} связей.",
            )
            yield _make_event(
                stage="draft",
                progress=0.48,
                message="Черновик сгенерирован.",
                log_text=log_text,
                run_dir=run_dir,
            )

            log_text = _append_log(log_lines, "Шаг 2/4: запускаю критику схемы.")
            yield _make_event(
                stage="critique",
                progress=0.56,
                message="Критика схемы.",
                log_text=log_text,
                run_dir=run_dir,
            )
            critique = critique_diagram(prompt, draft, references)
            save_json_artifact("critique.json", critique, base_dir=run_dir)
            log_text = _append_log(log_lines, "Критика завершена и сохранена.")
            yield _make_event(
                stage="critique",
                progress=0.66,
                message="Критика завершена.",
                log_text=log_text,
                run_dir=run_dir,
            )

            log_text = _append_log(log_lines, "Шаг 3/4: исправляю диаграмму по замечаниям критика.")
            yield _make_event(
                stage="improve",
                progress=0.72,
                message="Исправление диаграммы.",
                log_text=log_text,
                run_dir=run_dir,
            )
            final = improve_diagram(prompt, draft, critique, references)
            final_clean = clean_diagram_labels(final)
            save_json_artifact("final.json", final_clean, base_dir=run_dir)
            log_text = _append_log(log_lines, "Исправленная диаграмма сохранена.")
            yield _make_event(
                stage="improve",
                progress=0.80,
                message="Исправление завершено.",
                log_text=log_text,
                run_dir=run_dir,
            )

            if explain_critic_influence:
                log_text = _append_log(log_lines, "Шаг 4/4: анализирую влияние критика.")
                yield _make_event(
                    stage="influence",
                    progress=0.85,
                    message="Анализ влияния критика.",
                    log_text=log_text,
                    run_dir=run_dir,
                )
                try:
                    analyzer = CriticInfluenceAnalyzer(output_root="outputs")
                    influence_result = analyzer.analyze_and_save(
                        run_id=run_id,
                        run_dir=run_dir,
                        draft=draft,
                        critique=critique,
                        final=final_clean,
                    )
                    log_text = _append_log(
                        log_lines,
                        f"Анализ влияния критика завершён: status={getattr(influence_result, 'status', 'unknown')}.",
                    )
                except Exception as exc:  # noqa: BLE001
                    fallback_report = {
                        "status": "error_fallback",
                        "reason": str(exc),
                        "run_id": run_id,
                    }
                    save_json_artifact("critic_influence_report.json", fallback_report, base_dir=run_dir)
                    (run_dir / "critic_influence_summary.md").write_text(
                        "# Critic Influence Summary\n\n"
                        "status: `error_fallback`\n\n"
                        f"reason: `{exc}`\n",
                        encoding="utf-8",
                    )
                    log_text = _append_log(log_lines, f"Анализ влияния критика завершился fallback-режимом: {exc}")
                yield _make_event(
                    stage="influence",
                    progress=0.89,
                    message="Анализ влияния критика завершён.",
                    log_text=log_text,
                    run_dir=run_dir,
                )
            else:
                log_text = _append_log(log_lines, "Анализ влияния критика отключён.")
                yield _make_event(
                    stage="influence",
                    progress=0.84,
                    message="Анализ влияния критика отключён.",
                    log_text=log_text,
                    run_dir=run_dir,
                )

            log_text = _append_log(log_lines, "Запускаю рендер итоговой диаграммы.")
            yield _make_event(
                stage="render",
                progress=0.93,
                message="Рендер диаграммы.",
                log_text=log_text,
                run_dir=run_dir,
            )
            render_result = render_diagram(final_clean, run_id, output_dir=run_dir)
            preview_path, files = _collect_artifacts(run_dir, run_id, render_result)
            log_text = _append_log(log_lines, "Рендер завершён. Артефакты готовы к скачиванию.")
            yield _make_event(
                stage="done",
                progress=1.0,
                message="Готово.",
                log_text=log_text,
                run_dir=run_dir,
                preview_path=preview_path,
                files=files,
            )

    except Exception as exc:  # noqa: BLE001
        error_text = "\n".join(
            [
                f"Ошибка: {exc}",
                "",
                traceback.format_exc(),
            ]
        )
        (run_dir / "error.txt").write_text(error_text, encoding="utf-8")
        log_text = _append_log(log_lines, f"Пайплайн завершился с ошибкой: {exc}")
        preview_path, files = _collect_artifacts(run_dir, run_id)
        if str(run_dir / "error.txt") not in files:
            files.append(str(run_dir / "error.txt"))
        yield _make_event(
            stage="error",
            progress=1.0,
            message=str(exc),
            log_text=log_text,
            run_dir=run_dir,
            preview_path=preview_path,
            files=files,
            extra={"error": error_text},
        )
        return
