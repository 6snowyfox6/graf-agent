from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import gradio as gr

from app_service import run_pipeline_stream


OUTPUTS_DIR = Path("outputs")
PREFERRED_PREVIEW_NAMES = [
    "final_diagram.png",
    "final.png",
    "diagram.png",
    "preview.png",
    "result.png",
    "final_diagram.jpg",
    "final.jpg",
    "diagram.jpg",
    "final_diagram.jpeg",
    "final.jpeg",
    "diagram.jpeg",
    "final_diagram.webp",
    "final.webp",
    "diagram.webp",
    "final_diagram.svg",
    "final.svg",
    "diagram.svg",
]
IMAGE_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.svg")
PREVIEW_EXCLUDE_MARKERS = (
    "shap",
    "critic",
    "critique",
    "draft",
    "raw",
    "influence",
    "waterfall",
    "bar_",
    "beeswarm",
)
SHAP_INCLUDE_MARKERS = (
    "shap",
    "waterfall",
    "beeswarm",
    "bar_",
    "critic_influence",
)

CSS = r'''
:root,
html.light,
body.light,
.light,
html[data-theme="light"],
body[data-theme="light"],
[data-theme="light"] {
    --bg: #eef3fb;
    --bg-2: #f7f9fc;
    --surface: rgba(255, 255, 255, 0.96);
    --surface-2: rgba(246, 248, 252, 0.98);
    --surface-soft: rgba(243, 246, 251, 0.96);
    --border: rgba(15, 23, 42, 0.10);
    --dash-border: rgba(15, 23, 42, 0.12);
    --text: #1d2738;
    --muted: rgba(29, 39, 56, 0.62);
    --shadow: 0 20px 48px rgba(18, 31, 53, 0.12);
    --overlay-bg: rgba(17, 24, 39, 0.30);
    --button-secondary-bg: rgba(17, 24, 39, 0.06);
    --button-secondary-hover: rgba(17, 24, 39, 0.10);
    --button-secondary-border: rgba(15, 23, 42, 0.10);
    --button-secondary-text: #1d2738;
    --input-placeholder: rgba(29, 39, 56, 0.42);
    --accent: #5b5cf0;
    --accent-soft: rgba(91, 92, 240, 0.14);
}

html.dark,
body.dark,
.dark,
html[data-theme="dark"],
body[data-theme="dark"],
[data-theme="dark"] {
    --bg: #071120;
    --bg-2: #0a1730;
    --surface: rgba(26, 35, 52, 0.97);
    --surface-2: rgba(22, 31, 48, 0.98);
    --surface-soft: rgba(255, 255, 255, 0.02);
    --border: rgba(255, 255, 255, 0.08);
    --dash-border: rgba(255, 255, 255, 0.10);
    --text: #e8ecf5;
    --muted: rgba(232, 236, 245, 0.68);
    --shadow: 0 22px 50px rgba(0, 0, 0, 0.30);
    --overlay-bg: rgba(4, 10, 20, 0.72);
    --button-secondary-bg: rgba(255, 255, 255, 0.10);
    --button-secondary-hover: rgba(255, 255, 255, 0.16);
    --button-secondary-border: rgba(255, 255, 255, 0.10);
    --button-secondary-text: #e8ecf5;
    --input-placeholder: rgba(232, 236, 245, 0.45);
    --accent: #5b5cf0;
    --accent-soft: rgba(91, 92, 240, 0.16);
}

html,
body {
    min-width: 1360px;
    background: linear-gradient(180deg, var(--bg-2) 0%, var(--bg) 100%) !important;
    overflow-x: auto;
}

body::before {
    content: "";
    position: fixed;
    inset: 0;
    background: linear-gradient(180deg, var(--bg-2) 0%, var(--bg) 100%);
    z-index: -2;
}

html.dark body,
body.dark,
.dark body,
html[data-theme="dark"] body,
body[data-theme="dark"] {
    color-scheme: dark;
}

html.light body,
body.light,
.light body,
html[data-theme="light"] body,
body[data-theme="light"] {
    color-scheme: light;
}

.gradio-container {
    --background-fill-primary: transparent !important;
    --background-fill-secondary: transparent !important;
    --body-background-fill: transparent !important;
    --body-text-color: var(--text) !important;
    --body-text-color-subdued: var(--muted) !important;
    --color-accent: var(--accent) !important;
    --color-accent-soft: var(--accent-soft) !important;
    --border-color-primary: var(--border) !important;
    --block-background-fill: var(--surface) !important;
    --block-border-color: var(--border) !important;
    --block-label-text-color: var(--text) !important;
    --block-title-text-color: var(--text) !important;
    --input-background-fill: var(--surface-2) !important;
    --input-border-color: var(--border) !important;
    --input-placeholder-color: var(--input-placeholder) !important;
    --checkbox-label-text-color: var(--text) !important;
    --button-secondary-background-fill: var(--button-secondary-bg) !important;
    --button-secondary-background-fill-hover: var(--button-secondary-hover) !important;
    --button-secondary-border-color: var(--button-secondary-border) !important;
    --button-secondary-text-color: var(--button-secondary-text) !important;
    width: 100% !important;
    max-width: none !important;
    margin: 0 !important;
    padding: 20px 24px 24px !important;
    color: var(--text) !important;
}

.gradio-container,
.gradio-container .wrap,
.gradio-container .prose,
.gradio-container label,
.gradio-container span,
.gradio-container p,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4,
.gradio-container h5 {
    color: var(--text) !important;
}

.gradio-container textarea,
.gradio-container input,
.gradio-container select {
    background: var(--surface-2) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}

.gradio-container textarea::placeholder,
.gradio-container input::placeholder {
    color: var(--input-placeholder) !important;
}

.gradio-container button.secondary {
    background: var(--button-secondary-bg) !important;
    color: var(--button-secondary-text) !important;
    border-color: var(--button-secondary-border) !important;
}

.gradio-container button.secondary:hover {
    background: var(--button-secondary-hover) !important;
}

.gradio-container [data-testid="image"],
.gradio-container [data-testid="image"] > div,
.gradio-container .upload-container,
.gradio-container .empty {
    background: var(--surface-2) !important;
    color: var(--text) !important;
}

#app-shell {
    display: grid !important;
    grid-template-columns: 460px minmax(860px, 1fr);
    gap: 20px !important;
    align-items: stretch !important;
}

#left-panel,
#right-panel,
.card,
#history-modal,
#shap-modal {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 22px;
    box-shadow: var(--shadow);
}

#left-panel {
    width: 460px !important;
    min-width: 460px !important;
    max-width: 460px !important;
    padding: 14px !important;
    gap: 14px !important;
}

#right-panel {
    min-width: 0 !important;
    padding: 14px !important;
}

.card {
    padding: 14px !important;
}

.card-title {
    margin: 0 0 10px 0 !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    color: var(--text) !important;
}

.card-title p {
    margin: 0 !important;
}

#prompt-box textarea {
    min-height: 430px !important;
    max-height: 430px !important;
    resize: none !important;
    font-size: 16px !important;
    line-height: 1.45 !important;
}

#reference-box,
#reference-box .image-container,
#reference-box img {
    height: 120px !important;
    max-height: 120px !important;
    min-height: 120px !important;
    object-fit: contain !important;
}

#reference-box .wrap,
#reference-box [data-testid="image"] > div {
    min-height: 120px !important;
    height: 120px !important;
}

#reference-box .center,
#reference-box [data-testid="image"] > div,
#reference-box [data-testid="image"] label,
#reference-box .upload-container,
#reference-box .empty {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    flex-direction: column !important;
}

#reference-box .icon-wrap,
#reference-box svg {
    margin: 0 auto 8px auto !important;
    width: 28px !important;
    height: 28px !important;
}

#reference-box p,
#reference-box span,
#reference-box label,
#reference-box .wrap,
#reference-box .empty,
#reference-box .upload-container {
    font-size: 14px !important;
    line-height: 1.2 !important;
    white-space: normal !important;
    word-break: break-word !important;
}

#generate-btn button,
#history-btn button,
#shap-btn button,
#history-close button,
#shap-close button {
    min-height: 52px !important;
    border-radius: 16px !important;
    font-size: 17px !important;
    font-weight: 700 !important;
}

#preview-card {
    min-height: 820px !important;
}

#preview-placeholder {
    height: 760px;
    border: 1px dashed var(--dash-border);
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--muted);
    font-size: 18px;
    text-align: center;
    background: var(--surface-soft);
}

#preview-box,
#preview-box .image-container,
#preview-box img,
#preview-box svg {
    height: 760px !important;
    max-height: 760px !important;
    min-height: 760px !important;
    object-fit: contain !important;
}

#history-overlay.hidden,
#shap-overlay.hidden {
    display: none !important;
}

#history-overlay,
#shap-overlay {
    position: fixed !important;
    inset: 0 !important;
    z-index: 9999 !important;
    background: var(--overlay-bg) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 30px !important;
}

#history-modal {
    width: min(1200px, calc(100vw - 70px));
    min-height: 720px;
    max-height: calc(100vh - 60px);
    padding: 18px !important;
    overflow: hidden !important;
}

#shap-modal {
    width: min(1320px, calc(100vw - 70px));
    min-height: 820px;
    max-height: calc(100vh - 60px);
    padding: 18px !important;
    overflow: hidden !important;
}

#history-topbar,
#shap-topbar {
    align-items: center !important;
    margin-bottom: 10px !important;
}

#history-count,
#shap-status {
    color: var(--muted) !important;
    font-size: 14px !important;
}

#history-gallery {
    min-height: 590px !important;
}

#history-gallery .grid-wrap {
    max-height: 590px !important;
    overflow-y: auto !important;
    padding-right: 6px !important;
}

#history-gallery img,
#history-gallery svg,
#history-gallery .thumbnail-item {
    border-radius: 14px !important;
}

#history-help,
#shap-help {
    color: var(--muted) !important;
    font-size: 14px !important;
    margin-top: 6px !important;
}

#shap-placeholder {
    height: 700px;
    border: 1px dashed var(--dash-border);
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--muted);
    font-size: 18px;
    text-align: center;
    background: var(--surface-soft);
}

#shap-box,
#shap-box .image-container,
#shap-box img,
#shap-box svg {
    height: 700px !important;
    max-height: 700px !important;
    min-height: 700px !important;
    object-fit: contain !important;
}

@media (max-width: 1360px) {
    html, body {
        min-width: 1240px;
    }
    #app-shell {
        grid-template-columns: 430px minmax(780px, 1fr);
    }
    #left-panel {
        width: 430px !important;
        min-width: 430px !important;
        max-width: 430px !important;
    }
}

'''


def _normalize_name(path: Path) -> str:
    return path.name.lower()


def _is_regular_preview_file(path: Path) -> bool:
    name = _normalize_name(path)
    return not any(marker in name for marker in PREVIEW_EXCLUDE_MARKERS)


def _is_shap_file(path: Path) -> bool:
    name = _normalize_name(path)
    return any(marker in name for marker in SHAP_INCLUDE_MARKERS)


def _iter_image_files(run_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in IMAGE_PATTERNS:
        files.extend(run_dir.glob(pattern))
    return sorted({p.resolve() for p in files})


def _find_preview_in_dir(run_dir: Path) -> str | None:
    for name in PREFERRED_PREVIEW_NAMES:
        candidate = run_dir / name
        if candidate.exists() and _is_regular_preview_file(candidate):
            return str(candidate)

    regular_images = [p for p in _iter_image_files(run_dir) if _is_regular_preview_file(p)]
    if regular_images:
        return str(regular_images[0])
    return None


def _find_shap_in_dir(run_dir: Path) -> str | None:
    shap_images = [p for p in _iter_image_files(run_dir) if _is_shap_file(p)]
    if not shap_images:
        return None

    def shap_score(path: Path) -> tuple[int, str]:
        name = _normalize_name(path)
        if "summary" in name:
            priority = 0
        elif "beeswarm" in name:
            priority = 1
        elif "waterfall" in name:
            priority = 2
        elif "bar" in name:
            priority = 3
        else:
            priority = 4
        return priority, name

    return str(sorted(shap_images, key=shap_score)[0])


def _run_sort_key(run_dir: Path) -> tuple[int, float]:
    match = re.search(r"(\d+)$", run_dir.name)
    if match:
        return int(match.group(1)), run_dir.stat().st_mtime
    return 0, run_dir.stat().st_mtime


def _display_time_for_run(run_dir: Path) -> str:
    sort_ts, fallback_ts = _run_sort_key(run_dir)
    display_ts = sort_ts if sort_ts > 0 else int(fallback_ts)
    return datetime.fromtimestamp(display_ts).strftime("%Y-%m-%d %H:%M:%S")


def refresh_history_data():
    if not OUTPUTS_DIR.exists():
        return [], [], "История пуста.", "Найдено диаграмм: 0"

    run_dirs = sorted(
        [p for p in OUTPUTS_DIR.iterdir() if p.is_dir()],
        key=_run_sort_key,
        reverse=True,
    )

    gallery_items: list[tuple[str, str]] = []
    preview_paths: list[str] = []
    for run_dir in run_dirs:
        preview = _find_preview_in_dir(run_dir)
        if not preview:
            continue
        caption = f"{run_dir.name}\n{_display_time_for_run(run_dir)}"
        gallery_items.append((preview, caption))
        preview_paths.append(preview)

    if not gallery_items:
        return [], [], "История пуста.", "Найдено диаграмм: 0"

    return (
        gallery_items,
        preview_paths,
        "Нажми на миниатюру, чтобы открыть её в основном окне.",
        f"Найдено диаграмм: {len(gallery_items)} · сверху самые свежие",
    )


def _preview_updates(preview_path: str | None):
    has_preview = bool(preview_path)
    placeholder = gr.update(
        visible=not has_preview,
        value="<div id='preview-placeholder'>Здесь появится итоговая диаграмма</div>",
    )
    image = gr.update(visible=has_preview, value=preview_path if has_preview else None)
    return placeholder, image


def _shap_updates(shap_path: str | None, preview_path: str | None = None):
    if shap_path:
        return (
            gr.update(visible=False, value="<div id='shap-placeholder'>Для этой схемы SHAP не найден</div>"),
            gr.update(visible=True, value=shap_path),
            f"SHAP для: `{Path(preview_path).parent.name if preview_path else Path(shap_path).parent.name}`",
            "Открыт найденный SHAP-график.",
        )
    return (
        gr.update(visible=True, value="<div id='shap-placeholder'>Для этой схемы SHAP не найден</div>"),
        gr.update(visible=False, value=None),
        f"SHAP для: `{Path(preview_path).parent.name}`" if preview_path else "SHAP недоступен",
        "Для выбранной диаграммы SHAP-файл не найден.",
    )


def _path_to_run_dir(preview_path: str | None) -> Path | None:
    if not preview_path:
        return None
    path = Path(preview_path)
    return path.parent if path.exists() else path.parent


def _sync_selected_assets(preview_path: str | None):
    run_dir = _path_to_run_dir(preview_path)
    shap_path = _find_shap_in_dir(run_dir) if run_dir else None
    return preview_path, shap_path


def _resolve_preview_path(event_preview: str | None, run_dir_value: str | None) -> str | None:
    if isinstance(event_preview, str) and event_preview.strip():
        candidate = Path(event_preview)
        if candidate.exists():
            return str(candidate)
        return event_preview

    if run_dir_value:
        run_dir = Path(run_dir_value)
        if run_dir.exists():
            return _find_preview_in_dir(run_dir)
    return None


def stream_to_ui(
    user_prompt: str,
    reference_image_path: str | None,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    prompt = (user_prompt or "").strip()
    if not prompt:
        yield *_preview_updates(None), None, None, gr.update(interactive=False)
        return

    current_preview: str | None = None
    yield *_preview_updates(None), None, None, gr.update(interactive=False)

    progress(0.01, desc="Подготовка...")

    for event in run_pipeline_stream(
        user_prompt=prompt,
        reference_image_path=reference_image_path,
        explain_critic_influence=True,
        auto_start_servers=True,
    ):
        message = str(event.get("message", ""))
        raw_progress = float(event.get("progress", 0.0))

        preview_path = _resolve_preview_path(event.get("preview_path"), event.get("run_dir"))
        if isinstance(preview_path, str) and preview_path.strip():
            current_preview = preview_path

        stage = str(event.get("stage", ""))
        if stage == "error" and current_preview:
            progress(1.0, desc="Готово")
            _, shap_path = _sync_selected_assets(current_preview)
            yield *_preview_updates(current_preview), current_preview, shap_path, gr.update(interactive=bool(shap_path))
            return

        progress(max(0.01, min(1.0, raw_progress)), desc=message or "Генерация...")
        _, shap_path = _sync_selected_assets(current_preview)
        yield *_preview_updates(current_preview), current_preview, shap_path, gr.update(interactive=bool(shap_path))

    _, shap_path = _sync_selected_assets(current_preview)
    progress(1.0 if current_preview else 0.01, desc="Готово" if current_preview else "")
    yield *_preview_updates(current_preview), current_preview, shap_path, gr.update(interactive=bool(shap_path))


def show_history():
    return gr.update(visible=True, elem_classes=[])


def hide_history():
    return gr.update(visible=False, elem_classes=["hidden"])


def prepare_shap_popup(current_preview: str | None, current_shap: str | None):
    shap_path = current_shap
    if current_preview:
        _, resolved_shap = _sync_selected_assets(current_preview)
        shap_path = resolved_shap
    placeholder_update, image_update, status, help_text = _shap_updates(shap_path, current_preview)
    return (
        placeholder_update,
        image_update,
        status,
        help_text,
        shap_path,
    )


def show_shap_overlay():
    return gr.update(visible=True, elem_classes=[])


def hide_shap_popup():
    return gr.update(visible=False, elem_classes=["hidden"])


def choose_history_item(paths: list[str], evt: gr.SelectData):
    index = evt.index if evt else None
    if index is None or not isinstance(index, int) or index < 0 or index >= len(paths):
        return _preview_updates(None) + (None, None, gr.update(interactive=False))
    selected = paths[index]
    _, shap_path = _sync_selected_assets(selected)
    return _preview_updates(selected) + (selected, shap_path, gr.update(interactive=bool(shap_path)))


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="graf-agent UI") as demo:
        history_paths = gr.State([])
        current_preview_state = gr.State(None)
        current_shap_state = gr.State(None)

        with gr.Row(elem_id="app-shell", equal_height=True):
            with gr.Column(elem_id="left-panel"):
                with gr.Column(elem_classes=["card"]):
                    gr.Markdown("Промпт", elem_classes=["card-title"])
                    prompt = gr.Textbox(
                        show_label=False,
                        lines=16,
                        placeholder="Например: Нарисуй общую архитектурную диаграмму системы онлайн-обучения...",
                        autofocus=True,
                        elem_id="prompt-box",
                    )

                with gr.Column(elem_classes=["card"]):
                    gr.Markdown("Референс", elem_classes=["card-title"])
                    reference = gr.Image(
                        show_label=False,
                        type="filepath",
                        sources=["upload"],
                        height=120,
                        elem_id="reference-box",
                    )

                with gr.Row():
                    run_btn = gr.Button("Сгенерировать", variant="primary", elem_id="generate-btn")
                    history_btn = gr.Button("История", variant="secondary", elem_id="history-btn")

                shap_btn = gr.Button(
                    "Показать SHAP",
                    variant="secondary",
                    interactive=False,
                    elem_id="shap-btn",
                )

            with gr.Column(elem_id="right-panel"):
                with gr.Column(elem_id="preview-card", elem_classes=["card"]):
                    gr.Markdown("Предпросмотр результата", elem_classes=["card-title"])
                    preview_placeholder = gr.HTML(
                        "<div id='preview-placeholder'>Здесь появится итоговая диаграмма</div>",
                        visible=True,
                    )
                    preview = gr.Image(
                        show_label=False,
                        type="filepath",
                        interactive=False,
                        visible=False,
                        height=760,
                        buttons=["download", "fullscreen"],
                        elem_id="preview-box",
                    )

        with gr.Column(elem_id="history-overlay", elem_classes=["hidden"], visible=False) as history_overlay:
            with gr.Column(elem_id="history-modal"):
                with gr.Row(elem_id="history-topbar"):
                    gr.Markdown("История запусков", elem_classes=["card-title"])
                    history_count = gr.Markdown("", elem_id="history-count")
                    history_close = gr.Button("Закрыть", elem_id="history-close")
                history_gallery = gr.Gallery(
                    label=None,
                    show_label=False,
                    value=[],
                    columns=4,
                    rows=2,
                    height=590,
                    allow_preview=False,
                    elem_id="history-gallery",
                )
                history_help = gr.Markdown("", elem_id="history-help")

        with gr.Column(elem_id="shap-overlay", elem_classes=["hidden"], visible=False) as shap_overlay:
            with gr.Column(elem_id="shap-modal"):
                with gr.Row(elem_id="shap-topbar"):
                    gr.Markdown("SHAP", elem_classes=["card-title"])
                    shap_status = gr.Markdown("", elem_id="shap-status")
                    shap_close = gr.Button("Закрыть", elem_id="shap-close")
                shap_placeholder = gr.HTML(
                    "<div id='shap-placeholder'>Для этой схемы SHAP не найден</div>",
                    visible=True,
                )
                shap_image = gr.Image(
                    show_label=False,
                    type="filepath",
                    interactive=False,
                    visible=False,
                    height=700,
                    buttons=["download", "fullscreen"],
                    elem_id="shap-box",
                )
                shap_help = gr.Markdown("", elem_id="shap-help")

        run_evt = run_btn.click(
            fn=stream_to_ui,
            inputs=[prompt, reference],
            outputs=[preview_placeholder, preview, current_preview_state, current_shap_state, shap_btn],
            show_progress="full",
        )
        run_evt.then(
            fn=refresh_history_data,
            inputs=None,
            outputs=[history_gallery, history_paths, history_help, history_count],
        )

        history_evt = history_btn.click(
            fn=refresh_history_data,
            inputs=None,
            outputs=[history_gallery, history_paths, history_help, history_count],
            show_progress="hidden",
        )
        history_evt.then(
            fn=show_history,
            inputs=None,
            outputs=[history_overlay],
        )

        history_close.click(
            fn=hide_history,
            inputs=None,
            outputs=[history_overlay],
            show_progress="hidden",
        )

        select_evt = history_gallery.select(
            fn=choose_history_item,
            inputs=[history_paths],
            outputs=[preview_placeholder, preview, current_preview_state, current_shap_state, shap_btn],
            show_progress="hidden",
        )
        select_evt.then(
            fn=hide_history,
            inputs=None,
            outputs=[history_overlay],
        )

        shap_evt = shap_btn.click(
            fn=prepare_shap_popup,
            inputs=[current_preview_state, current_shap_state],
            outputs=[shap_placeholder, shap_image, shap_status, shap_help, current_shap_state],
            show_progress="hidden",
        )
        shap_evt.then(
            fn=show_shap_overlay,
            inputs=None,
            outputs=[shap_overlay],
        )

        shap_close.click(
            fn=hide_shap_popup,
            inputs=None,
            outputs=[shap_overlay],
            show_progress="hidden",
        )

        demo.load(
            fn=refresh_history_data,
            inputs=None,
            outputs=[history_gallery, history_paths, history_help, history_count],
        )

        demo.queue(default_concurrency_limit=1)

    return demo


if __name__ == "__main__":
    app = build_demo()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        theme=gr.themes.Soft(),
        css=CSS,
    )
