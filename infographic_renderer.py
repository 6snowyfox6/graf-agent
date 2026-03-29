from __future__ import annotations

import re
from pathlib import Path
from typing import Any


class InfographicRenderError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Цветовые схемы
# ---------------------------------------------------------------------------

COLOR_SCHEMES: dict[str, dict[str, str]] = {
    "blue": {
        "primary": "#2563EB",
        "accent": "#60A5FA",
        "bg": "#F0F4FF",
        "text": "#1E293B",
        "text_secondary": "#475569",
        "card_bg": "#FFFFFF",
        "card_border": "#BFDBFE",
        "bar_bg": "#DBEAFE",
        "bar_fill": "#2563EB",
    },
    "green": {
        "primary": "#059669",
        "accent": "#34D399",
        "bg": "#ECFDF5",
        "text": "#1E293B",
        "text_secondary": "#475569",
        "card_bg": "#FFFFFF",
        "card_border": "#A7F3D0",
        "bar_bg": "#D1FAE5",
        "bar_fill": "#059669",
    },
    "orange": {
        "primary": "#EA580C",
        "accent": "#FB923C",
        "bg": "#FFF7ED",
        "text": "#1E293B",
        "text_secondary": "#475569",
        "card_bg": "#FFFFFF",
        "card_border": "#FED7AA",
        "bar_bg": "#FFEDD5",
        "bar_fill": "#EA580C",
    },
    "purple": {
        "primary": "#7C3AED",
        "accent": "#A78BFA",
        "bg": "#F5F3FF",
        "text": "#1E293B",
        "text_secondary": "#475569",
        "card_bg": "#FFFFFF",
        "card_border": "#C4B5FD",
        "bar_bg": "#EDE9FE",
        "bar_fill": "#7C3AED",
    },
    "dark": {
        "primary": "#60A5FA",
        "accent": "#93C5FD",
        "bg": "#111827",
        "text": "#F9FAFB",
        "text_secondary": "#9CA3AF",
        "card_bg": "#1F2937",
        "card_border": "#374151",
        "bar_bg": "#374151",
        "bar_fill": "#60A5FA",
    },
}

DEFAULT_SCHEME = "blue"

# ---------------------------------------------------------------------------
# SVG-иконки (упрощённые paths)
# ---------------------------------------------------------------------------

ICONS: dict[str, str] = {
    "chart": (
        '<path d="M4 20h16M4 20V4m4 16V10m4 10V6m4 10v-4" '
        'stroke="{color}" stroke-width="2" fill="none" stroke-linecap="round"/>'
    ),
    "users": (
        '<circle cx="12" cy="8" r="4" fill="none" stroke="{color}" stroke-width="2"/>'
        '<path d="M4 20c0-4 4-6 8-6s8 2 8 6" fill="none" stroke="{color}" stroke-width="2"/>'
    ),
    "money": (
        '<path d="M12 2v20M6 6h8a4 4 0 010 8H6m0-8h10a4 4 0 010 8H6" '
        'fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round"/>'
    ),
    "time": (
        '<circle cx="12" cy="12" r="10" fill="none" stroke="{color}" stroke-width="2"/>'
        '<path d="M12 6v6l4 2" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round"/>'
    ),
    "star": (
        '<path d="M12 2l3.09 6.26L22 9.27l-5 4.87L18.18 '
        '21 12 17.27 5.82 21 7 14.14 2 9.27l6.91-1.01L12 2z" '
        'fill="{color}" stroke="none"/>'
    ),
}


# ---------------------------------------------------------------------------
# Рендерер
# ---------------------------------------------------------------------------

class InfographicRenderer:
    """
    Генерирует SVG-инфографику из JSON-описания от LLM.

    Принимает diagram с полями:
      - title, subtitle (опционально)
      - color_scheme: blue | green | orange | purple | dark
      - sections: список секций разных типов

    Типы секций: stat, text_block, comparison, steps, timeline.
    """

    CANVAS_WIDTH = 800
    PADDING = 40
    CONTENT_WIDTH = 800 - 2 * 40  # 720

    def __init__(
        self,
        output_root: str | Path | None = None,
    ) -> None:
        if output_root is None:
            output_root = Path(__file__).resolve().parent / "outputs"
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def render(self, diagram: dict[str, Any], output_name: str = "diagram") -> dict[str, str]:
        self._validate(diagram)

        safe_name = self._sanitize_filename(output_name)
        work_dir = self.output_root / safe_name
        work_dir.mkdir(parents=True, exist_ok=True)

        svg_content = self.build_svg(diagram)

        svg_path = work_dir / f"{safe_name}.svg"
        png_path = work_dir / f"{safe_name}.png"
        pdf_path = work_dir / f"{safe_name}.pdf"

        svg_path.write_text(svg_content, encoding="utf-8")
        self._convert_to_png(svg_path, png_path)
        self._convert_to_pdf(svg_path, pdf_path)

        print(f"Инфографика SVG: {svg_path}")
        print(f"Инфографика PNG: {png_path}")
        print(f"Инфографика PDF: {pdf_path}")

        return {
            "status": "ok",
            "svg_path": str(svg_path),
            "png_path": str(png_path),
            "pdf_path": str(pdf_path),
            "work_dir": str(work_dir),
        }

    # ------------------------------------------------------------------
    # SVG builder
    # ------------------------------------------------------------------

    def build_svg(self, diagram: dict[str, Any]) -> str:
        scheme_name = str(diagram.get("color_scheme", DEFAULT_SCHEME)).lower()
        colors = COLOR_SCHEMES.get(scheme_name, COLOR_SCHEMES[DEFAULT_SCHEME])

        title = diagram.get("title", "Инфографика")
        subtitle = diagram.get("subtitle", "")
        sections = diagram.get("sections", [])

        parts: list[str] = []
        y = self.PADDING

        # --- header ---
        header_svg, y = self._render_header(title, subtitle, colors, y)
        parts.append(header_svg)

        # --- sections ---
        for section in sections:
            sec_type = str(section.get("type", "")).lower()

            if sec_type == "stat":
                block, y = self._render_stat(section, colors, y)
            elif sec_type == "text_block":
                block, y = self._render_text_block(section, colors, y)
            elif sec_type == "comparison":
                block, y = self._render_comparison(section, colors, y)
            elif sec_type == "steps":
                block, y = self._render_steps(section, colors, y)
            elif sec_type == "timeline":
                block, y = self._render_timeline(section, colors, y)
            else:
                # неизвестный тип — пропускаем с предупреждением
                print(f"Неизвестный тип секции: {sec_type}, пропускаю")
                continue

            parts.append(block)

        total_height = y + self.PADDING
        content = "\n".join(parts)

        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{self.CANVAS_WIDTH}" height="{total_height}" '
            f'viewBox="0 0 {self.CANVAS_WIDTH} {total_height}">\n'
            f'  <defs>\n'
            f'    <filter id="shadow" x="-4%" y="-4%" width="108%" height="116%">\n'
            f'      <feDropShadow dx="0" dy="2" stdDeviation="4" flood-opacity="0.08"/>\n'
            f'    </filter>\n'
            f'  </defs>\n'
            f'  <rect width="{self.CANVAS_WIDTH}" height="{total_height}" fill="{colors["bg"]}"/>\n'
            f'{content}\n'
            f'</svg>'
        )
        return svg

    # ------------------------------------------------------------------
    # Секции
    # ------------------------------------------------------------------

    def _render_header(
        self, title: str, subtitle: str, colors: dict, y: float
    ) -> tuple[str, float]:
        parts: list[str] = []
        x_center = self.CANVAS_WIDTH / 2

        # title
        parts.append(
            f'  <text x="{x_center}" y="{y + 36}" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="28" font-weight="bold" '
            f'fill="{colors["text"]}" text-anchor="middle">'
            f'{self._escape(title)}</text>'
        )
        y += 50

        # subtitle
        if subtitle:
            parts.append(
                f'  <text x="{x_center}" y="{y + 18}" '
                f'font-family="Arial, Helvetica, sans-serif" font-size="16" '
                f'fill="{colors["text_secondary"]}" text-anchor="middle">'
                f'{self._escape(subtitle)}</text>'
            )
            y += 32

        # divider
        y += 10
        parts.append(
            f'  <line x1="{self.PADDING + 100}" y1="{y}" '
            f'x2="{self.CANVAS_WIDTH - self.PADDING - 100}" y2="{y}" '
            f'stroke="{colors["primary"]}" stroke-width="3" stroke-linecap="round"/>'
        )
        y += 30
        return "\n".join(parts), y

    def _render_stat(
        self, section: dict, colors: dict, y: float
    ) -> tuple[str, float]:
        value = str(section.get("value", "—"))
        label = str(section.get("label", ""))
        icon_name = str(section.get("icon", "")).lower()

        card_h = 110
        card_x = self.PADDING
        card_w = self.CONTENT_WIDTH

        parts: list[str] = []

        # card background
        parts.append(
            f'  <rect x="{card_x}" y="{y}" width="{card_w}" height="{card_h}" '
            f'rx="12" fill="{colors["card_bg"]}" stroke="{colors["card_border"]}" '
            f'stroke-width="1" filter="url(#shadow)"/>'
        )

        # accent bar on left
        parts.append(
            f'  <rect x="{card_x}" y="{y}" width="6" height="{card_h}" '
            f'rx="3" fill="{colors["primary"]}"/>'
        )

        content_x = card_x + 24
        icon_offset = 0

        # icon
        if icon_name and icon_name in ICONS:
            icon_svg = ICONS[icon_name].format(color=colors["primary"])
            parts.append(
                f'  <g transform="translate({content_x},{y + card_h / 2 - 12}) scale(1)">'
                f'{icon_svg}</g>'
            )
            icon_offset = 36

        # value
        parts.append(
            f'  <text x="{content_x + icon_offset}" y="{y + 48}" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="36" font-weight="bold" '
            f'fill="{colors["primary"]}">'
            f'{self._escape(value)}</text>'
        )

        # label
        parts.append(
            f'  <text x="{content_x + icon_offset}" y="{y + 78}" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="16" '
            f'fill="{colors["text_secondary"]}">'
            f'{self._escape(label)}</text>'
        )

        y += card_h + 20
        return "\n".join(parts), y

    def _render_text_block(
        self, section: dict, colors: dict, y: float
    ) -> tuple[str, float]:
        title = str(section.get("title", ""))
        content = str(section.get("content", ""))

        # estimate height based on text length
        content_lines = self._wrap_text(content, max_chars=80)
        card_h = 50 + len(content_lines) * 22 + 20

        card_x = self.PADDING
        card_w = self.CONTENT_WIDTH

        parts: list[str] = []

        # card
        parts.append(
            f'  <rect x="{card_x}" y="{y}" width="{card_w}" height="{card_h}" '
            f'rx="12" fill="{colors["card_bg"]}" stroke="{colors["card_border"]}" '
            f'stroke-width="1" filter="url(#shadow)"/>'
        )

        content_x = card_x + 24
        text_y = y + 32

        # title
        if title:
            parts.append(
                f'  <text x="{content_x}" y="{text_y}" '
                f'font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="bold" '
                f'fill="{colors["text"]}">'
                f'{self._escape(title)}</text>'
            )
            text_y += 28

        # content lines
        for line in content_lines:
            parts.append(
                f'  <text x="{content_x}" y="{text_y}" '
                f'font-family="Arial, Helvetica, sans-serif" font-size="14" '
                f'fill="{colors["text_secondary"]}">'
                f'{self._escape(line)}</text>'
            )
            text_y += 22

        y += card_h + 20
        return "\n".join(parts), y

    def _render_comparison(
        self, section: dict, colors: dict, y: float
    ) -> tuple[str, float]:
        items = section.get("items", [])
        title = str(section.get("title", ""))

        if not items:
            return "", y

        bar_height = 32
        bar_gap = 14
        card_h = 30 + len(items) * (bar_height + bar_gap) + 20
        if title:
            card_h += 32

        card_x = self.PADDING
        card_w = self.CONTENT_WIDTH

        parts: list[str] = []

        # card
        parts.append(
            f'  <rect x="{card_x}" y="{y}" width="{card_w}" height="{card_h}" '
            f'rx="12" fill="{colors["card_bg"]}" stroke="{colors["card_border"]}" '
            f'stroke-width="1" filter="url(#shadow)"/>'
        )

        content_x = card_x + 24
        inner_w = card_w - 48
        bar_y = y + 24

        # title
        if title:
            parts.append(
                f'  <text x="{content_x}" y="{bar_y + 14}" '
                f'font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="bold" '
                f'fill="{colors["text"]}">'
                f'{self._escape(title)}</text>'
            )
            bar_y += 32

        # find max numeric value for scaling
        max_val = self._find_max_numeric(items)

        for item in items:
            label = str(item.get("label", ""))
            value = str(item.get("value", ""))
            numeric = self._extract_numeric(value)
            ratio = (numeric / max_val) if max_val > 0 else 0.5

            # label
            parts.append(
                f'  <text x="{content_x}" y="{bar_y + 14}" '
                f'font-family="Arial, Helvetica, sans-serif" font-size="13" '
                f'fill="{colors["text"]}">'
                f'{self._escape(label)}</text>'
            )

            # bar background
            bar_x = content_x + 160
            bar_w = inner_w - 220
            parts.append(
                f'  <rect x="{bar_x}" y="{bar_y + 2}" width="{bar_w}" height="{bar_height - 8}" '
                f'rx="4" fill="{colors["bar_bg"]}"/>'
            )

            # bar fill
            fill_w = max(bar_w * ratio, 4)
            parts.append(
                f'  <rect x="{bar_x}" y="{bar_y + 2}" width="{fill_w:.1f}" height="{bar_height - 8}" '
                f'rx="4" fill="{colors["bar_fill"]}"/>'
            )

            # value text
            parts.append(
                f'  <text x="{bar_x + bar_w + 10}" y="{bar_y + 16}" '
                f'font-family="Arial, Helvetica, sans-serif" font-size="14" font-weight="bold" '
                f'fill="{colors["primary"]}">'
                f'{self._escape(value)}</text>'
            )

            bar_y += bar_height + bar_gap

        y += card_h + 20
        return "\n".join(parts), y

    def _render_steps(
        self, section: dict, colors: dict, y: float
    ) -> tuple[str, float]:
        items = section.get("items", [])
        title = str(section.get("title", ""))

        if not items:
            return "", y

        step_height = 64
        step_gap = 8
        card_h = 30 + len(items) * (step_height + step_gap) + 10
        if title:
            card_h += 32

        card_x = self.PADDING
        card_w = self.CONTENT_WIDTH

        parts: list[str] = []

        # card
        parts.append(
            f'  <rect x="{card_x}" y="{y}" width="{card_w}" height="{card_h}" '
            f'rx="12" fill="{colors["card_bg"]}" stroke="{colors["card_border"]}" '
            f'stroke-width="1" filter="url(#shadow)"/>'
        )

        content_x = card_x + 24
        step_y = y + 24

        # title
        if title:
            parts.append(
                f'  <text x="{content_x}" y="{step_y + 14}" '
                f'font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="bold" '
                f'fill="{colors["text"]}">'
                f'{self._escape(title)}</text>'
            )
            step_y += 32

        for item in items:
            step_num = str(item.get("step", ""))
            step_title = str(item.get("title", ""))
            step_desc = str(item.get("description", ""))

            # circle with number
            circle_cx = content_x + 20
            circle_cy = step_y + 26
            parts.append(
                f'  <circle cx="{circle_cx}" cy="{circle_cy}" r="16" '
                f'fill="{colors["primary"]}"/>'
            )
            parts.append(
                f'  <text x="{circle_cx}" y="{circle_cy + 5}" '
                f'font-family="Arial, Helvetica, sans-serif" font-size="14" font-weight="bold" '
                f'fill="white" text-anchor="middle">'
                f'{self._escape(step_num)}</text>'
            )

            # connector line (except last)
            if item != items[-1]:
                parts.append(
                    f'  <line x1="{circle_cx}" y1="{circle_cy + 16}" '
                    f'x2="{circle_cx}" y2="{circle_cy + 16 + step_gap + 20}" '
                    f'stroke="{colors["card_border"]}" stroke-width="2" stroke-dasharray="4,4"/>'
                )

            # title
            text_x = content_x + 52
            parts.append(
                f'  <text x="{text_x}" y="{step_y + 22}" '
                f'font-family="Arial, Helvetica, sans-serif" font-size="15" font-weight="bold" '
                f'fill="{colors["text"]}">'
                f'{self._escape(step_title)}</text>'
            )

            # description
            if step_desc:
                parts.append(
                    f'  <text x="{text_x}" y="{step_y + 44}" '
                    f'font-family="Arial, Helvetica, sans-serif" font-size="13" '
                    f'fill="{colors["text_secondary"]}">'
                    f'{self._escape(step_desc[:90])}</text>'
                )

            step_y += step_height + step_gap

        y += card_h + 20
        return "\n".join(parts), y

    def _render_timeline(
        self, section: dict, colors: dict, y: float
    ) -> tuple[str, float]:
        items = section.get("items", [])
        title = str(section.get("title", ""))

        if not items:
            return "", y

        item_height = 52
        item_gap = 10
        card_h = 30 + len(items) * (item_height + item_gap) + 10
        if title:
            card_h += 32

        card_x = self.PADDING
        card_w = self.CONTENT_WIDTH

        parts: list[str] = []

        # card
        parts.append(
            f'  <rect x="{card_x}" y="{y}" width="{card_w}" height="{card_h}" '
            f'rx="12" fill="{colors["card_bg"]}" stroke="{colors["card_border"]}" '
            f'stroke-width="1" filter="url(#shadow)"/>'
        )

        content_x = card_x + 24
        tl_y = y + 24

        # title
        if title:
            parts.append(
                f'  <text x="{content_x}" y="{tl_y + 14}" '
                f'font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="bold" '
                f'fill="{colors["text"]}">'
                f'{self._escape(title)}</text>'
            )
            tl_y += 32

        line_x = content_x + 60

        for i, item in enumerate(items):
            date = str(item.get("date", ""))
            event = str(item.get("event", ""))

            dot_cy = tl_y + 20

            # vertical line
            if i < len(items) - 1:
                parts.append(
                    f'  <line x1="{line_x}" y1="{dot_cy + 8}" '
                    f'x2="{line_x}" y2="{dot_cy + item_height + item_gap - 4}" '
                    f'stroke="{colors["accent"]}" stroke-width="2"/>'
                )

            # dot
            parts.append(
                f'  <circle cx="{line_x}" cy="{dot_cy}" r="7" '
                f'fill="{colors["primary"]}" stroke="{colors["card_bg"]}" stroke-width="3"/>'
            )

            # date
            parts.append(
                f'  <text x="{content_x}" y="{dot_cy + 5}" '
                f'font-family="Arial, Helvetica, sans-serif" font-size="13" font-weight="bold" '
                f'fill="{colors["primary"]}" text-anchor="start">'
                f'{self._escape(date)}</text>'
            )

            # event
            parts.append(
                f'  <text x="{line_x + 20}" y="{dot_cy + 5}" '
                f'font-family="Arial, Helvetica, sans-serif" font-size="14" '
                f'fill="{colors["text"]}">'
                f'{self._escape(event[:80])}</text>'
            )

            tl_y += item_height + item_gap

        y += card_h + 20
        return "\n".join(parts), y

    # ------------------------------------------------------------------
    # Конвертация
    # ------------------------------------------------------------------

    def _convert_to_png(self, svg_path: Path, png_path: Path) -> None:
        try:
            import cairosvg
            cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), scale=2)
        except ImportError:
            print("cairosvg не установлен — PNG не создан. pip install cairosvg")
        except Exception as e:
            print(f"Ошибка конвертации SVG→PNG: {e}")

    def _convert_to_pdf(self, svg_path: Path, pdf_path: Path) -> None:
        try:
            import cairosvg
            cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))
        except ImportError:
            print("cairosvg не установлен — PDF не создан. pip install cairosvg")
        except Exception as e:
            print(f"Ошибка конвертации SVG→PDF: {e}")

    # ------------------------------------------------------------------
    # Утилиты
    # ------------------------------------------------------------------

    def _validate(self, diagram: dict[str, Any]) -> None:
        if not isinstance(diagram, dict):
            raise InfographicRenderError("diagram должен быть dict")

        sections = diagram.get("sections")
        if not isinstance(sections, list) or not sections:
            raise InfographicRenderError("diagram['sections'] должен быть непустым списком")

    def _escape(self, text: str) -> str:
        text = str(text)
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        text = text.replace('"', "&quot;")
        return text

    def _wrap_text(self, text: str, max_chars: int = 80) -> list[str]:
        words = text.split()
        lines: list[str] = []
        current: list[str] = []
        current_len = 0

        for word in words:
            if current_len + len(word) + 1 > max_chars and current:
                lines.append(" ".join(current))
                current = [word]
                current_len = len(word)
            else:
                current.append(word)
                current_len += len(word) + 1

        if current:
            lines.append(" ".join(current))

        return lines or [""]

    def _extract_numeric(self, value: str) -> float:
        match = re.search(r"[\d]+(?:[.,]\d+)?", str(value))
        if match:
            return float(match.group().replace(",", "."))
        return 0.0

    def _find_max_numeric(self, items: list[dict]) -> float:
        values = [self._extract_numeric(str(item.get("value", ""))) for item in items]
        return max(values) if values else 1.0

    def _sanitize_filename(self, value: str) -> str:
        value = value.strip()
        value = re.sub(r"[^\w\-\.]+", "_", value, flags=re.UNICODE)
        value = re.sub(r"_+", "_", value)
        return value.strip("._") or "infographic"
