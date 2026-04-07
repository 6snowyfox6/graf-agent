from __future__ import annotations

import re
from pathlib import Path
from typing import Any


class InfographicRenderError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Цветовые схемы
# ---------------------------------------------------------------------------

COLOR_SCHEMES: dict[str, dict[str, str | list[str]]] = {
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
        "palette": ["#2563EB", "#3B82F6", "#60A5FA", "#93C5FD", "#1D4ED8"]
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
        "palette": ["#059669", "#10B981", "#34D399", "#6EE7B7", "#047857"]
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
        "palette": ["#EA580C", "#F97316", "#FB923C", "#FDBA74", "#C2410C"]
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
        "palette": ["#7C3AED", "#8B5CF6", "#A78BFA", "#C4B5FD", "#6D28D9"]
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
        "palette": ["#60A5FA", "#93C5FD", "#BFDBFE", "#3B82F6", "#2563EB"]
    },
    "sunset": {
        "primary": "#E11D48",
        "accent": "#FB7185",
        "bg": "#FFF1F2",
        "text": "#4C0519",
        "text_secondary": "#881337",
        "card_bg": "#FFFFFF",
        "card_border": "#FECDD3",
        "bar_bg": "#FFE4E6",
        "bar_fill": "#E11D48",
        "palette": ["#E11D48", "#F43F5E", "#FB7185", "#FDA4AF", "#BE123C"]
    },
    "cyberpunk": {
        "primary": "#D946EF",
        "accent": "#F0ABFC",
        "bg": "#FDF4FF",
        "text": "#4A044E",
        "text_secondary": "#701A75",
        "card_bg": "#FFFFFF",
        "card_border": "#F5D0FE",
        "bar_bg": "#FAE8FF",
        "bar_fill": "#D946EF",
        "palette": ["#D946EF", "#E879F9", "#F0ABFC", "#F5D0FE", "#A21CAF"]
    },
    "neon": {
        "primary": "#0D9488",
        "accent": "#2DD4BF",
        "bg": "#F0FDFA",
        "text": "#134E4A",
        "text_secondary": "#115E59",
        "card_bg": "#FFFFFF",
        "card_border": "#99F6E4",
        "bar_bg": "#CCFBF1",
        "bar_fill": "#0D9488",
        "palette": ["#0D9488", "#14B8A6", "#2DD4BF", "#5EEAD4", "#0F766E"]
    },
    "pastel": {
        "primary": "#F59E0B",
        "accent": "#FCD34D",
        "bg": "#FFFBEB",
        "text": "#78350F",
        "text_secondary": "#92400E",
        "card_bg": "#FFFFFF",
        "card_border": "#FDE68A",
        "bar_bg": "#FEF3C7",
        "bar_fill": "#F59E0B",
        "palette": ["#F59E0B", "#FBBF24", "#FCD34D", "#FDE68A", "#D97706"]
    },
}

DEFAULT_SCHEME = "blue"
PRESENTATION_SAFE_SCHEMES = {"blue", "green", "orange", "pastel"}
SCHEME_FALLBACKS = {
    "purple": "blue",
    "dark": "blue",
    "sunset": "orange",
    "cyberpunk": "blue",
    "neon": "green",
}

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
    "arrow-up": (
        '<path d="M12 19V5M5 12l7-7 7 7" '
        'fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
    ),
    "check": (
        '<path d="M20 6L9 17l-5-5" '
        'fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
    ),
    "gear": (
        '<path d="M12 15a3 3 0 100-6 3 3 0 000 6z" fill="none" stroke="{color}" stroke-width="2"/>'
        '<path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06a1.65 1.65 0 001.82.33H9a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06a1.65 1.65 0 00-.33 1.82V9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z" '
        'fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
    ),
    "lightning": (
        '<path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" '
        'fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
    ),
    "globe": (
        '<circle cx="12" cy="12" r="10" fill="none" stroke="{color}" stroke-width="2"/>'
        '<path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" '
        'fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
    ),
    "shield": (
        '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" '
        'fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
    ),
    "smartphone": (
        '<rect x="5" y="2" width="14" height="20" rx="2" ry="2" fill="none" stroke="{color}" stroke-width="2"/>'
        '<path d="M12 18h.01" stroke="{color}" stroke-width="2" stroke-linecap="round"/>'
    ),
    "heart": (
        '<path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" '
        'fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
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
      - color_scheme: blue | green | orange | pastel
      - sections: список секций разных типов

    Типы секций: stat, text_block, comparison, steps, timeline, donut_chart, gauge.
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

    def _dispatch_render(self, section: dict, sec_type: str, colors: dict, y: float, x: float, w: float) -> tuple[str, float]:
        if sec_type == "text_block":
            return self._render_text_block(section, colors, y, x, w)
        elif sec_type == "comparison":
            return self._render_comparison(section, colors, y)
        elif sec_type == "steps":
            return self._render_steps(section, colors, y)
        elif sec_type == "timeline":
            return self._render_timeline(section, colors, y)
        elif sec_type == "donut_chart":
            return self._render_donut_chart(section, colors, y, x, w)
        elif sec_type == "gauge":
            return self._render_gauge(section, colors, y, x, w)
        elif sec_type == "tags":
            return self._render_tags(section, colors, y, x, w)
        elif sec_type == "process":
            return self._render_process(section, colors, y)
        elif sec_type == "image":
            return self._render_image(section, colors, y)
        elif sec_type == "neural_network":
            return self._render_neural_network(section, colors, y, x, w)
        return "", y

    def build_svg(self, diagram: dict[str, Any]) -> str:
        scheme_name = str(diagram.get("color_scheme", DEFAULT_SCHEME)).lower()
        scheme_name = SCHEME_FALLBACKS.get(scheme_name, scheme_name)
        if scheme_name not in PRESENTATION_SAFE_SCHEMES:
            scheme_name = DEFAULT_SCHEME
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
        i = 0
        combinable = ["donut_chart", "gauge", "text_block", "tags"]
        
        while i < len(sections):
            section = sections[i]
            sec_type = str(section.get("type", "")).lower()

            if sec_type == "stat":
                # Lookahead for stat groups to render in a multi-col grid
                stat_group = [section]
                j = i + 1
                while j < len(sections) and str(sections[j].get("type", "")).lower() == "stat":
                    stat_group.append(sections[j])
                    j += 1
                i = j - 1
                block, y = self._render_stat_group(stat_group, colors, y)
                parts.append(block)
                i += 1
                continue

            if sec_type in combinable and i + 1 < len(sections):
                next_type = str(sections[i + 1].get("type", "")).lower()
                
                content_len = len(str(section.get("content", ""))) if sec_type == "text_block" else 0
                next_len = len(str(sections[i+1].get("content", ""))) if next_type == "text_block" else 0
                
                if next_type in combinable and content_len < 180 and next_len < 180:
                    half_w = (self.CONTENT_WIDTH - 20) / 2
                    b1, y1 = self._dispatch_render(section, sec_type, colors, y, self.PADDING, half_w)
                    b2, y2 = self._dispatch_render(sections[i+1], next_type, colors, y, self.PADDING + half_w + 20, half_w)
                    
                    parts.append(b1)
                    parts.append(b2)
                    y = max(y1, y2)
                    i += 2
                    continue

            block, y = self._dispatch_render(section, sec_type, colors, y, self.PADDING, self.CONTENT_WIDTH)
            if block:
                parts.append(block)
            else:
                print(f"Неизвестный тип секции: {sec_type}, пропускаю")
            i += 1

        total_height = y + self.PADDING
        content = "\n".join(parts)

        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{self.CANVAS_WIDTH}" height="{total_height}" '
            f'viewBox="0 0 {self.CANVAS_WIDTH} {total_height}">\n'
            f'  <defs>\n'
            f'    <linearGradient id="bar-grad" x1="0" y1="0" x2="1" y2="0">\n'
            f'      <stop offset="0%" stop-color="{colors["accent"]}"/>\n'
            f'      <stop offset="100%" stop-color="{colors["primary"]}"/>\n'
            f'    </linearGradient>\n'
            f'    <linearGradient id="donut-grad" x1="0" y1="0" x2="1" y2="1">\n'
            f'      <stop offset="0%" stop-color="{colors["accent"]}"/>\n'
            f'      <stop offset="100%" stop-color="{colors["primary"]}"/>\n'
            f'    </linearGradient>\n'
            f'    <filter id="shadow" x="-4%" y="-4%" width="108%" height="116%">\n'
            f'      <feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="{colors["primary"]}" flood-opacity="0.12"/>\n'
            f'    </filter>\n'
            f'    <pattern id="bg-grid" width="20" height="20" patternUnits="userSpaceOnUse">\n'
            f'      <circle cx="2" cy="2" r="1.5" fill="{colors["accent"]}" fill-opacity="0.15"/>\n'
            f'    </pattern>\n'
            f'  </defs>\n'
            f'  <rect width="{self.CANVAS_WIDTH}" height="{total_height}" fill="{colors["bg"]}"/>\n'
            f'  <rect width="{self.CANVAS_WIDTH}" height="{total_height}" fill="url(#bg-grid)"/>\n'
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

    def _render_stat_group(
        self, stat_group: list[dict], colors: dict, y: float
    ) -> tuple[str, float]:
        parts: list[str] = []
        
        max_cols = 3
        rows = []
        for i in range(0, len(stat_group), max_cols):
            rows.append(stat_group[i:i+max_cols])
            
        card_h = 110
        gap = 20
        total_w = self.CONTENT_WIDTH
        
        for row in rows:
            cols = len(row)
            item_w = (total_w - (cols - 1) * gap) / cols
            x = self.PADDING
            
            for section in row:
                value = str(section.get("value", "—"))
                label = str(section.get("label", ""))
                icon_name = str(section.get("icon", "")).lower()
                
                # card background
                parts.append(
                    f'  <rect x="{x}" y="{y}" width="{item_w}" height="{card_h}" '
                    f'rx="12" fill="{colors["card_bg"]}" stroke="{colors["card_border"]}" '
                    f'stroke-width="1" filter="url(#shadow)"/>'
                )
                
                # left bar (with donut gradient instead of flat)
                parts.append(
                    f'  <rect x="{x}" y="{y}" width="6" height="{card_h}" '
                    f'rx="3" fill="url(#donut-grad)"/>'
                )
                
                content_x = x + 24
                icon_offset = 0
                
                if icon_name and icon_name in ICONS:
                    icon_svg = ICONS[icon_name].format(color=colors["primary"])
                    parts.append(
                        f'  <g transform="translate({content_x},{y + card_h / 2 - 12}) scale(1)">'
                        f'{icon_svg}</g>'
                    )
                    icon_offset = 36
                
                val_fs = "32" if cols < 3 else "28"
                lbl_fs = "14" if cols < 3 else "12"

                parts.append(
                    f'  <text x="{content_x + icon_offset}" y="{y + 48}" '
                    f'font-family="Arial, Helvetica, sans-serif" font-size="{val_fs}" font-weight="bold" '
                    f'fill="{colors["primary"]}">'
                    f'{self._escape(value)}</text>'
                )
                
                parts.append(
                    f'  <text x="{content_x + icon_offset}" y="{y + 78}" '
                    f'font-family="Arial, Helvetica, sans-serif" font-size="{lbl_fs}" '
                    f'fill="{colors["text_secondary"]}">'
                    f'{self._escape(label[:30])}</text>'
                )
                
                x += item_w + gap
            y += card_h + gap
            
        return "\n".join(parts), y

    def _render_text_block(
        self, section: dict, colors: dict, y: float, x: float = None, w: float = None
    ) -> tuple[str, float]:
        title = str(section.get("title", ""))
        content = str(section.get("content", ""))

        card_w = w if w is not None else self.CONTENT_WIDTH
        content_lines = self._wrap_text(content, max_chars=int(card_w / 12))
        card_h = 40 + len(content_lines) * 22 + 20
        if title:
            card_h += 28

        card_x = x if x is not None else self.PADDING
        
        is_accent = (len(content) + len(title)) % 3 == 0
        bg_color = colors["primary"] if is_accent else colors["card_bg"]
        text_color = "#FFFFFF" if is_accent else colors["text"]
        text_sec_color = "#F8FAFC" if is_accent else colors["text_secondary"]
        border_color = colors["primary"] if is_accent else colors["card_border"]

        parts: list[str] = []

        # card
        parts.append(
            f'  <rect x="{card_x}" y="{y}" width="{card_w}" height="{card_h}" '
            f'rx="12" fill="{bg_color}" stroke="{border_color}" '
            f'stroke-width="1" filter="url(#shadow)"/>'
        )

        content_x = card_x + 24
        text_y = y + 32

        # title
        if title:
            parts.append(
                f'  <text x="{content_x}" y="{text_y}" '
                f'font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="bold" '
                f'fill="{text_color}">'
                f'{self._escape(title)}</text>'
            )
            text_y += 28

        # content lines
        for line in content_lines:
            parts.append(
                f'  <text x="{content_x}" y="{text_y}" '
                f'font-family="Arial, Helvetica, sans-serif" font-size="14" '
                f'fill="{text_sec_color}">'
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
        palette = colors.get("palette", [colors["primary"], colors["accent"]])

        for i, item in enumerate(items):
            label = str(item.get("label", ""))
            value = str(item.get("value", ""))
            numeric = self._extract_numeric(value)
            ratio = (numeric / max_val) if max_val > 0 else 0.5
            
            c_fill = palette[i % len(palette)]

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

            # bar fill (using palette colors instead of single gradient)
            fill_w = max(bar_w * ratio, 4)
            parts.append(
                f'  <rect x="{bar_x}" y="{bar_y + 2}" width="{fill_w:.1f}" height="{bar_height - 8}" '
                f'rx="4" fill="{c_fill}"/>'
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

        center_line_x = card_x + card_w / 2

        for i, item in enumerate(items):
            date = str(item.get("date", ""))
            event = str(item.get("event", ""))

            dot_cy = tl_y + 20

            # vertical line
            if i < len(items) - 1:
                parts.append(
                    f'  <line x1="{center_line_x}" y1="{dot_cy + 8}" '
                    f'x2="{center_line_x}" y2="{dot_cy + item_height + item_gap - 4}" '
                    f'stroke="{colors["accent"]}" stroke-width="2" stroke-linecap="round" stroke-dasharray="4,4"/>'
                )

            # dot
            parts.append(
                f'  <circle cx="{center_line_x}" cy="{dot_cy}" r="7" '
                f'fill="{colors["primary"]}" stroke="{colors["card_bg"]}" stroke-width="3"/>'
            )

            is_left = (i % 2 == 0)
            
            if is_left:
                parts.append(
                    f'  <text x="{center_line_x - 24}" y="{dot_cy - 4}" '
                    f'font-family="Arial, Helvetica, sans-serif" font-size="14" font-weight="bold" '
                    f'fill="{colors["primary"]}" text-anchor="end">'
                    f'{self._escape(date)}</text>'
                )
                parts.append(
                    f'  <text x="{center_line_x - 24}" y="{dot_cy + 14}" '
                    f'font-family="Arial, Helvetica, sans-serif" font-size="13" '
                    f'fill="{colors["text"]}" text-anchor="end">'
                    f'{self._escape(event[:40])}</text>'
                )
            else:
                parts.append(
                    f'  <text x="{center_line_x + 24}" y="{dot_cy - 4}" '
                    f'font-family="Arial, Helvetica, sans-serif" font-size="14" font-weight="bold" '
                    f'fill="{colors["primary"]}" text-anchor="start">'
                    f'{self._escape(date)}</text>'
                )
                parts.append(
                    f'  <text x="{center_line_x + 24}" y="{dot_cy + 14}" '
                    f'font-family="Arial, Helvetica, sans-serif" font-size="13" '
                    f'fill="{colors["text"]}" text-anchor="start">'
                    f'{self._escape(event[:40])}</text>'
                )

            tl_y += item_height + item_gap

        y += card_h + 20
        return "\n".join(parts), y

    def _render_donut_chart(
        self, section: dict, colors: dict, y: float, x: float = None, w: float = None
    ) -> tuple[str, float]:
        items = section.get("items", [])
        title = str(section.get("title", ""))

        if not items:
            return "", y

        card_h = 240
        card_x = x if x is not None else self.PADDING
        card_w = w if w is not None else self.CONTENT_WIDTH

        parts: list[str] = []

        # card
        parts.append(
            f'  <rect x="{card_x}" y="{y}" width="{card_w}" height="{card_h}" '
            f'rx="12" fill="{colors["card_bg"]}" stroke="{colors["card_border"]}" '
            f'stroke-width="1" filter="url(#shadow)"/>'
        )

        content_x = card_x + 24
        chart_y = y + 24

        if title:
            parts.append(
                f'  <text x="{content_x}" y="{chart_y + 14}" '
                f'font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="bold" '
                f'fill="{colors["text"]}">'
                f'{self._escape(title)}</text>'
            )
            chart_y += 32

        is_narrow = card_w < 500
        radius = 50 if is_narrow else 70
        cx = card_x + card_w * 0.3 if is_narrow else card_x + card_w * 0.25
        cy = chart_y + (60 if is_narrow else 80)
        circumference = 2 * 3.14159265 * radius
        
        total = sum(self._extract_numeric(item.get("value", "0")) for item in items)
        if total == 0:
            total = 1
            
        parts.append(
            f'  <circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" '
            f'stroke="{colors["bar_bg"]}" stroke-width="24"/>'
        )
        
        palette = [colors["primary"], colors["accent"], colors["text_secondary"], colors["bar_fill"], colors["card_border"]]
        
        current_offset = circumference
        legend_x = card_x + card_w * 0.55 if not is_narrow else card_x + card_w * 0.65
        legend_y = chart_y + 10 if is_narrow else chart_y + 20
        legend_fs = "12" if is_narrow else "14"
        
        for i, item in enumerate(items):
            val = self._extract_numeric(item.get("value", "0"))
            label = str(item.get("label", ""))
            ratio = val / total
            dasharray = f"{circumference * ratio} {circumference}"
            color = palette[i % len(palette)]
            
            if ratio > 0:
                parts.append(
                    f'  <circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" '
                    f'stroke="{color}" stroke-width="24" '
                    f'stroke-dasharray="{dasharray}" stroke-dashoffset="{current_offset}" '
                    f'transform="rotate(-90 {cx} {cy})"/>'
                )
                current_offset -= circumference * ratio
            
            parts.append(
                f'  <rect x="{legend_x}" y="{legend_y}" width="14" height="14" rx="2" fill="{color}"/>'
            )
            parts.append(
                f'  <text x="{legend_x + 24}" y="{legend_y + 12}" '
                f'font-family="Arial, Helvetica, sans-serif" font-size="{legend_fs}" '
                f'fill="{colors["text"]}">'
                f'{self._escape(label[:12])} ({self._escape(str(item.get("value", "")))})</text>'
            )
            legend_y += 24 if is_narrow else 28

        parts.append(
            f'  <text x="{cx}" y="{cy + 8}" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="20" font-weight="bold" '
            f'fill="{colors["text"]}" text-anchor="middle">'
            f'{self._escape(str(int(total)))}</text>'
        )

        y += card_h + 20
        return "\n".join(parts), y

    def _render_gauge(
        self, section: dict, colors: dict, y: float, x: float = None, w: float = None
    ) -> tuple[str, float]:
        label = str(section.get("label", ""))
        value_str = str(section.get("value", "0"))
        max_str = str(section.get("max_value", "100"))
        
        val = self._extract_numeric(value_str)
        max_val = self._extract_numeric(max_str)
        if max_val <= 0:
            max_val = 1
            
        ratio = min(max(val / max_val, 0), 1)

        card_h = 180
        card_x = x if x is not None else self.PADDING
        card_w = w if w is not None else self.CONTENT_WIDTH

        parts: list[str] = []

        parts.append(
            f'  <rect x="{card_x}" y="{y}" width="{card_w}" height="{card_h}" '
            f'rx="12" fill="{colors["card_bg"]}" stroke="{colors["card_border"]}" '
            f'stroke-width="1" filter="url(#shadow)"/>'
        )

        cx = card_x + card_w / 2
        cy = y + 130
        radius = 80
        circumference = 3.14159265 * radius
        
        if label:
            parts.append(
                f'  <text x="{card_x + 24}" y="{y + 36}" '
                f'font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="bold" '
                f'fill="{colors["text"]}">'
                f'{self._escape(label)}</text>'
            )

        parts.append(
            f'  <path d="M {cx - radius} {cy} A {radius} {radius} 0 0 1 {cx + radius} {cy}" '
            f'fill="none" stroke="{colors["bar_bg"]}" stroke-width="20" stroke-linecap="round"/>'
        )

        dasharray = circumference
        dashoffset = circumference * (1 - ratio)
        parts.append(
            f'  <path d="M {cx - radius} {cy} A {radius} {radius} 0 0 1 {cx + radius} {cy}" '
            f'fill="none" stroke="url(#donut-grad)" stroke-width="20" stroke-linecap="round" '
            f'stroke-dasharray="{dasharray}" stroke-dashoffset="{dashoffset}"/>'
        )
        
        parts.append(
            f'  <text x="{cx}" y="{cy - 10}" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="36" font-weight="bold" '
            f'fill="{colors["text"]}" text-anchor="middle">'
            f'{self._escape(value_str)}</text>'
        )
        
        parts.append(
            f'  <text x="{cx}" y="{cy + 15}" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="14" '
            f'fill="{colors["text_secondary"]}" text-anchor="middle">'
            f'из {self._escape(max_str)}</text>'
        )

        y += card_h + 20
        return "\n".join(parts), y

    def _render_tags(
        self, section: dict, colors: dict, y: float, x: float = None, w: float = None
    ) -> tuple[str, float]:
        tags = section.get("items", [])
        title = str(section.get("title", ""))
        
        if not tags: return "", y
        
        card_w = w if w is not None else self.CONTENT_WIDTH
        card_x = x if x is not None else self.PADDING
        content_x = card_x + 24
        
        parts: list[str] = []
        badge_y = y + 24
        if title:
            parts.append(
                f'  <text x="{content_x}" y="{badge_y + 14}" font-family="Arial, Helvetica, sans-serif" '
                f'font-size="18" font-weight="bold" fill="{colors["text"]}">{self._escape(title)}</text>'
            )
            badge_y += 40
            
        palette = colors.get("palette", [colors["primary"], colors["accent"]])
        
        current_x = content_x
        row_y = badge_y
        for i, t in enumerate(tags):
            text = str(t.get("label", t) if isinstance(t, dict) else t)
            text_w = len(text) * 9.5
            badge_w = max(text_w + 30, 40)
            
            if current_x + badge_w > card_x + card_w - 24:
                current_x = content_x
                row_y += 40
            
            c_bg = palette[i % len(palette)]
            
            parts.append(
                f'  <rect x="{current_x}" y="{row_y}" width="{badge_w}" height="28" rx="14" '
                f'fill="{c_bg}" fill-opacity="0.15" stroke="{c_bg}" stroke-width="1"/>'
            )
            parts.append(
                f'  <text x="{current_x + badge_w/2}" y="{row_y + 19}" '
                f'font-family="Arial, Helvetica, sans-serif" font-size="13" font-weight="bold" '
                f'fill="{c_bg}" text-anchor="middle">{self._escape(text)}</text>'
            )
            
            current_x += badge_w + 12
            
        card_h = (row_y + 40) - y
        
        bg_card = (
            f'  <rect x="{card_x}" y="{y}" width="{card_w}" height="{card_h}" rx="12" '
            f'fill="{colors["card_bg"]}" stroke="{colors["card_border"]}" stroke-width="1" filter="url(#shadow)"/>'
        )
        return bg_card + "\n" + "\n".join(parts), y + card_h + 20

    def _render_process(
        self, section: dict, colors: dict, y: float
    ) -> tuple[str, float]:
        items = section.get("items", [])
        title = str(section.get("title", ""))
        
        if not items: return "", y
        
        card_h = 140
        card_x = self.PADDING
        card_w = self.CONTENT_WIDTH
        parts: list[str] = []
        
        parts.append(
            f'  <rect x="{card_x}" y="{y}" width="{card_w}" height="{card_h}" rx="12" '
            f'fill="{colors["card_bg"]}" stroke="{colors["card_border"]}" stroke-width="1" filter="url(#shadow)"/>'
        )
        
        content_x = card_x + 24
        py = y + 24
        if title:
            parts.append(
                f'  <text x="{content_x}" y="{py + 14}" font-family="Arial, Helvetica, sans-serif" '
                f'font-size="18" font-weight="bold" fill="{colors["text"]}">{self._escape(title)}</text>'
            )
            py += 32
        
        step_w = (card_w - 48 - (len(items)-1) * 15) / len(items)
        palette = colors.get("palette", [colors["primary"], colors["accent"]])
        
        for i, item in enumerate(items):
            step = str(item.get("step", str(i+1)))
            stitle = str(item.get("title", ""))
            
            sx = content_x + i * (step_w + 15)
            c_fill = palette[i % len(palette)]
            
            ch_h = 50
            if i < len(items) - 1:
                chev_path = f"M {sx} {py} H {sx + step_w - 15} L {sx + step_w} {py + ch_h/2} L {sx + step_w - 15} {py + ch_h} H {sx} L {sx + 15} {py + ch_h/2} Z"
                if i == 0: chev_path = f"M {sx} {py} H {sx + step_w - 15} L {sx + step_w} {py + ch_h/2} L {sx + step_w - 15} {py + ch_h} H {sx} V {py} Z"
            else:
                chev_path = f"M {sx} {py} H {sx + step_w} V {py + ch_h} H {sx} L {sx + 15} {py + ch_h/2} Z"
                if len(items) == 1: chev_path = f"M {sx} {py} H {sx + step_w} V {py + ch_h} H {sx} V {py} Z"
                
            parts.append(f'  <path d="{chev_path}" fill="{c_fill}" fill-opacity="0.9"/>')
            
            parts.append(
                f'  <text x="{sx + step_w/2 + 5}" y="{py + 20}" font-family="Arial, Helvetica, sans-serif" '
                f'font-size="12" font-weight="bold" fill="#ffffff" fill-opacity="0.8" text-anchor="middle">ШАГ {self._escape(step)}</text>'
            )
            parts.append(
                f'  <text x="{sx + step_w/2 + 5}" y="{py + 36}" font-family="Arial, Helvetica, sans-serif" '
                f'font-size="14" font-weight="bold" fill="#ffffff" text-anchor="middle">{self._escape(stitle[:30])}</text>'
            )

        y += card_h + 20
        return "\n".join(parts), y

    def _render_image(
        self, section: dict, colors: dict, y: float
    ) -> tuple[str, float]:
        url = str(section.get("url", "https://picsum.photos/720/300"))
        caption = str(section.get("caption", ""))
        
        card_w = self.CONTENT_WIDTH
        img_h = 300
        card_h = img_h
        if caption: card_h += 40
        
        parts: list[str] = []
        clip_id = f"clip-{int(y)}"
        
        parts.append(f'  <defs><clipPath id="{clip_id}"><rect x="{self.PADDING}" y="{y}" width="{card_w}" height="{img_h}" rx="12"/></clipPath></defs>')
        parts.append(
            f'  <rect x="{self.PADDING}" y="{y}" width="{card_w}" height="{card_h}" rx="12" '
            f'fill="{colors["card_bg"]}" stroke="{colors["card_border"]}" stroke-width="1" filter="url(#shadow)"/>'
        )
        parts.append(
            f'  <image href="{self._escape(url)}" x="{self.PADDING}" y="{y}" width="{card_w}" height="{img_h}" '
            f'preserveAspectRatio="xMidYMid slice" clip-path="url(#{clip_id})"/>'
        )
        
        if caption:
            parts.append(
                f'  <text x="{self.PADDING + card_w/2}" y="{y + img_h + 26}" font-family="Arial, Helvetica, sans-serif" '
                f'font-size="15" fill="{colors["text_secondary"]}" text-anchor="middle">{self._escape(caption)}</text>'
            )
            
        y += card_h + 20
        return "\n".join(parts), y

    def _render_neural_network(
        self, section: dict, colors: dict, y: float, x: float = None, w: float = None
    ) -> tuple[str, float]:
        diagram = section.get("diagram")
        if not diagram:
            return "", y
            
        card_w = w if w is not None else self.CONTENT_WIDTH
        card_x = x if x is not None else self.PADDING
        
        import uuid
        import subprocess
        from plotneuralnet_renderer import PlotNeuralNetRenderer
        
        job_id = f"nested_nn_{uuid.uuid4().hex[:8]}"
        pnn = PlotNeuralNetRenderer(output_root=self.output_root)
        
        try:
            res = pnn.render(diagram, job_id)
            pdf_path = res["pdf_path"]
            work_dir = res["work_dir"]
            
            png_base = str(Path(work_dir) / job_id)
            subprocess.run(["pdftoppm", "-png", "-singlefile", "-r", "300", pdf_path, png_base], check=True)
            
            png_path = png_base + ".png"
            if not Path(png_path).exists():
                print(f"Ошибка pdftoppm: PNG не сгенерирован {png_path}")
                return "", y
                
            with open(png_path, "rb") as f:
                b64 = __import__("base64").b64encode(f.read()).decode("utf-8")
                img_url = f"data:image/png;base64,{b64}"
        except Exception as e:
            print(f"Ошибка вложенного рендера PlotNeuralNet: {e}")
            return "", y

        card_h = 450
        parts = []
        
        parts.append(
            f'  <rect x="{card_x}" y="{y}" width="{card_w}" height="{card_h}" rx="12" '
            f'fill="{colors["card_bg"]}" stroke="{colors["card_border"]}" stroke-width="1" filter="url(#shadow)"/>'
        )
        
        clip_id = f"clip_{job_id}"
        parts.append(f'  <defs><clipPath id="{clip_id}"><rect x="{card_x}" y="{y}" width="{card_w}" height="{card_h}" rx="12"/></clipPath></defs>')
        
        title = str(section.get("title", "")) or str(diagram.get("title", ""))
        
        img_y = y + 20
        img_h = card_h - 40
        
        if title:
            img_y = y + 50
            img_h = card_h - 60
            parts.append(
                f'  <text x="{card_x + card_w/2}" y="{y + 32}" '
                f'font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="bold" '
                f'fill="{colors["text"]}" text-anchor="middle">{self._escape(title)}</text>'
            )
            
        parts.append(
            f'  <image href="{img_url}" x="{card_x+10}" y="{img_y}" width="{card_w-20}" height="{img_h}" '
            f'preserveAspectRatio="xMidYMid meet" clip-path="url(#{clip_id})"/>'
        )
        
        return "\n".join(parts), y + card_h + 20

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
