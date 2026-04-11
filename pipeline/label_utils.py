"""Label cleaning, kind inference and edge deduplication utilities."""

from __future__ import annotations

import re


FORBIDDEN_VISIBLE_TOKENS = {
    "input", "output", "block", "conv", "ui", "api", "db", "service"
}


def clean_visible_label(label: str) -> str:
    s = str(label or "").strip()

    # убрать служебные хвосты
    s = re.sub(
        r"\s*\((input|output|block|conv|pool|service|db|ui|api)\)\s*",
        "",
        s,
        flags=re.IGNORECASE,
    )

    # сначала длинные и специальные шаблоны
    replacements = [
        (r"\bweb\s*ui\b", "Веб-интерфейс"),
        (r"\bweb\b", "Веб-интерфейс"),
        (r"\bapi\s*v?\.?\s*(\d+)\b", r"Программный интерфейс \1"),
        (r"\bapi\b", "Программный интерфейс"),
        (r"\bui\b", "Интерфейс"),
        (r"\bdb\s+user\b", "База пользователей"),
        (r"\buser\s+db\b", "База пользователей"),
        (r"\bdb\s+material\b", "База материалов"),
        (r"\bmaterial\s+db\b", "База материалов"),
        (r"\bdb\s+result\b", "База результатов"),
        (r"\bresult\s+db\b", "База результатов"),
        (r"\bdb\b", "База данных"),
        (r"\bnotification\b", "Уведомления"),
        (r"\banalytics\b", "Аналитика"),
        (r"\bauth\b", "Аутентификация"),
        (r"\bdata\b", "Данные"),
        (r"\bservice\b", ""),
        (r"\bblock\b", ""),
        (r"\bconv\b", ""),
        (r"\binput\b", ""),
        (r"\boutput\b", ""),
        (r"\bpool\b", ""),
        (r"\bвнутр\.\b", "внутренний"),
    ]

    for pattern, repl in replacements:
        s = re.sub(pattern, repl, s, flags=re.IGNORECASE)

    # специальные нормализации
    s_low = s.lower().strip()

    if s_low in {"api", "программный интерфейс api"}:
        s = "Программный интерфейс"
    elif s_low in {"web", "веб", "внешний интерфейс"}:
        s = "Веб-интерфейс"
    elif s_low in {"api пользователя", "api админа", "api вход", "api управление"}:
        s = "Программный интерфейс"
    elif s_low in {"пользователи"}:
        s = "Пользователь"
    elif s_low in {"администраторы", "админ"}:
        s = "Администратор"

    # не допускать "Интерфейс Программный интерфейс"
    s = re.sub(
        r"(?i)\bинтерфейс\s+программный интерфейс\b",
        "Программный интерфейс",
        s,
    )

    # не допускать "Программный интерфейс пользователей"
    s = re.sub(
        r"(?i)\bпрограммный интерфейс\s+пользователей\b",
        "Программный интерфейс",
        s,
    )

    # косметика
    s = re.sub(r"\s+", " ", s).strip(" ,;:()[]-")
    s = s.replace("  ", " ")

    if not s:
        s = "Компонент"

    return s


def infer_general_kind_from_label(label: str, node_id: str = "") -> str:
    t = f"{node_id} {label}".lower()

    if any(x in t for x in [
        "пользователь", "админ", "администратор", "модератор",
        "клиент", "оператор", "внешние системы", "внешние сервисы",
        "интеграции"
    ]):
        return "input"

    if any(x in t for x in [
        "интерфейс", "веб", "api", "ui", "портал", "панель"
    ]):
        return "conv"

    if any(x in t for x in [
        "база", "хранилище", "лог", "телеметр", "архив"
    ]):
        return "output"

    return "block"


def dedupe_edges(edges: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    result: list[dict] = []

    for edge in edges or []:
        source = edge.get("source")
        target = edge.get("target")

        if not source or not target or source == target:
            continue

        key = (source, target)
        if key in seen:
            continue

        seen.add(key)
        new_edge = edge.copy()
        label = str(new_edge.get("label", "")).strip()

        if label in {"→", "->", "-->", "=>"}:
            label = ""

        new_edge["label"] = label
        result.append(new_edge)

    return result
