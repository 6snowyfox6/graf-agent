import os
import base64
import mimetypes
import json
import re
import time
from pathlib import Path
import requests
from typing import Any

from plotneuralnet_renderer import PlotNeuralNetRenderer, build_model_renderer
from infographic_renderer import InfographicRenderer
from graphviz import Digraph
from server_manager import ServerManager

# Локальные endpoints llama-cpp-python
# Qwen (генератор) на порту 8000, Gemma (критик/vision) на порту 8001
QWEN_URL = "http://127.0.0.1:8000/v1/chat/completions"
GEMMA_URL = "http://127.0.0.1:8001/v1/chat/completions"

GENERATOR_MODEL = "qwen2.5-7b-instruct"
VISION_MODEL = "gemma-3-4b-it"
CRITIC_MODEL = "gemma-3-4b-it"

MODEL_ENDPOINTS: dict[str, str] = {
    GENERATOR_MODEL: QWEN_URL,
    VISION_MODEL: GEMMA_URL,
    CRITIC_MODEL: GEMMA_URL,
}

def _get_endpoint(model: str) -> str:
    return MODEL_ENDPOINTS.get(model, QWEN_URL)

def load_user_prompt(default_prompt: str) -> str:
    test_prompt_path = Path("_test_prompt.txt")
    if test_prompt_path.exists():
        text = test_prompt_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return default_prompt

FORBIDDEN_VISIBLE_TOKENS = {
    "input", "output", "block", "conv", "ui", "api", "db", "service"
}


def clean_visible_label(label: str) -> str:
    s = str(label or "").strip()

    if not s:
        return "Компонент"

    # убрать служебные хвосты и мусор
    s = re.sub(
        r"\s*\((input|output|block|conv|pool|service|db|ui|api|database)\)\s*",
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = s.replace("_", " ")
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()

    # сначала многословные и самые важные шаблоны
    replacements = [
        # роли
        (r"\bsales\s+manager\b", "Менеджер продаж"),
        (r"\bteam\s+lead\b", "Руководитель команды"),
        (r"\bdata\s+engineer\b", "Инженер данных"),
        (r"\bsupport\s+agent\b", "Оператор поддержки"),

        (r"\bstudent\b", "Студент"),
        (r"\bteacher\b", "Преподаватель"),
        (r"\badministrator\b", "Администратор"),
        (r"\badmin\b", "Администратор"),
        (r"\bclient\b", "Клиент"),
        (r"\buser\b", "Пользователь"),
        (r"\boperator\b", "Оператор"),
        (r"\banalyst\b", "Аналитик"),
        (r"\bmanager\b", "Менеджер"),
        (r"\bengineer\b", "Инженер"),
        (r"\blead\b", "Руководитель"),

        # интерфейсы
        (r"\bweb\s+portal\b", "Веб-портал"),
        (r"\bmobile\s+app\b", "Мобильное приложение"),
        (r"\bweb\s+app\b", "Веб-приложение"),
        (r"\bweb\s+ui\b", "Веб-интерфейс"),
        (r"\bdashboard\b", "Панель"),
        (r"\bportal\b", "Портал"),
        (r"\bweb\b", "Веб-интерфейс"),
        (r"\bapi\s+gateway\b", "API-шлюз"),
        (r"\bgateway\b", "Шлюз"),
        (r"\bapi\s*v?\.?\s*(\d+)\b", r"API \1"),
        (r"\bapi\b", "API"),
        (r"\bui\b", "Интерфейс"),

        # сервисы
        (r"\bauth\s+service\b", "Сервис аутентификации"),
        (r"\bcourse\s+service\b", "Сервис курсов"),
        (r"\bnotification\s+service\b", "Сервис уведомлений"),
        (r"\border\s+service\b", "Сервис заказов"),
        (r"\bpayment\s+service\b", "Сервис платежей"),
        (r"\binventory\s+service\b", "Сервис склада"),
        (r"\bdelivery\s+service\b", "Сервис доставки"),
        (r"\bleads\s+service\b", "Сервис лидов"),
        (r"\bdeals\s+service\b", "Сервис сделок"),
        (r"\breports\s+service\b", "Сервис отчётов"),
        (r"\bticket\s+service\b", "Сервис заявок"),
        (r"\bassignment\s+service\b", "Сервис распределения"),
        (r"\bsla\s+monitor\b", "Контроль SLA"),
        (r"\banalytics\s+service\b", "Сервис аналитики"),
        (r"\bingestion\s+service\b", "Сервис загрузки"),
        (r"\bprocessing\s+service\b", "Сервис обработки"),
        (r"\bmodel\s+service\b", "Сервис моделей"),

        # данные и хранилища
        (r"\buser\s+db\b", "База пользователей"),
        (r"\busers\s+db\b", "База пользователей"),
        (r"\bcourse\s+db\b", "База курсов"),
        (r"\bcourses\s+db\b", "База курсов"),
        (r"\border[s]?\s+db\b", "База заказов"),
        (r"\bproduct[s]?\s+db\b", "База товаров"),
        (r"\bticket[s]?\s+db\b", "База заявок"),
        (r"\busers?\s+database\b", "База пользователей"),
        (r"\bcourses?\s+database\b", "База курсов"),
        (r"\border[s]?\s+database\b", "База заказов"),
        (r"\bproduct[s]?\s+database\b", "База товаров"),
        (r"\bticket[s]?\s+database\b", "База заявок"),
        (r"\bcrm\s+database\b", "База CRM"),
        (r"\bmetadata\s+db\b", "База метаданных"),
        (r"\bresults?\s+db\b", "База результатов"),
        (r"\bfeature\s+store\b", "Хранилище признаков"),
        (r"\bmetadata\b", "Метаданные"),
        (r"\bresults?\b", "Результаты"),
        (r"\bdatabase\b", "База данных"),
        (r"\bdb\b", "База данных"),

        # отдельные слова
        (r"\bnotifications?\b", "Уведомления"),
        (r"\banalytics?\b", "Аналитика"),
        (r"\bauth\b", "Аутентификация"),
        (r"\bpayment\b", "Платежи"),
        (r"\binventory\b", "Склад"),
        (r"\bdelivery\b", "Доставка"),
        (r"\border[s]?\b", "Заказы"),
        (r"\bproduct[s]?\b", "Товары"),
        (r"\bcourse[s]?\b", "Курсы"),
        (r"\bticket[s]?\b", "Заявки"),
        (r"\bleads\b", "Лиды"),
        (r"\bdeals\b", "Сделки"),
        (r"\breports?\b", "Отчёты"),
        (r"\bprocessing\b", "Обработка"),
        (r"\bingestion\b", "Загрузка"),
        (r"\bmonitoring\b", "Мониторинг"),
        (r"\bmodel\b", "Модель"),
        (r"\bservice\b", "Сервис"),

        # удалить служебные слова
        (r"\bblock\b", ""),
        (r"\bconv\b", ""),
        (r"\binput\b", ""),
        (r"\boutput\b", ""),
        (r"\bpool\b", ""),
    ]

    for pattern, repl in replacements:
        s = re.sub(pattern, repl, s, flags=re.IGNORECASE)

    # чинить гибриды вида "Course База данных"
    hybrid_db_patterns = [
        (r"(?i)\bкурсы?\s+база данных\b", "База курсов"),
        (r"(?i)\bзаказы?\s+база данных\b", "База заказов"),
        (r"(?i)\bтовары?\s+база данных\b", "База товаров"),
        (r"(?i)\bзаявки?\s+база данных\b", "База заявок"),
        (r"(?i)\bпользователи?\s+база данных\b", "База пользователей"),
        (r"(?i)\bметаданные\s+база данных\b", "База метаданных"),
        (r"(?i)\bрезультаты?\s+база данных\b", "База результатов"),
    ]
    for pattern, repl in hybrid_db_patterns:
        s = re.sub(pattern, repl, s)

    # чинить гибриды вида "API сервис", "Программный интерфейс сервис"
    s = re.sub(r"(?i)\bapi\s+сервис\b", "API", s)
    s = re.sub(r"(?i)\bинтерфейс\s+сервис\b", "Интерфейс", s)
    s = re.sub(r"(?i)\bшлюз\s+api\b", "API-шлюз", s)

    # убрать повтор слов
    s = re.sub(r"(?i)\b(сервис)\s+\1\b", r"\1", s)
    s = re.sub(r"(?i)\b(база данных)\s+\1\b", r"\1", s)
    s = re.sub(r"(?i)\b(интерфейс)\s+\1\b", r"\1", s)

    # косметика
    s = re.sub(r"\s+", " ", s).strip(" ,;:()[]-")
    s = s.replace("  ", " ")

    # если остался короткий английский хвост — не уродовать транслитом, а аккуратно оформить
    if re.fullmatch(r"[A-Za-z0-9 ]+", s):
        s = s.title()

    # слишком общие названия считаем плохими
    bad_generic = {
        "", "Component", "Компонент", "Service", "Сервис", "Database",
        "База Данных", "Data", "Данные", "Module", "Модуль", "Block"
    }
    if s in bad_generic:
        return "Компонент"

    return s


def infer_general_kind_from_label(label: str, node_id: str = "") -> str:
    t = f"{node_id} {label}".lower()

    if any(x in t for x in [
        "пользователь", "админ", "администратор", "модератор",
        "клиент", "оператор", "внешние системы", "внешние сервисы",
        "интеграции","student", "teacher", "analyst", "manager", "engineer", "lead"
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
    seen = set()
    result = []

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





def load_diagram_types(folder: str | None = None) -> list:
    if folder is None:
        folder = str(Path(__file__).resolve().parent / "diagram_types")
    path = Path(folder)
    configs = []

    if not path.exists():
        print(f"Папка не найдена: {folder}")
        return configs

    for file in path.glob("*.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                print(f"Пропускаю пустой файл: {file}")
                continue

            configs.append(json.loads(content))

        except json.JSONDecodeError as e:
            print(f"Пропускаю битый JSON-файл: {file}")
            print(f"Ошибка: {e}")
            continue

    return configs


DIAGRAM_CONFIGS = load_diagram_types()

if not DIAGRAM_CONFIGS:
    DIAGRAM_CONFIGS = [
        {
            "name": "general",
            "keywords": [],
            "system_prompt": (
                "Ты генератор диаграмм. "
                "Создавай логически правильные и визуально привлекательные схемы. "
                "Верни только валидный JSON без markdown и пояснений."
            ),
            "extra_rules": [
                "схема должна быть визуально чистой",
                "подписи должны быть короткими",
                "для обычных переходов label должен быть пустым"
            ],
            "layout_hint": "general",
            "renderer": "general"
        }
    ]





def ask_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: tuple[int, int] = (30, 300),
    temperature: float = 0.15
) -> str:
    endpoint = _get_endpoint(model)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }

    response = requests.post(endpoint, json=payload, timeout=timeout)
    if response.status_code >= 400:
        print(f"[LLM ERROR] {response.status_code}: {response.text}")
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def extract_json(text: str) -> dict:
    text = text.strip()

    decoder = json.JSONDecoder()

    # пробуем найти первый JSON-объект в тексте
    for start_idx, ch in enumerate(text):
        if ch != "{":
            continue

        try:
            obj, end_idx = decoder.raw_decode(text[start_idx:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    print("=== BAD JSON TEXT ===")
    print(text)
    print("=== END BAD JSON TEXT ===")
    raise ValueError("Не удалось извлечь JSON-объект из ответа модели")


def restore_node_kinds(original: dict, improved: dict) -> dict:
    original_kinds = {
        node["id"]: node.get("kind")
        for node in original.get("nodes", [])
        if "id" in node
    }

    for node in improved.get("nodes", []):
        if "kind" not in node:
            old_kind = original_kinds.get(node.get("id"))
            if old_kind:
                node["kind"] = old_kind

    return improved


def detect_diagram_mode(user_task: str, configs: list[dict]) -> str:
    text = user_task.lower()
    matches = []

    for config in configs:
        keywords = config.get("keywords", [])
        for word in keywords:
            idx = text.find(word.lower())
            if idx != -1:
                matches.append((idx, config["name"]))

    if matches:
        matches.sort(key=lambda x: x[0])
        return matches[0][1]

    return "general"


def get_diagram_config(mode: str, configs: list[dict]) -> dict:
    for config in configs:
        if config["name"] == mode:
            return config

    for config in configs:
        if config["name"] == "general":
            return config

    raise ValueError(f"Не найден конфиг для режима: {mode}")


def looks_like_general_request(user_task: str) -> bool:
    text = (user_task or "").lower()

    general_markers = [
        "архитектур", "архитектура", "система", "платформа", "сервис", "сервисы",
        "портал", "веб", "web", "ui", "api", "gateway", "дашборд", "dashboard",
        "база", "database", "db", "клиент", "пользователь", "администратор",
        "оператор", "менеджер", "analyst", "аналитик", "crm", "магазин",
        "shop", "store", "portal", "service"
    ]
    model_markers = [
        "unet", "u-net", "cnn", "resnet", "transformer", "encoder", "decoder",
        "bottleneck", "skip", "attention", "feature map", "featuremap",
        "нейросет", "сегментац", "слой", "embedding", "backbone", "neck",
        "head", "conv2d", "maxpool", "upsample"
    ]

    general_score = sum(1 for marker in general_markers if marker in text)
    model_score = sum(1 for marker in model_markers if marker in text)

    return general_score >= 2 and model_score == 0


def force_general_contract(diagram: dict, fallback: dict | None = None) -> dict:
    fixed = normalize_general_diagram(diagram, fallback=fallback or {})

    fixed["layout_hint"] = "general"
    fixed["renderer"] = "general"

    style = fixed.get("style")
    if not isinstance(style, dict):
        style = {}
    style.setdefault("direction", "TB")
    fixed["style"] = style

    for edge in fixed.get("edges", []):
        label = str(edge.get("label", "")).strip()
        if len(label) > 24:
            edge["label"] = ""

    return fixed


def get_node_level(node_id: str, label: str, kind: str | None = None) -> int:
    kind = str(kind or "").lower().strip()
    text = f"{node_id} {label}".lower()

    if kind == "input":
        if any(x in text for x in ["интерфейс", "веб", "панель", "портал", "api", "ui"]):
            return 1
        return 0

    if kind == "conv":
        return 1

    if kind == "block":
        return 2

    if kind == "output":
        return 3

    if any(x in text for x in ["пользователь", "модератор", "администратор", "оператор"]):
        return 0
    if any(x in text for x in ["интерфейс", "веб", "панель", "портал", "api", "ui"]):
        return 1
    if any(x in text for x in ["база", "хранилище", "телеметр", "лог", "данн"]):
        return 3

    return 2


def get_pipeline_lane(node_id: str, label: str) -> str:
    text = f"{node_id} {label}".lower()

    if any(x in text for x in [
        "user input", "query", "запрос", "query embedding", "query vectorization",
        "document retrieval", "retrieval", "поиск", "relevance ranking", "ranking",
        "ранжирование", "llm context", "language model", "llm",
        "final response", "answer", "ответ", "context", "контекст"
    ]):
        return "query_pipeline"

    if any(x in text for x in [
        "document", "text extraction", "извлечение текста",
        "text preprocessing", "cleaning", "очистка",
        "document segmentation", "chunking", "чанк",
        "embedding generation", "embeddings",
        "vector db", "vector database", "векторная бд", "index", "индексация"
    ]):
        return "document_pipeline"

    return "document_pipeline"


def auto_style_node(node: dict) -> dict:
    label = node.get("label", "").lower()
    styled = node.copy()

    if (
        "input" in label or "output" in label or "вход" in label
        or "выход" in label or "результат" in label
    ):
        styled.setdefault("shape", "ellipse")
        styled.setdefault("fillcolor", "#d9f2d9")
        styled.setdefault("color", "#4f7f4f")
        styled.setdefault("style", "filled,bold")
    elif (
        "backbone" in label or "encoder" in label or "decoder" in label
        or "transformer" in label or "darknet" in label
        or "resnet" in label or "block" in label or "neck" in label or "модуль" in label
    ):
        styled.setdefault("shape", "box")
        styled.setdefault("fillcolor", "#d9e8fb")
        styled.setdefault("color", "#4a6fa5")
        styled.setdefault("style", "filled,rounded")
    elif (
        "feature" in label or "embedding" in label or "эмбед" in label
        or "map" in label or "representation" in label or "features" in label
        or "database" in label or "бд" in label or "vector" in label
    ):
        styled.setdefault("shape", "cylinder")
        styled.setdefault("fillcolor", "#fbe5d6")
        styled.setdefault("color", "#a65e2e")
        styled.setdefault("style", "filled,bold")
    elif (
        "head" in label or "classifier" in label or "detector" in label
        or "segmentation" in label or "classification" in label
        or "классиф" in label or "детектор" in label
    ):
        styled.setdefault("shape", "box")
        styled.setdefault("fillcolor", "#e8dcf8")
        styled.setdefault("color", "#6b4fa3")
        styled.setdefault("style", "filled,rounded")
    elif (
        "ontology" in label or "онтолог" in label or "kg" in label or "graph" in label
    ):
        styled.setdefault("shape", "cylinder")
        styled.setdefault("fillcolor", "#fbe5d6")
        styled.setdefault("color", "#a65e2e")
        styled.setdefault("style", "filled,bold")
    else:
        styled.setdefault("shape", "box")
        styled.setdefault("fillcolor", "#d9e8fb")
        styled.setdefault("color", "#4a6fa5")
        styled.setdefault("style", "filled,rounded")

    return styled


def get_general_node_sort_key(node: dict) -> tuple[int, int, str]:
    label = str(node.get("label", "")).lower()
    kind = str(node.get("kind", "")).lower()

    if kind == "input":
        if "пользователь" in label:
            return (0, 0, label)
        if "админ" in label or "администратор" in label or "модератор" in label:
            return (0, 1, label)
        if "внеш" in label or "интеграц" in label:
            return (0, 2, label)
        return (0, 9, label)

    if kind == "conv":
        if "веб" in label:
            return (1, 0, label)
        if "программный интерфейс" in label:
            return (1, 1, label)
        return (1, 9, label)

    if kind == "block":
        if "обработ" in label:
            return (2, 0, label)
        if "аутентиф" in label:
            return (2, 1, label)
        if "анализ" in label or "аналит" in label:
            return (2, 2, label)
        if "уведом" in label:
            return (2, 3, label)
        if "логир" in label:
            return (2, 4, label)
        return (2, 9, label)

    if kind == "output":
        if "пользоват" in label:
            return (3, 0, label)
        if "материал" in label:
            return (3, 1, label)
        if "результ" in label:
            return (3, 2, label)
        if "аналит" in label:
            return (3, 3, label)
        if "лог" in label:
            return (3, 4, label)
        if "телеметр" in label:
            return (3, 5, label)
        return (3, 9, label)

    return (9, 9, label)





def score_general_candidate(diagram: dict) -> int:
    score = 100

    nodes = diagram.get("nodes", []) or []
    edges = diagram.get("edges", []) or []

    if not isinstance(nodes, list) or not isinstance(edges, list):
        return -1000

    node_ids = set()
    labels = []
    levels = []
    kinds_count = {"input": 0, "conv": 0, "block": 0, "output": 0}

    generic_bad = {
        "компонент", "узел", "сервис", "модуль", "данные",
        "база данных", "component", "module", "service", "data", "database"
    }

    actor_markers = [
        "студент", "преподаватель", "администратор", "клиент", "оператор",
        "аналитик", "менеджер", "инженер", "руководитель",
        "student", "teacher", "admin", "administrator", "client",
        "operator", "analyst", "manager", "engineer", "lead"
    ]

    db_markers = [
        "база", "хранилище", "database", "db", "store", "repository"
    ]

    ui_markers = [
        "веб", "портал", "интерфейс", "api", "шлюз", "панель",
        "web", "gateway", "dashboard", "ui"
    ]

    english_leftovers = [
        "gateway", "dashboard", "student", "teacher", "manager",
        "engineer", "database", "service", "inventory", "delivery",
        "orders", "products", "course", "courses", "report", "reports"
    ]

    if len(nodes) < 4:
        score -= 35
    elif len(nodes) > 14:
        score -= (len(nodes) - 14) * 5

    if len(edges) < 3:
        score -= 20
    elif len(edges) > 20:
        score -= (len(edges) - 20) * 3

    for node in nodes:
        if not isinstance(node, dict):
            score -= 20
            continue

        node_id = str(node.get("id", "")).strip()
        label = str(node.get("label", "")).strip()
        kind = str(node.get("kind", "")).strip().lower()

        if not node_id:
            score -= 20
            continue

        if node_id in node_ids:
            score -= 20
            continue
        node_ids.add(node_id)

        if not label:
            score -= 15
            continue

        lower = label.lower()
        labels.append(lower)

        if kind in kinds_count:
            kinds_count[kind] += 1
        else:
            score -= 10

        if any(tok in lower for tok in FORBIDDEN_VISIBLE_TOKENS):
            score -= 25

        if lower in generic_bad:
            score -= 30

        if len(label) > 30:
            score -= 8
        elif len(label) > 24:
            score -= 4

        if any(x in lower for x in english_leftovers):
            score -= 10

        if any(ch.isascii() and ch.isalpha() for ch in label) and any("а" <= ch.lower() <= "я" for ch in label):
            score -= 8

        level = get_node_level(node_id, label, kind)
        levels.append(level)

        if kind == "input" and any(x in lower for x in db_markers):
            score -= 18
        if kind == "output" and any(x in lower for x in actor_markers):
            score -= 18
        if kind == "conv" and any(x in lower for x in db_markers):
            score -= 14
        if kind == "block" and any(x in lower for x in actor_markers):
            score -= 14
        if kind == "output" and any(x in lower for x in ui_markers):
            score -= 12

    # хотим видеть все 4 слоя хотя бы в базовом виде
    if kinds_count["input"] == 0:
        score -= 25
    if kinds_count["conv"] == 0:
        score -= 20
    if kinds_count["block"] == 0:
        score -= 25
    if kinds_count["output"] == 0:
        score -= 20

    if kinds_count["input"] >= 2:
        score += 6
    if kinds_count["output"] >= 2:
        score += 4

    in_deg = {nid: 0 for nid in node_ids}
    out_deg = {nid: 0 for nid in node_ids}

    seen_edges = set()

    for edge in edges:
        if not isinstance(edge, dict):
            score -= 8
            continue

        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        label = str(edge.get("label", "")).strip()

        if not source or not target:
            score -= 12
            continue

        if source == target:
            score -= 18
            continue

        if source not in node_ids or target not in node_ids:
            score -= 18
            continue

        key = (source, target)
        if key in seen_edges:
            score -= 10
            continue
        seen_edges.add(key)

        in_deg[target] += 1
        out_deg[source] += 1

        src_node = next((n for n in nodes if n.get("id") == source), None)
        dst_node = next((n for n in nodes if n.get("id") == target), None)
        if src_node and dst_node:
            src_level = get_node_level(source, src_node.get("label", ""), src_node.get("kind"))
            dst_level = get_node_level(target, dst_node.get("label", ""), dst_node.get("kind"))

            if dst_level < src_level:
                score -= 6
            if abs(dst_level - src_level) > 2:
                score -= 6

        if label and len(label) > 18:
            score -= 5

    # штраф за изолированные и "центральные хабы"
    isolated = 0
    heavy_hubs = 0
    for nid in node_ids:
        deg = in_deg[nid] + out_deg[nid]
        if deg == 0:
            isolated += 1
        if deg >= 5:
            heavy_hubs += 1

    score -= isolated * 12
    score -= heavy_hubs * 8

    # бонус за более ровную уровневую структуру
    if levels:
        unique_levels = len(set(levels))
        if unique_levels >= 3:
            score += 6
        if unique_levels == 4:
            score += 6

    return int(score)


def render_general_diagram(diagram: dict, output_name: str = "final_diagram") -> str:
    def chunk_nodes(items, size):
        if size <= 0:
            size = 3
        return [items[i:i + size] for i in range(0, len(items), size)]

    def apply_default_style(node: dict) -> dict:
        node = dict(node)
        kind = str(node.get("kind", "block")).lower().strip()

        style_map = {
            "input": {
                "shape": "ellipse",
                "style": "filled",
                "fillcolor": "#d9ead3",
                "color": "#4f7f4f",
            },
            "conv": {
                "shape": "box",
                "style": "rounded,filled",
                "fillcolor": "#d9e7f7",
                "color": "#5a84c9",
            },
            "block": {
                "shape": "box",
                "style": "rounded,filled",
                "fillcolor": "#d9e7f7",
                "color": "#5a84c9",
            },
            "output": {
                "shape": "ellipse",
                "style": "filled",
                "fillcolor": "#f4cccc",
                "color": "#b45f06",
            },
            "pool": {
                "shape": "ellipse",
                "style": "filled",
                "fillcolor": "#f4cccc",
                "color": "#b45f06",
            },
        }

        base = style_map.get(kind, style_map["block"])
        for k, v in base.items():
            node.setdefault(k, v)

        return node

    def normalize_edge_attrs(edge: dict, src_level: int, dst_level: int, layout_hint: str) -> dict:
        raw_style = edge.get("style", "solid")

        edge_kwargs = {
            "color": "#555555",
            "style": "solid",
            "penwidth": "1.1",
        }

        raw_color = edge.get("color")
        if isinstance(raw_color, str) and raw_color.strip():
            edge_kwargs["color"] = raw_color.strip()

        if isinstance(raw_style, str) and raw_style.strip():
            edge_kwargs["style"] = raw_style.strip()

        if isinstance(raw_style, dict):
            arrowhead = raw_style.get("arrowhead")
            if isinstance(arrowhead, str) and arrowhead.strip():
                if arrowhead.strip().lower() == "triangle":
                    edge_kwargs["arrowhead"] = "normal"
                else:
                    edge_kwargs["arrowhead"] = arrowhead.strip()

            line_style = raw_style.get("line_style")
            if isinstance(line_style, str) and line_style.strip():
                edge_kwargs["style"] = line_style.strip()

            color = raw_style.get("color")
            if isinstance(color, str) and color.strip():
                edge_kwargs["color"] = color.strip()

            penwidth = raw_style.get("penwidth")
            if penwidth is not None:
                edge_kwargs["penwidth"] = str(penwidth)

        label = edge.get("label", "")
        if not isinstance(label, str):
            label = ""
        label = label.strip()

        if label in {"", "->", "-->", "=>", "→"}:
            label = ""

        if len(label) > 22:
            label = ""

        if layout_hint == "general":
            label = ""

        if label:
            edge_kwargs["label"] = label

        if abs(dst_level - src_level) > 1:
            edge_kwargs["constraint"] = "false"
            edge_kwargs["color"] = "#777777"
            edge_kwargs["penwidth"] = "0.9"
            edge_kwargs["minlen"] = "2"

        return edge_kwargs

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    dot = Digraph(comment=diagram.get("title", "Diagram"))
    dot.attr(
        rankdir="TB",
        splines="polyline",
        newrank="true",
        overlap="false",
        nodesep="0.24",
        ranksep="0.55",
        pad="0.20",
        bgcolor="white",
    )

    dot.attr(
        "node",
        fontname="Arial",
        fontsize="12",
        margin="0.14,0.08",
        penwidth="1.2",
    )
    dot.attr(
        "edge",
        fontname="Arial",
        fontsize="10",
        arrowsize="0.7",
    )

    styled_nodes = [apply_default_style(n) for n in diagram.get("nodes", [])]

    node_levels = {}
    level_groups = {}

    for node in styled_nodes:
        node_id = node["id"]
        label = node.get("label", node_id)
        kind = node.get("kind")
        level = get_node_level(node_id, label, kind)

        node_levels[node_id] = level
        level_groups.setdefault(level, []).append(node)

    for level in level_groups:
        level_groups[level] = sorted(
            level_groups[level], key=get_general_node_sort_key)

    max_per_row_by_level = {
        0: 2,
        1: 2,
        2: 3,
        3: 3,
    }

    sorted_levels = sorted(level_groups.keys())

    for level in sorted_levels:
        max_per_row = max_per_row_by_level.get(level, 3)
        rows = chunk_nodes(level_groups[level], max_per_row)

        for row_idx, row in enumerate(rows):
            with dot.subgraph(name=f"cluster_level_{level}_row_{row_idx}") as sub:
                sub.attr(rank="same")
                sub.attr(color="transparent")
                sub.attr(penwidth="0")

                previous_id = None
                for node in row:
                    sub.node(
                        node["id"],
                        node.get("label", node["id"]),
                        shape=node.get("shape", "box"),
                        style=node.get("style", "rounded,filled"),
                        fillcolor=node.get("fillcolor", "#d9e7f7"),
                        color=node.get("color", "#5a84c9"),
                    )

                    if previous_id is not None:
                        sub.edge(
                            previous_id,
                            node["id"],
                            style="invis",
                            weight="10",
                        )
                    previous_id = node["id"]

    for edge in diagram.get("edges", []):
        src = edge.get("source")
        dst = edge.get("target")

        if not src or not dst or src == dst:
            continue

        src_level = node_levels.get(src, 2)
        dst_level = node_levels.get(dst, 2)

        edge_kwargs = normalize_edge_attrs(
            edge=edge,
            src_level=src_level,
            dst_level=dst_level,
            layout_hint=diagram.get("layout_hint", "general"),
        )

        dot.edge(src, dst, **edge_kwargs)

    dot.filename = f"{output_name}.gv"
    dot.directory = str(output_dir)

    png_path = dot.render(format="png", cleanup=True)
    print(f"Схема сохранена: {Path(png_path).name}")

    svg_path = dot.render(format="svg", cleanup=True)
    print(f"Схема сохранена: {Path(svg_path).name}")

    pdf_path = dot.render(format="pdf", cleanup=True)
    print(f"Схема сохранена: {Path(pdf_path).name}")

    return png_path


def render_pipeline_diagram(diagram: dict, output_name: str = "diagram"):
    from graphviz import Digraph

    dot = Digraph(comment=diagram.get("title", "Pipeline"))
    dot.attr(
        rankdir="TB",
        splines="spline",
        overlap="false",
        nodesep="0.35",
        ranksep="0.45",
        bgcolor="white",
    )

    dot.attr(
        "node",
        shape="box",
        style="filled,rounded",
        fillcolor="#d9e8fb",
        color="#4a6fa5",
        fontname="Arial",
        fontsize="13",
        margin="0.12,0.08",
    )

    dot.attr(
        "edge",
        color="#555555",
        fontname="Arial",
        fontsize="10",
        arrowsize="0.7",
    )

    styled_nodes = [auto_style_node(node) for node in diagram.get("nodes", [])]

    lanes = diagram.get("lanes", [])
    lane_map: dict[str, str] = {}
    if lanes:
        for lane in lanes:
            lane_name = lane.get("name", "lane")
            for node_id in lane.get("nodes", []):
                lane_map[node_id] = lane_name
    else:
        for node in styled_nodes:
            lane_map[node["id"]] = get_pipeline_lane(node["id"], node["label"])

    lane_order = ["query_pipeline", "document_pipeline"]
    grouped = {name: [] for name in lane_order}
    for node in styled_nodes:
        lane = lane_map.get(node["id"], "document_pipeline")
        if lane not in grouped:
            grouped[lane] = []
        grouped[lane].append(node)

    for lane_name in lane_order:
        if not grouped.get(lane_name):
            continue
        with dot.subgraph() as s:
            s.attr(rank="same")
            for node in grouped[lane_name]:
                s.node(
                    node["id"],
                    node["label"],
                    shape=node.get("shape", "box"),
                    style=node.get("style", "filled,rounded"),
                    fillcolor=node.get("fillcolor", "#d9e8fb"),
                    color=node.get("color", "#4a6fa5"),
                )

    query_order = [
        "user_input",
        "query_vectorization",
        "document_retrieval",
        "relevance_ranking",
        "llm_context",
        "language_model",
        "final_response",
    ]
    document_order = [
        "document",
        "text_extraction",
        "text_preprocessing",
        "document_segmentation",
        "embedding_generation",
        "vector_database",
    ]

    def add_invisible_order_edges(node_order: list[str], existing_ids: set[str]):
        filtered = [n for n in node_order if n in existing_ids]
        for a, b in zip(filtered, filtered[1:]):
            dot.edge(a, b, style="invis", weight="10")

    existing_ids = {node["id"] for node in styled_nodes}
    add_invisible_order_edges(query_order, existing_ids)
    add_invisible_order_edges(document_order, existing_ids)

    for edge in diagram.get("edges", []):
        raw_label = edge.get("label", "")
        label = raw_label.strip() if isinstance(raw_label, str) else ""

        if label in {"→", "->", "-->", "=>", ""}:
            label = ""
        if len(label) > 18:
            label = ""

        edge_kwargs = {
            "color": edge.get("color", "#555555"),
            "style": edge.get("style", "solid"),
            "penwidth": "1.2",
        }

        source_lane = lane_map.get(edge["source"], "document_pipeline")
        target_lane = lane_map.get(edge["target"], "document_pipeline")
        cross_lane = source_lane != target_lane
        if cross_lane:
            edge_kwargs["constraint"] = "false"
            edge_kwargs["color"] = "#666666"
            edge_kwargs["penwidth"] = "1.0"

        if label:
            edge_kwargs["label"] = label

        dot.edge(edge["source"], edge["target"], **edge_kwargs)

    dot.render(output_name, format="png", cleanup=True)
    dot.render(output_name, format="svg", cleanup=True)
    dot.render(output_name, format="pdf", cleanup=True)

    print(f"Схема сохранена: {output_name}.png")
    print(f"Схема сохранена: {output_name}.svg")
    print(f"Схема сохранена: {output_name}.pdf")


def load_references_for_mode(mode: str, base_folder: str = "references") -> list[dict]:
    refs: list[dict] = []

    base_path = Path(base_folder)
    shared_path = base_path / "shared"
    mode_path = base_path / mode

    folders_to_scan = [shared_path, mode_path]

    for folder in folders_to_scan:
        if not folder.exists() or not folder.is_dir():
            continue

        for file in sorted(folder.iterdir()):
            if not file.is_file():
                continue

            suffix = file.suffix.lower()

            if suffix == ".txt":
                content = file.read_text(encoding="utf-8").strip()
                if content:
                    refs.append({
                        "type": "text",
                        "name": file.stem,
                        "content": content
                    })

            elif suffix == ".json":
                try:
                    raw = file.read_text(encoding="utf-8").strip()
                    if not raw:
                        continue

                    content = json.loads(raw)
                    refs.append({
                        "type": "json",
                        "name": file.stem,
                        "content": content
                    })
                except json.JSONDecodeError:
                    print(f"Пропускаю битый JSON-референс: {file}")

    return refs


def normalize_references(references: list[dict] | None) -> list[dict]:
    normalized: list[dict] = []

    for ref in references or []:
        ref_type = ref.get("type", "text")
        name = ref.get("name", "reference")
        content = ref.get("content", "")

        if ref_type == "json" and isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False, indent=2)

        elif ref_type == "text":
            content = str(content)

        else:
            content = str(content)

        normalized.append({
            "type": ref_type,
            "name": name,
            "content": content
        })

    return normalized


def merge_reference_sources(
    mode: str,
    reference_description: dict | str | None = None,
    base_folder: str = "references"
) -> list[dict]:
    folder_refs = load_references_for_mode(mode, base_folder=base_folder)

    direct_refs: list[dict] = []
    if reference_description:
        if isinstance(reference_description, dict):
            direct_refs.append({
                "type": "json",
                "name": "direct_reference",
                "content": reference_description
            })
        else:
            direct_refs.append({
                "type": "text",
                "name": "direct_reference",
                "content": str(reference_description)
            })

    return normalize_references(folder_refs + direct_refs)


def format_references_for_prompt(references: list[dict]) -> str:
    if not references:
        return "Референсы не заданы."

    parts: list[str] = []

    for i, ref in enumerate(references, start=1):
        ref_type = ref["type"]
        name = ref["name"]
        content = ref["content"]

        if ref_type == "text":
            parts.append(
                f"Референс {i} (text, {name}):\n{content}"
            )

        elif ref_type == "json":
            parts.append(
                f"Референс {i} (json, {name}):\n{content}"
            )

        else:
            parts.append(
                f"Референс {i} ({ref_type}, {name}):\n{content}"
            )

    return "\n\n".join(parts)


def render_diagram(diagram: dict, output_name: str = "diagram"):
    renderer = diagram.get("renderer", "")
    layout_hint = diagram.get("layout_hint", "")

    if renderer == "general" or layout_hint == "general":
        return render_general_diagram(diagram, output_name)

    if renderer == "plotneuralnet" or layout_hint == "model_architecture":
        plot_renderer = build_model_renderer(diagram, project_root=".")
        return plot_renderer.render(diagram, output_name=output_name)

    if renderer == "infographic" or layout_hint == "infographic":
        info_renderer = InfographicRenderer()
        return info_renderer.render(diagram, output_name=output_name)

    if renderer == "pipeline" or layout_hint == "pipeline":
        return render_pipeline_diagram(diagram, output_name)

    return render_general_diagram(diagram, output_name)


def critique_diagram(user_task: str, draft_json: dict, references: list[dict] | None = None) -> dict:
    references = normalize_references(references or [])
    refs_text = format_references_for_prompt(references)

    system_prompt = (
        "Ты критик диаграмм. "
        "Оцени соответствие пользовательскому запросу и качество структуры. "
        "Но не критикуй и не предлагай удалять служебные поля внутреннего формата, "
        "если они нужны пайплайну рендера. "
        "К таким полям относятся title, renderer, layout_hint, style, nodes[].kind и другие системные поля. "
        "Ты можешь критиковать только смысл схемы, состав элементов, связи, читаемость и визуальную организацию. "
        "Нельзя предлагать менять JSON-контракт проекта."
    )

    user_prompt = f"""
Референсы:
{refs_text}

Запрос пользователя:
{user_task}

Черновая схема:
{json.dumps(draft_json, ensure_ascii=False, indent=2)}

Сделай проверку в таком порядке:

1. Выдели из запроса пользователя ключевые требования:
   - тип диаграммы
   - обязательные сущности / блоки
   - обязательные связи / поток
   - ограничения по стилю / компактности / читаемости
   - важные подписи / входы / выходы

2. Сравни требования с черновой схемой.

3. Найди:
   - что отсутствует
   - что добавлено лишнее
   - что интерпретировано неверно
   - что мешает читаемости

Верни JSON строго такого вида:
{{
  "score": 0.0,
  "task_fit_score": 0.0,
  "visual_score": 0.0,
  "missing_requirements": ["string"],
  "wrong_interpretations": ["string"],
  "extra_elements": ["string"],
  "visual_problems": ["string"],
  "problems": ["string"],
  "fixes": ["string"]
}}
"""
    raw_answer = ask_llm(CRITIC_MODEL, system_prompt, user_prompt)
    return extract_json(raw_answer)


def _balance_parentheses(text: str) -> str:
    opened = text.count('(')
    closed = text.count(')')
    if opened > closed:
        return text + ')' * (opened - closed)
    return text

def _recursive_clean_strings(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _recursive_clean_strings(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_recursive_clean_strings(v) for v in data]
    elif isinstance(data, str):
        return _balance_parentheses(data)
    return data

def clean_diagram_labels(diagram: dict) -> dict:
    cleaned = _recursive_clean_strings(diagram)

    bad_labels = {"→", "->", "-->", "=>", ""}
    if "edges" in cleaned:
        new_edges = []
        for edge in cleaned.get("edges", []):
            new_edge = edge.copy()
            label = str(new_edge.get("label", "")).strip()
            if label in bad_labels:
                new_edge["label"] = ""
            new_edges.append(new_edge)
        cleaned["edges"] = new_edges

    return cleaned


def improve_diagram(
    user_task: str,
    draft_json: dict,
    critique_json: dict,
    references: list[dict] | None = None
) -> dict:
    references = normalize_references(references or [])
    refs_text = format_references_for_prompt(references)

    system_prompt = (
        "Ты исправляешь JSON диаграммы. "
        "Главное правило: сохраняй совместимость с внутренним форматом проекта. "
        "Нельзя придумывать новые поля и новые значения перечислений. "
        "Для general-диаграмм допустимые значения nodes[].kind: input, conv, block, output. "
        "Нельзя заменять их на actor, ui, service, database и другие значения. "
        "Если хочешь отразить акторов, UI, сервисы и базы, делай это через существующие допустимые kind: "
        "actor -> input, ui -> conv, service -> block, database -> output. "
        "Сохраняй существующие корректные поля: title, layout_hint, renderer, style, nodes[].kind. "
        "Не удаляй поля, если они уже есть и корректны. "
        "Исправляй только то, что действительно нужно исправить. "
        "Верни только валидный JSON."
        "Все видимые подписи и title должны быть только на русском языке. "
        "Служебные kind могут оставаться внутренними: input, conv, block, output. "
        "Но label нельзя писать как input, output, block, conv, UI, API, DB. "
    )

    user_prompt = f"""
        Запрос пользователя:
        {user_task}

        Черновой JSON:
        {json.dumps(draft_json, ensure_ascii=False, indent=2)}

        Замечания критика:
        {json.dumps(critique_json, ensure_ascii=False, indent=2)}

        Важно:
        - сохрани title, layout_hint, renderer, style, если они уже есть;
        - все видимые label и title должны быть только на русском языке;
        - служебные kind оставь внутренними, но не показывай их в label;
        - не удаляй существующие корректные поля;
        - для general-диаграмм допустимые kind только: input, conv, block, output;
        - не используй actor, ui, service, database;
        - если критик пишет про actors/services/databases, отрази это через существующие kind:
        input = actors
        conv = UI
        block = services
        output = databases

        Верни только исправленный JSON.
    """
    raw_answer = ask_llm(GENERATOR_MODEL, system_prompt, user_prompt)

    print("\n=== RAW ANSWER FROM IMPROVER ===")
    print(raw_answer)
    print("=== END RAW ANSWER FROM IMPROVER ===\n")

    try:
        improved = extract_json(raw_answer)
        improved = clean_diagram_labels(improved)
        improved = restore_node_kinds(draft_json, improved)

        should_force_general = (
            draft_json.get("renderer") == "general"
            or draft_json.get("layout_hint") == "general"
            or looks_like_general_request(user_task)
        )

        if should_force_general:
            improved = force_general_contract(improved, fallback=draft_json)
        else:
            improved = normalize_general_diagram(improved, fallback=draft_json)

        return improved
    except Exception:
        print("Ошибка парсинга improve-ответа. Возвращаю draft_json.")
        return draft_json


def image_to_data_url(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/png"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def normalize_general_diagram(diagram: dict, fallback: dict | None = None) -> dict:
    if fallback is None:
        fallback = {}

    result = dict(diagram)

    for key in ["title", "layout_hint", "renderer", "style"]:
        if key not in result and key in fallback:
            result[key] = fallback[key]

    result.setdefault("title", "Архитектура системы")
    result["title"] = clean_visible_label(result.get("title", "Архитектура системы"))

    result["layout_hint"] = "general"
    result["renderer"] = "general"

    style = result.get("style")
    if not isinstance(style, dict):
        style = {}
    style.setdefault("direction", "TB")
    result["style"] = style

    kind_map = {
        "actor": "input",
        "user": "input",
        "external": "input",
        "input": "input",

        "ui": "conv",
        "interface": "conv",
        "frontend": "conv",
        "api": "conv",
        "conv": "conv",

        "service": "block",
        "module": "block",
        "processor": "block",
        "block": "block",

        "database": "output",
        "db": "output",
        "storage": "output",
        "repository": "output",
        "output": "output",
    }

    fallback_nodes = {}
    for node in fallback.get("nodes", []):
        node_id = str(node.get("id", "")).strip()
        if node_id:
            fallback_nodes[node_id] = dict(node)

    def build_label_from_id(node_id: str) -> str:
        text = str(node_id or "").strip().replace("_", " ").replace("-", " ")
        text = re.sub(r"\s+", " ", text).strip()
        text = clean_visible_label(text)
        return text if text != "Компонент" else "Узел"

    new_nodes = []
    seen_ids = set()

    for raw_node in result.get("nodes", []):
        node = dict(raw_node)

        node_id = str(node.get("id", "")).strip()
        if not node_id or node_id in seen_ids:
            continue
        seen_ids.add(node_id)

        raw_label = node.get("label", node_id)
        cleaned_label = clean_visible_label(raw_label)

        # если модель дала слишком общий label — пытаемся спасти его из fallback/id
        if cleaned_label == "Компонент":
            fb = fallback_nodes.get(node_id, {})
            fb_label = clean_visible_label(fb.get("label", ""))
            if fb_label and fb_label != "Компонент":
                cleaned_label = fb_label
            else:
                cleaned_label = build_label_from_id(node_id)

        raw_kind = str(node.get("kind", "")).strip().lower()
        if raw_kind in kind_map:
            kind = kind_map[raw_kind]
        else:
            fb_kind = str(fallback_nodes.get(node_id, {}).get("kind", "")).strip().lower()
            if fb_kind in {"input", "conv", "block", "output"}:
                kind = fb_kind
            else:
                kind = infer_general_kind_from_label(cleaned_label, node_id)

        # если актор случайно стал базой/сервисом — подправляем
        actor_markers = [
            "студент", "преподаватель", "администратор", "клиент", "оператор",
            "аналитик", "менеджер", "инженер", "руководитель"
        ]
        if any(x in cleaned_label.lower() for x in actor_markers):
            kind = "input"

        if "база" in cleaned_label.lower() or "хранилище" in cleaned_label.lower():
            kind = "output"

        if any(x in cleaned_label.lower() for x in ["веб", "портал", "интерфейс", "api", "шлюз", "панель"]):
            if kind != "input":
                kind = "conv"

        new_nodes.append({
            **node,
            "id": node_id,
            "label": cleaned_label,
            "kind": kind,
        })

    result["nodes"] = new_nodes

    valid_ids = {n["id"] for n in new_nodes}
    cleaned_edges = []

    for edge in dedupe_edges(result.get("edges", [])):
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()

        if not source or not target or source == target:
            continue
        if source not in valid_ids or target not in valid_ids:
            continue

        label = str(edge.get("label", "")).strip()
        if label in {"", "->", "-->", "=>", "→"}:
            label = ""

        if len(label) > 18:
            label = ""

        cleaned_edges.append({
            "source": source,
            "target": target,
            "label": label,
            **{k: v for k, v in edge.items() if k not in {"source", "target", "label"}},
        })

    result["edges"] = cleaned_edges

    return result

def analyze_reference_image(image_path: str) -> dict:
    image_data_url = image_to_data_url(image_path)

    system_prompt = (
        "Ты анализируешь PNG-референс диаграммы. "
        "Определи тип диаграммы, визуальный стиль, направление, форму узлов и общую структуру. "
        "Верни только валидный JSON без markdown и пояснений."
    )

    user_text = """
Посмотри на PNG-референс и верни JSON строго такого вида:
{
  "reference_type": "flowchart|chart|diagram|unknown",
  "style": {
    "direction": "TB|LR|unknown",
    "theme": "clean|dense|academic|minimal|unknown",
    "node_shape": "ellipse|box|mixed|unknown",
    "label_style": "short|medium|long|unknown"
  },
  "observations": [
    "string"
  ],
  "guidance_for_generation": [
    "string"
  ]
}
"""

    payload = {
        "model": VISION_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        "temperature": 0.2,
    }

    response = requests.post(_get_endpoint(VISION_MODEL), json=payload, timeout=(30, 180))
    print("STATUS:", response.status_code)
    print("RESPONSE TEXT:")
    print(response.text)
    response.raise_for_status()

    raw_answer = response.json()["choices"][0]["message"]["content"]
    print("\n=== RAW REFERENCE ANALYSIS ===")
    print(raw_answer)
    print("=== END RAW REFERENCE ANALYSIS ===\n")
    return extract_json(raw_answer)


def generate_diagram(user_task: str, reference_description: dict | str | None = None) -> dict:
    def looks_like_general_request(text: str) -> bool:
        t = (text or "").lower()

        general_markers = [
            "архитектур", "система", "платформ", "сервис", "портал",
            "api", "gateway", "веб", "web", "база", "db", "database",
            "клиент", "пользователь", "администратор", "оператор",
            "аналитик", "менеджер", "инженер", "actor", "актор",
            "компонент", "компоненты"
        ]
        model_markers = [
            "unet", "u-net", "cnn", "resnet", "transformer", "encoder",
            "decoder", "bottleneck", "skip", "attention", "слой",
            "нейросет", "сегментац", "feature map", "plotneuralnet"
        ]

        g = sum(1 for x in general_markers if x in t)
        m = sum(1 for x in model_markers if x in t)

        return g >= 2 and m == 0

    def normalize_term_for_match(text: str) -> str:
        s = clean_visible_label(text)
        s = s.lower()
        s = s.replace("ё", "е")
        s = re.sub(r"[^a-zа-я0-9 ]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def parse_named_section(text: str, section_name: str) -> list[str]:
        """
        Ищет фрагменты вида:
        'Акторы: A, B, C. Компоненты: X, Y, Z.'
        """
        pattern = rf"(?is){section_name}\s*:\s*(.+?)(?:\n[A-ЯA-Z][^:\n]{{0,40}}:|$)"
        m = re.search(pattern, text)
        if not m:
            return []

        chunk = m.group(1)
        chunk = chunk.replace("\n", " ")
        chunk = re.split(r"[.;]", chunk)[0]

        parts = [p.strip() for p in re.split(r",|/|\|", chunk) if p.strip()]
        result = []

        for p in parts:
            p = p.strip(" -")
            if not p:
                continue
            if len(p) > 60:
                continue
            result.append(p)

        return result

    def extract_expected_entities(task_text: str) -> dict[str, list[str]]:
        text = task_text or ""

        actors = []
        components = []
        storages = []

        actors += parse_named_section(text, "Акторы")
        actors += parse_named_section(text, "Actors")

        components += parse_named_section(text, "Компоненты")
        components += parse_named_section(text, "Components")
        components += parse_named_section(text, "Основные блоки")
        components += parse_named_section(text, "Блоки")

        # если база явно перечислена среди компонентов, дополнительно отнесём её в storages
        for item in components:
            low = item.lower()
            if any(x in low for x in [" db", "db", "database", "база", "хранилище", "store"]):
                storages.append(item)

        # fallback: грубая эвристика по роли в тексте
        role_words = [
            "студент", "преподаватель", "администратор", "клиент", "оператор",
            "аналитик", "менеджер", "инженер", "руководитель",
            "student", "teacher", "administrator", "admin",
            "client", "operator", "analyst", "manager", "engineer", "lead"
        ]
        for role in role_words:
            if re.search(rf"(?i)\b{re.escape(role)}\b", text):
                actors.append(role)

        actors = list(dict.fromkeys([a for a in actors if a]))
        components = list(dict.fromkeys([c for c in components if c]))
        storages = list(dict.fromkeys([s for s in storages if s]))

        return {
            "actors": actors,
            "components": components,
            "storages": storages,
        }

    def candidate_task_fit_score(candidate: dict, task_text: str) -> int:
        """
        Добавочный task-aware score поверх score_general_candidate().
        """
        expected = extract_expected_entities(task_text)

        labels = [
            normalize_term_for_match(n.get("label", ""))
            for n in candidate.get("nodes", [])
            if isinstance(n, dict)
        ]
        joined = " | ".join(labels)

        score = 0

        # 1) покрытие акторов
        actor_hits = 0
        for actor in expected["actors"]:
            norm_actor = normalize_term_for_match(actor)
            if not norm_actor:
                continue
            if norm_actor in joined:
                actor_hits += 1
                score += 14
            else:
                score -= 16

        # 2) покрытие компонентов
        component_hits = 0
        for comp in expected["components"]:
            norm_comp = normalize_term_for_match(comp)
            if not norm_comp:
                continue
            if norm_comp in joined:
                component_hits += 1
                score += 8
            else:
                score -= 8

        # 3) бонус за достаточное число входных ролей
        input_count = sum(
            1 for n in candidate.get("nodes", [])
            if isinstance(n, dict) and str(n.get("kind", "")).lower() == "input"
        )
        if len(expected["actors"]) >= 2 and input_count < 2:
            score -= 25
        elif len(expected["actors"]) >= 2 and input_count >= 2:
            score += 8

        # 4) штраф за generic labels
        bad_generic = {"компонент", "узел", "сервис", "модуль", "данные", "база данных"}
        for node in candidate.get("nodes", []):
            if not isinstance(node, dict):
                continue
            label = normalize_term_for_match(node.get("label", ""))
            if label in bad_generic:
                score -= 25

        # 5) штраф за недопокрытие вообще
        if expected["actors"] and actor_hits == 0:
            score -= 30
        if expected["components"] and component_hits < max(1, len(expected["components"]) // 3):
            score -= 20

        return score

    configs = DIAGRAM_CONFIGS or load_diagram_types()

    mode = detect_diagram_mode(user_task, configs)
    if looks_like_general_request(user_task):
        mode = "general"

    config = get_diagram_config(mode, configs)

    system_prompt = config["system_prompt"]
    layout_hint = config.get("layout_hint", "general")
    extra_rules = "\n".join(f"- {rule}" for rule in config.get("extra_rules", []))

    references = merge_reference_sources(mode, reference_description)
    reference_text = format_references_for_prompt(references)

    custom_format = config.get("json_format", "")
    if custom_format:
        json_format_block = custom_format
    else:
        json_format_block = (
            '{\n'
            '  "title": "string",\n'
            f'  "layout_hint": "{layout_hint}",\n'
            f'  "renderer": "{config.get("renderer", "general")}",\n'
            '  "layout": "linear|u_shape",\n'
            '  "nodes": [\n'
            '    {\n'
            '      "id": "string",\n'
            '      "label": "string",\n'
            '      "kind": "input|conv|pool|block|fc|sum|concat|mul|output"\n'
            '    }\n'
            '  ],\n'
            '  "edges": [\n'
            '    {\n'
            '      "source": "string",\n'
            '      "target": "string"\n'
            '    }\n'
            '  ]\n'
            '}'
        )

    general_override = looks_like_general_request(user_task)

    user_prompt = f"""
Референсы:
{reference_text}

Запрос пользователя:
{user_task}

Дополнительные требования:
{extra_rules}

Важно:
- все видимые подписи и title должны быть только на русском языке;
- не используй в label слова input, output, block, conv, ui, api, db, service;
- kind может оставаться служебным внутренним полем;
- не делай лишних связей;
- не делай один центральный хаб без необходимости;
- для general-схемы предпочитай уровни: верхний слой -> интерфейсы -> сервисы -> хранилища;
- если запрос похож на архитектуру системы, результат обязан быть general-диаграммой;
- не превращай general-схему в model_architecture, plotneuralnet или infographic;
- не теряй акторов и ключевые компоненты из запроса.

Верни только валидный JSON без markdown и пояснений.

Формат:
{json_format_block}
"""

    best_candidate = None
    best_score = -10**9

    # для general увеличиваем число попыток, чтобы был реальный выбор
    attempts = 5 if (layout_hint == "general" or general_override or mode == "general") else 3

    for attempt in range(attempts):
        raw_answer = ask_llm(
            GENERATOR_MODEL,
            system_prompt,
            user_prompt,
            temperature=0.10 if (layout_hint == "general" or general_override or mode == "general") else 0.12,
        )

        print(f"\n=== RAW ANSWER FROM GENERATOR #{attempt + 1} ===")
        print(raw_answer)
        print("=== END RAW ANSWER ===\n")

        try:
            candidate = extract_json(raw_answer)

            if layout_hint == "general" or general_override or mode == "general":
                candidate = normalize_general_diagram(candidate)
                candidate["layout_hint"] = "general"
                candidate["renderer"] = "general"

                style = candidate.get("style")
                if not isinstance(style, dict):
                    style = {}
                style.setdefault("direction", "TB")
                candidate["style"] = style

            candidate = clean_diagram_labels(candidate)

            if layout_hint == "general" or general_override or mode == "general":
                base_score = score_general_candidate(candidate)
                fit_score = candidate_task_fit_score(candidate, user_task)
                total_score = base_score + fit_score

                print(f"[GEN SCORE #{attempt + 1}] base={base_score}, fit={fit_score}, total={total_score}")

                if total_score > best_score:
                    best_score = total_score
                    best_candidate = candidate
            else:
                if best_candidate is None:
                    best_candidate = candidate

        except Exception:
            print(f"Ошибка парсинга JSON в generate_diagram(), попытка {attempt + 1}")

    if best_candidate is not None:
        if layout_hint == "general" or general_override or mode == "general":
            best_candidate = normalize_general_diagram(best_candidate)
            best_candidate["layout_hint"] = "general"
            best_candidate["renderer"] = "general"
        return best_candidate

    fallback_layout = "general" if general_override else layout_hint
    fallback_renderer = "general" if general_override else config.get("renderer", "general")

    return {
        "type": "flowchart",
        "title": "Ошибка генерации",
        "layout_hint": fallback_layout,
        "renderer": fallback_renderer,
        "style": {"direction": "TB", "theme": "clean"},
        "lanes": [],
        "nodes": [{"id": "error", "label": "Ошибка генерации JSON", "kind": "block"}],
        "edges": [],
    }



def main():
    default_prompt = """2д схема автосервиса"""
    user_task = load_user_prompt(default_prompt)

    # Референсы теперь пустые по умолчанию. Скрипт будет подтягивать 
    # только те файлы, что лежат в папке references/ (если они там есть).
    references = []

    print("=== ШАГ 1: Генерация черновика ===")
    draft = generate_diagram(user_task, references)
    print(json.dumps(draft, ensure_ascii=False, indent=2))

    print("\n=== ШАГ 2: Критика ===")
    critique = critique_diagram(user_task, draft, references)
    print(json.dumps(critique, ensure_ascii=False, indent=2))

    print("\n=== ШАГ 3: Исправление ===")
    final = improve_diagram(user_task, draft, critique)
    print(json.dumps(final, ensure_ascii=False, indent=2))

    final_clean = clean_diagram_labels(final)
    
    output_filename = f"diagram_{int(time.time())}"
    render_diagram(final_clean, output_filename)

    pass


if __name__ == "__main__":
    with ServerManager() as manager:
        manager.start_default_servers()
        main()
        manager.stop_all()
