from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path


def detect_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "main.py").exists():
            return candidate
    raise FileNotFoundError(
        f"Не удалось найти корень репозитория от {start}. "
        f"Ожидался файл main.py в одном из родительских каталогов."
    )


THIS_FILE = Path(__file__).resolve()
TESTS_DIR = THIS_FILE.parent
ROOT = detect_repo_root(TESTS_DIR)
CASES_DIR = TESTS_DIR / "cases"
RUNS_DIR = TESTS_DIR / "runs"
PROMPT_FILE = ROOT / "_test_prompt.txt"
TIMEOUT_SEC = 900

KNOWN_JSON_OUTPUTS = [
    ROOT / "draft.json",
    ROOT / "critique.json",
    ROOT / "final.json",
]

IMAGE_PATTERNS = [
    "diagram_*.png",
    "diagram_*.svg",
    "diagram_*.pdf",
    "diagram_*.gv.png",
    "diagram_*.gv.svg",
    "diagram_*.gv.pdf",
    "final_diagram*.png",
    "final_diagram*.svg",
    "final_diagram*.pdf",
]

SEARCH_DIRS = [ROOT]


def load_case(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("type") != "general":
        raise ValueError(f"Этот runner принимает только general, а не {data.get('type')}")
    return data


def is_inside(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
        return True
    except Exception:
        return False


def should_skip_search_path(path: Path, current_run_dir: Path) -> bool:
    resolved = path.resolve()

    if is_inside(resolved, RUNS_DIR):
        return True

    if is_inside(resolved, current_run_dir):
        return True

    return False


def cleanup_outputs() -> None:
    for path in KNOWN_JSON_OUTPUTS:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass

    for base in SEARCH_DIRS:
        if not base.exists():
            continue
        for pattern in IMAGE_PATTERNS:
            for path in base.rglob(pattern):
                try:
                    if is_inside(path, RUNS_DIR):
                        continue
                    path.unlink()
                except Exception:
                    pass

    try:
        if PROMPT_FILE.exists():
            PROMPT_FILE.unlink()
    except Exception:
        pass


def write_prompt(case: dict) -> None:
    PROMPT_FILE.write_text(case["prompt"], encoding="utf-8")


def safe_read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def extract_last_json_block(text: str) -> dict | None:
    positions = [m.start() for m in re.finditer(r"\{", text)]
    for start in reversed(positions):
        candidate = text[start:].strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return None


def extract_json_after_marker(text: str, marker: str) -> dict | None:
    idx = text.rfind(marker)
    if idx == -1:
        return None

    tail = text[idx + len(marker):].strip()
    positions = [m.start() for m in re.finditer(r"\{", tail)]

    for start in positions:
        candidate = tail[start:].strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return None


def extract_final_json_from_stdout(stdout: str) -> dict | None:
    markers = [
        "=== END RAW ANSWER FROM IMPROVER ===",
        "=== RAW ANSWER FROM IMPROVER ===",
    ]
    for marker in markers:
        data = extract_json_after_marker(stdout, marker)
        if data is not None:
            return data

    return extract_last_json_block(stdout)


def copy_if_new_enough(
    path: Path,
    dst_dir: Path,
    started_ts: float,
    copied: list[str],
) -> None:
    if not path.exists() or not path.is_file():
        return
    if path.stat().st_mtime < started_ts:
        return

    dst = dst_dir / path.name

    try:
        if path.resolve() == dst.resolve():
            return
    except Exception:
        pass

    if dst.exists():
        stem = dst.stem
        suffix = dst.suffix
        i = 1
        while True:
            alt = dst_dir / f"{stem}_{i}{suffix}"
            if not alt.exists():
                dst = alt
                break
            i += 1

    shutil.copy2(path, dst)
    copied.append(dst.name)


def collect_render_outputs(
    artifacts_dir: Path,
    started_ts: float,
    stdout_text: str,
    current_run_dir: Path,
) -> tuple[list[str], Path | None]:
    copied: list[str] = []

    for path in KNOWN_JSON_OUTPUTS:
        copy_if_new_enough(path, artifacts_dir, started_ts, copied)

    for base in SEARCH_DIRS:
        if not base.exists():
            continue

        for pattern in IMAGE_PATTERNS:
            for path in base.rglob(pattern):
                if should_skip_search_path(path, current_run_dir):
                    continue
                copy_if_new_enough(path, artifacts_dir, started_ts, copied)

    extracted_json_path = None

    if "final.json" not in copied:
        final_data = extract_final_json_from_stdout(stdout_text)
        if final_data is not None:
            extracted_json_path = artifacts_dir / "final_from_stdout.json"
            extracted_json_path.write_text(
                json.dumps(final_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            copied.append(extracted_json_path.name)

    return sorted(set(copied)), extracted_json_path


def build_metrics_from_json(data: dict | None) -> dict | None:
    if not isinstance(data, dict):
        return None

    return {
        "renderer": data.get("renderer"),
        "layout_hint": data.get("layout_hint"),
        "nodes_count": len(data.get("nodes", [])),
        "edges_count": len(data.get("edges", [])),
    }


def run_case(case: dict) -> dict:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"{stamp}_{case['id']}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cleanup_outputs()
    write_prompt(case)

    cmd = ["python3", "main.py"]
    started = time.time()

    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SEC,
        )
        status = "ok" if proc.returncode == 0 else "failed"
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        returncode = proc.returncode
    except subprocess.TimeoutExpired as e:
        status = "timeout"
        stdout = e.stdout.decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        returncode = None

    (run_dir / "stdout.log").write_text(stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(stderr, encoding="utf-8")
    (run_dir / "case.json").write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8")

    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    copied, extracted_json_path = collect_render_outputs(
        artifacts_dir=artifacts_dir,
        started_ts=started,
        stdout_text=stdout,
        current_run_dir=run_dir,
    )

    report = {
        "case_id": case["id"],
        "type": case["type"],
        "status": status,
        "returncode": returncode,
        "duration_sec": round(time.time() - started, 2),
        "command": "python3 main.py",
        "cwd": str(ROOT),
        "copied_artifacts": copied,
        "expected": case.get("expected"),
        "notes": case.get("notes"),
    }

    final_data = safe_read_json(artifacts_dir / "final.json")
    if final_data is None and extracted_json_path is not None:
        final_data = safe_read_json(extracted_json_path)

    metrics = build_metrics_from_json(final_data)
    if metrics is not None:
        report["final_json_metrics"] = metrics

    if isinstance(final_data, dict):
        report["routing_flags"] = {
            "is_general": final_data.get("renderer") == "general" and final_data.get("layout_hint") == "general",
            "renderer": final_data.get("renderer"),
            "layout_hint": final_data.get("layout_hint"),
        }

    (run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    shutil.make_archive(str(run_dir), "zip", root_dir=run_dir)
    return report


def main() -> None:
    cases = sorted(CASES_DIR.glob("G-*.json"))
    if not cases:
        print("Нет general-кейсов")
        raise SystemExit(1)

    reports = []
    for case_file in cases:
        case = load_case(case_file)
        print(f"=== {case['id']} ===")
        report = run_case(case)
        reports.append(report)
        print(report["status"], report.get("duration_sec"))
        print("artifacts:", ", ".join(report.get("copied_artifacts", [])) or "none")
        if "final_json_metrics" in report:
            print("metrics:", report["final_json_metrics"])
        if "routing_flags" in report:
            print("routing:", report["routing_flags"])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_json = RUNS_DIR / f"summary_{stamp}.json"
    summary_csv = RUNS_DIR / f"summary_{stamp}.csv"

    summary_json.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_id",
            "status",
            "returncode",
            "duration_sec",
            "renderer",
            "layout_hint",
            "nodes_count",
            "edges_count",
            "is_general",
            "copied_artifacts",
        ])
        for item in reports:
            metrics = item.get("final_json_metrics", {})
            routing = item.get("routing_flags", {})
            writer.writerow([
                item.get("case_id"),
                item.get("status"),
                item.get("returncode"),
                item.get("duration_sec"),
                metrics.get("renderer"),
                metrics.get("layout_hint"),
                metrics.get("nodes_count"),
                metrics.get("edges_count"),
                routing.get("is_general"),
                "; ".join(item.get("copied_artifacts", [])),
            ])

    print("Готово. Смотри tests/runs/")


if __name__ == "__main__":
    main()