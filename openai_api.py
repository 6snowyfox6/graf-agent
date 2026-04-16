from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
import io
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent
RUNS_ROOT = REPO_ROOT / "api_runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

from config import (  # noqa: E402
    MAX_PROMPT_CHARS,
    PIPELINE_TIMEOUT_SEC,
    MAX_CONCURRENT_RUNS,
)

_run_semaphore = threading.Semaphore(MAX_CONCURRENT_RUNS)

APP_MODELS = [
    "graf-agent-auto",
    "graf-agent-general",
    "graf-agent-pipeline",
    "graf-agent-model-architecture",
    "graf-agent-infographic",
]

MODEL_PREFIXES = {
    "graf-agent-auto": "",
    "graf-agent-general": "general architecture\n\nЭто general architecture diagram.\nЭто не pipeline и не neural network.\n\n",
    "graf-agent-pipeline": "pipeline diagram\n\nЭто pipeline diagram.\nЭто не general architecture и не neural network.\n\n",
    "graf-agent-model-architecture": "model architecture\n\nЭто model architecture diagram.\nИспользуй режим архитектуры модели.\n\n",
    "graf-agent-infographic": "infographic\n\nЭто infographic diagram.\nИспользуй инфографический режим.\n\n",
}

from contextlib import asynccontextmanager
from server_manager import ServerManager

server_manager_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global server_manager_instance
    try:
        server_manager_instance = ServerManager()
        server_manager_instance.start_default_servers()
        yield
    finally:
        if server_manager_instance:
            server_manager_instance.stop_all()

app = FastAPI(
    title="graf-agent OpenAI-compatible API",
    version="0.1.0",
    description="OpenAI-compatible wrapper for graf-agent",
    lifespan=lifespan
)

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/artifacts", StaticFiles(directory=str(RUNS_ROOT)), name="artifacts")

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = Field(default="graf-agent-auto")
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = False


def build_prompt(messages: list[ChatMessage], model_name: str) -> str:
    system_parts: list[str] = []
    dialog_parts: list[str] = []

    for msg in messages:
        role = (msg.role or "").strip().lower()
        content = (msg.content or "").strip()
        if not content:
            continue

        if role == "system":
            system_parts.append(content)
        elif role == "user":
            dialog_parts.append(content)
        else:
            dialog_parts.append(f"[{role}] {content}")

    if not dialog_parts and system_parts:
        dialog_parts = system_parts
        system_parts = []

    prompt_body = "\n\n".join(dialog_parts).strip()
    system_block = "\n\n".join(system_parts).strip()

    prefix = MODEL_PREFIXES.get(model_name or "graf-agent-auto", "")

    if system_block:
        return f"{prefix}{system_block}\n\n{prompt_body}".strip()
    return f"{prefix}{prompt_body}".strip()


def rough_token_count(text: str) -> int:
    text = text or ""
    return max(1, len(text) // 4)


def find_latest_run_dir(outputs_dir: Path) -> Path | None:
    if not outputs_dir.exists():
        return None
    runs = [p for p in outputs_dir.glob("diagram_*") if p.is_dir()]
    if not runs:
        return None
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def collect_artifact_urls(workdir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for p in workdir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".png", ".svg", ".pdf", ".json", ".md"}:
            continue
        rel = p.relative_to(RUNS_ROOT).as_posix()
        result[p.name] = f"/artifacts/{rel}"
    return result


def make_openai_response(
    content: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": now,
                "owned_by": "graf-agent",
            }
            for model_id in APP_MODELS
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    model_name = req.model or "graf-agent-auto"

    if model_name not in APP_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_name}'. Allowed: {', '.join(APP_MODELS)}",
        )

    if req.stream:
        raise HTTPException(
            status_code=400,
            detail="stream=true is not supported yet. Use stream=false.",
        )

    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    prompt = build_prompt(req.messages, model_name)
    if len(prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt too long ({len(prompt)} chars). Max allowed: {MAX_PROMPT_CHARS}.",
        )

    acquired = _run_semaphore.acquire(blocking=False)
    if not acquired:
        raise HTTPException(
            status_code=429,
            detail=f"Too many concurrent requests. Max {MAX_CONCURRENT_RUNS} pipelines at a time.",
        )

    request_id = uuid.uuid4().hex[:12]
    workdir = RUNS_ROOT / request_id
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        (workdir / "_test_prompt.txt").write_text(prompt, encoding="utf-8")

        env = os.environ.copy()
        old_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + old_pythonpath if old_pythonpath else "")

        cmd = ["python3", str(REPO_ROOT / "main.py")]

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(workdir),
                env=env,
                capture_output=True,
                text=True,
                timeout=PIPELINE_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired as e:
            raise HTTPException(
                status_code=504,
                detail={
                    "error": f"graf-agent timed out after {PIPELINE_TIMEOUT_SEC}s",
                    "request_id": request_id,
                    "stdout_tail": (e.stdout or "")[-4000:],
                    "stderr_tail": (e.stderr or "")[-4000:],
                },
            )
    finally:
        _run_semaphore.release()

    stdout_tail = (proc.stdout or "")[-4000:]
    stderr_tail = (proc.stderr or "")[-4000:]

    if proc.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "graf-agent finished with non-zero exit code",
                "request_id": request_id,
                "returncode": proc.returncode,
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
            },
        )

    latest_run = find_latest_run_dir(workdir / "outputs")
    if latest_run is None:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "No outputs/diagram_* directory found after run",
                "request_id": request_id,
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
            },
        )

    final_json = read_json_if_exists(latest_run / "final.json")
    critique_json = read_json_if_exists(latest_run / "critique.json")
    draft_json = read_json_if_exists(latest_run / "draft.json")
    shap_json = read_json_if_exists(latest_run / "critic_influence_report.json")
    artifact_urls = collect_artifact_urls(workdir)

    payload = {
        "request_id": request_id,
        "run_id": latest_run.name,
        "final": final_json,
        "draft": draft_json,
        "critique": critique_json,
        "critic_influence": shap_json,
        "artifacts": artifact_urls,
    }

    content = json.dumps(payload, ensure_ascii=False, indent=2)

    prompt_tokens = rough_token_count(prompt)
    completion_tokens = rough_token_count(content)

    return JSONResponse(
        content=make_openai_response(
            content=content,
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    )

# --- Interactive Canvas API Routes ---

from pipeline.generator import generate_diagram
from pipeline.normalizer import normalize_general_diagram
from pipeline.critic import critique_diagram, build_patch_plan
from pipeline.improver import improve_diagram

class GenerateRequest(BaseModel):
    prompt: str

@app.post("/api/generate")
def api_generate(req: GenerateRequest):
    try:
        draft = generate_diagram(
            user_task=req.prompt,
            references=[],
            normalize_general_diagram_fn=normalize_general_diagram
        )
        return {"status": "success", "graph": draft}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ImproveRequest(BaseModel):
    prompt: str
    graph: dict
    user_feedback: str | None = None

@app.post("/api/improve")
def api_improve(req: ImproveRequest):
    try:
        from critic_influence import CriticAnalyzer
        
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        run_dir = RUNS_ROOT / "export_api" / run_id
        
        critique = critique_diagram(req.prompt, req.graph, references=[], user_feedback=req.user_feedback)
        patch_plan = build_patch_plan(critique)
        final, meta = improve_diagram(
            req.prompt,
            req.graph,
            critique,
            references=[],
            patch_plan=patch_plan,
            normalize_general_diagram_fn=normalize_general_diagram
        )
        
        shap_urls = {}
        try:
            analyzer = CriticAnalyzer(output_root=str(RUNS_ROOT / "shap_history"))
            influence = analyzer.analyze_and_save(
                run_id=run_id,
                run_dir=run_dir,
                draft=req.graph,
                critique=critique,
                final=final,
                verify=None
            )
            
            if influence.local_contrib_plot and Path(influence.local_contrib_plot).exists():
               shap_urls['bar'] = f"/artifacts/export_api/{run_id}/{Path(influence.local_contrib_plot).name}"
            if influence.traceability_plot and Path(influence.traceability_plot).exists():
               shap_urls['network'] = f"/artifacts/export_api/{run_id}/{Path(influence.traceability_plot).name}"
        except Exception as e:
            print(f"SHAP Error: {e}")

        return {"status": "success", "graph": final, "critique": critique, "meta": meta, "shap": shap_urls}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/export")
def api_export(req: dict):
    try:
        from renderers.graphviz_renderers import render_general_diagram
        export_id = uuid.uuid4().hex[:8]
        out_dir = RUNS_ROOT / f"export_{export_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        render_general_diagram(req, output_name="diagram", output_dir=str(out_dir))
        
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for p in out_dir.iterdir():
                if p.is_file():
                    zf.write(p, arcname=p.name)
        memory_file.seek(0)
        return StreamingResponse(
            memory_file, 
            media_type="application/zip", 
            headers={"Content-Disposition": f"attachment; filename=diagram_export_{export_id}.zip"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class GeneratePNNRequest(BaseModel):
    prompt: str

@app.post("/api/generate_pnn")
def api_generate_pnn(req: GeneratePNNRequest):
    try:
        from plotneuralnet_renderer import PlotNeuralNetRenderer
        draft = generate_diagram(
            user_task=req.prompt + " (plotneuralnet architecture)",
            references=[],
            normalize_general_diagram_fn=None
        )
        
        critique = critique_diagram(req.prompt, draft, references=[])
        patch_plan = build_patch_plan(critique)
        final, meta = improve_diagram(
            req.prompt,
            draft,
            critique,
            references=[],
            patch_plan=patch_plan,
            normalize_general_diagram_fn=None
        )

        safe_hex = uuid.uuid4().hex[:8]
        pnn_root = RUNS_ROOT / "export_pnn"
        renderer = PlotNeuralNetRenderer(
            project_root=str(REPO_ROOT),
            output_root=str(pnn_root)
        )
        res = renderer.render(final, output_name=f"pnn_{safe_hex}")
        
        png_path = Path(res.get("preview_path", ""))
        pdf_path = Path(res.get("pdf_path", ""))
        
        return {
            "status": "success",
            "png_url": f"/artifacts/export_pnn/pnn_{safe_hex}/{png_path.name}" if png_path.exists() else None,
            "pdf_url": f"/artifacts/export_pnn/pnn_{safe_hex}/{pdf_path.name}" if pdf_path.exists() else None,
            "graph": final
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


