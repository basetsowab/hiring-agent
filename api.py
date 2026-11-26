"""HTTP API for the resume scoring pipeline.

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health           liveness check
    GET  /roles            list roles that resumes can be scored against
    POST /score            upload a resume PDF and get a structured evaluation
"""

import os
import tempfile
from typing import Any, Dict, List, Optional

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from github import fetch_and_display_github_info
from models import JSONResume, build_evaluation_model
from pdf import PDFHandler
from prompt import DEFAULT_MODEL
from roles import Role, list_available_roles, load_role
from score import _evaluate_resume, find_profile, is_valid_resume_data

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))

app = FastAPI(
    title="Hiring Agent API",
    description="Score resume PDFs against a role rubric, optionally enriched with GitHub signals.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _summarize(evaluation, role: Role) -> Dict[str, Any]:
    """Turn the evaluation model into a JSON-friendly result with totals."""
    categories: List[Dict[str, Any]] = []
    total = 0.0
    max_score = 0

    for category in role.categories:
        cat = getattr(evaluation.scores, category.key, None)
        if cat is None:
            continue
        capped = min(cat.score, category.max)
        total += capped
        max_score += category.max
        categories.append(
            {
                "key": category.key,
                "label": category.label,
                "score": capped,
                "max": category.max,
                "evidence": cat.evidence,
            }
        )

    bonus = evaluation.bonus_points.model_dump() if evaluation.bonus_points else None
    deductions = evaluation.deductions.model_dump() if evaluation.deductions else None
    if bonus:
        total += bonus.get("total", 0) or 0
    if deductions:
        total -= deductions.get("total", 0) or 0
    total = min(total, max_score + role.bonus_max)

    return {
        "total_score": round(total, 1),
        "max_score": max_score,
        "categories": categories,
        "bonus_points": bonus,
        "deductions": deductions,
        "key_strengths": list(evaluation.key_strengths or []),
        "areas_for_improvement": list(evaluation.areas_for_improvement or []),
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "model": DEFAULT_MODEL}


@app.get("/roles")
def roles() -> Dict[str, Any]:
    result = []
    for name in list_available_roles():
        role = load_role(name)
        result.append(
            {
                "name": role.name,
                "position_title": role.position_title,
                "bonus_max": role.bonus_max,
                "categories": [
                    {"key": c.key, "label": c.label, "max": c.max}
                    for c in role.categories
                ],
            }
        )
    return {"roles": result}


@app.post("/score")
def score_resume(
    file: UploadFile = File(..., description="Resume PDF"),
    role: Optional[str] = Form(None, description="Role name (see GET /roles)"),
    include_github: bool = Form(True, description="Enrich with GitHub profile data"),
    github_url: Optional[str] = Form(None, description="Override the GitHub profile URL"),
    include_resume: bool = Form(False, description="Include the extracted resume JSON"),
) -> Dict[str, Any]:
    # Declared as a sync endpoint so FastAPI runs the blocking LLM calls in a threadpool.
    available = list_available_roles()
    role_name = role or (available[0] if available else None)
    if not role_name:
        raise HTTPException(500, "No roles are configured under roles/")
    try:
        role_def = load_role(role_name)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if file.content_type not in ("application/pdf", "application/octet-stream") and not (
        file.filename or ""
    ).lower().endswith(".pdf"):
        raise HTTPException(415, "Upload must be a PDF")

    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File exceeds {MAX_UPLOAD_BYTES} bytes")
    if not data.startswith(b"%PDF"):
        raise HTTPException(415, "Upload is not a valid PDF")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        pdf_path = tmp.name

    try:
        resume_data: Optional[JSONResume] = PDFHandler().extract_json_from_pdf(pdf_path)
    finally:
        os.unlink(pdf_path)

    if not is_valid_resume_data(resume_data):
        raise HTTPException(422, "Could not extract resume content from the PDF")

    github_data: Dict[str, Any] = {}
    if include_github:
        url = github_url
        if not url:
            profiles = (resume_data.basics.profiles or []) if resume_data.basics else []
            profile = find_profile(profiles, "Github")
            url = profile.url if profile else None
        if url:
            github_data = (
                fetch_and_display_github_info(url, position_title=role_def.position_title)
                or {}
            )

    evaluation = _evaluate_resume(
        resume_data, role_def, build_evaluation_model(role_def), github_data
    )
    if evaluation is None:
        raise HTTPException(502, "The LLM did not return an evaluation")

    candidate = (
        resume_data.basics.name
        if resume_data.basics and resume_data.basics.name
        else os.path.splitext(file.filename or "candidate")[0]
    )

    result = {
        "candidate_name": candidate,
        "role": role_def.name,
        "model": DEFAULT_MODEL,
        "github_used": bool(github_data),
        **_summarize(evaluation, role_def),
    }
    if include_resume:
        result["resume"] = resume_data.model_dump()
    return result
