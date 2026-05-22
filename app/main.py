"""
SkillMap AI - FastAPI Application
==================================
API untuk analisis CV, deteksi skill gap, dan rekomendasi learning path.

Jalankan:
    uvicorn app.main:app --reload
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.inference import (
    get_skillmap_result,
    job_skill_map,
    known_skills,
)


app = FastAPI(
    title="SkillMap AI API",
    description=(
        "API untuk analisis CV, prediksi karier, deteksi skill gap, "
        "dan rekomendasi learning path berbasis AI."
    ),
    version="1.0.0",
)


# ============================================================
# Request / Response Models
# ============================================================
class SkillMapRequest(BaseModel):
    cv_text: str = Field(
        ...,
        description="Teks CV user (hasil ekstraksi dari PDF/text)",
        min_length=1,
    )
    target_job: str = Field(
        default="",
        description="Target pekerjaan yang diinginkan user (opsional)",
    )
    quiz_score: float = Field(
        default=80,
        ge=0,
        le=100,
        description="Skor mini-quiz (0-100)",
    )


# ============================================================
# Endpoints
# ============================================================
@app.get("/")
def root():
    """Root endpoint - info API."""
    return {
        "message": "SkillMap AI API is running",
        "version": "1.0.0",
        "endpoints": [
            "GET /",
            "GET /health",
            "GET /jobs",
            "GET /skills",
            "POST /predict",
        ],
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "skillmap-ai"}


@app.get("/jobs")
def list_jobs():
    """Menampilkan daftar target_job yang tersedia."""
    jobs = sorted(list(job_skill_map.keys()))
    return {"total": len(jobs), "jobs": jobs}


@app.get("/skills")
def list_skills():
    """Menampilkan daftar known_skills."""
    return {"total": len(known_skills), "skills": known_skills}


@app.post("/predict")
def predict(request: SkillMapRequest):
    """
    Prediksi karier, hitung career match score, deteksi skill gap,
    dan generate learning path berdasarkan CV text.
    """
    result = get_skillmap_result(
        cv_text=request.cv_text,
        target_job=request.target_job,
        quiz_score=request.quiz_score,
    )
    return result