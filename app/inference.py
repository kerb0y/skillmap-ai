"""
SkillMap AI - Inference Module
==============================
Fungsi-fungsi utama untuk:
- Ekstraksi skill dari teks CV
- Prediksi karier berbasis model AI
- Deteksi skill gap
- Perhitungan career match score
- Generate learning path
"""

import json
import pickle
import re
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import tensorflow as tf


# ============================================================
# Load artifacts
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

model = tf.keras.models.load_model(ARTIFACT_DIR / "skillmap_model.keras")

with open(ARTIFACT_DIR / "mlb.pkl", "rb") as f:
    mlb = pickle.load(f)

with open(ARTIFACT_DIR / "label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open(ARTIFACT_DIR / "job_skill_map.json", "r", encoding="utf-8") as f:
    job_skill_map = json.load(f)

with open(ARTIFACT_DIR / "course_links.json", "r", encoding="utf-8") as f:
    course_links = json.load(f)

with open(ARTIFACT_DIR / "known_skills.json", "r", encoding="utf-8") as f:
    known_skills = json.load(f)


# ============================================================
# Skill alias & validation
# ============================================================
SKILL_ALIASES = {
    "problem solving": "problem solving skills",
    "analytic thinking": "analytical skills",
    "data visualization skills( power bi/ tableau )": "data visualization",
    "programming language skills": "programming",
}

INVALID_VALUES = {
    "tidak", "ya", "nan", "na", "not applicable",
    "student (unemployed)", "belum bekerja", "",
}


def normalize_skill(skill):
    """Lowercase, strip, dan terapkan alias."""
    skill = str(skill).lower().strip()
    skill = skill.strip(";,. ")
    return SKILL_ALIASES.get(skill, skill)


def normalize_skill_list(skills):
    """Normalisasi dan deduplikasi list skill."""
    return sorted(list(set([normalize_skill(s) for s in skills])))


def is_valid_skill(skill):
    """Cek apakah skill valid (bukan noise/invalid value)."""
    skill = str(skill).strip().lower()
    if skill in INVALID_VALUES:
        return False
    if skill.replace(".", "", 1).isdigit():
        return False
    if len(skill) <= 1:
        return False
    return True


# ============================================================
# Core Functions
# ============================================================
def extract_skills_from_cv_text(cv_text):
    """
    Ekstrak skill dari teks CV berdasarkan daftar known_skills.
    Mengembalikan list skill yang terdeteksi (normalized).
    """
    cv_text = cv_text.lower()
    detected_skills = []

    for skill in known_skills:
        skill_lower = skill.lower().strip()
        if not is_valid_skill(skill_lower):
            continue

        pattern = r"\b" + re.escape(skill_lower) + r"\b"
        if re.search(pattern, cv_text):
            detected_skills.append(skill_lower)

    return normalize_skill_list(detected_skills)


def predict_career_from_skills(user_skills):
    """
    Prediksi karier menggunakan model AI berdasarkan skill user.
    Returns: (predicted_job, confidence_score)
    """
    user_skills = normalize_skill_list(user_skills)
    user_vector = mlb.transform([user_skills])

    prediction = model.predict(user_vector, verbose=0)
    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100  # Convert to 0-100

    predicted_job = label_encoder.inverse_transform([predicted_index])[0]

    # Filter invalid predictions
    invalid_jobs = {"not applicable", "tidak", "student (unemployed)", "nan", "belum bekerja"}
    if predicted_job.lower() in invalid_jobs:
        return None, 0.0

    return predicted_job, round(confidence, 2)


def detect_skill_gap(user_skills, target_job):
    """
    Deteksi skill gap antara skill user dan required skills target job.
    Returns: (owned_skills, skill_gap, required_skills)
    """
    user_skills = normalize_skill_list(user_skills)
    target_job = target_job.lower().strip()

    required_skills = job_skill_map.get(target_job, [])
    required_skills = normalize_skill_list(required_skills)

    owned_skills = [s for s in required_skills if s in user_skills]
    skill_gap = [s for s in required_skills if s not in user_skills]

    return owned_skills, skill_gap, required_skills


def calculate_readiness_score(
    owned_skills, required_skills, model_career_score, quiz_score
):
    """
    Hitung career match score menggunakan formula:
    career_match_score = (0.60 * skill_match_score) + (0.25 * model_career_score) + (0.15 * quiz_score)

    Returns: dict dengan semua komponen skor
    """
    # Skill match score (0-100)
    if len(required_skills) > 0:
        skill_match_score = (len(owned_skills) / len(required_skills)) * 100
    else:
        skill_match_score = 0.0

    # Clamp values to 0-100
    skill_match_score = min(max(skill_match_score, 0), 100)
    model_career_score = min(max(model_career_score, 0), 100)
    quiz_score = min(max(quiz_score, 0), 100)

    # Weighted formula
    career_match_score = (
        (0.60 * skill_match_score)
        + (0.25 * model_career_score)
        + (0.15 * quiz_score)
    )

    gap_score = 100 - career_match_score

    return {
        "career_match_score": round(career_match_score, 2),
        "gap_score": round(gap_score, 2),
        "skill_match_score": round(skill_match_score, 2),
        "model_career_score": round(model_career_score, 2),
        "quiz_score": round(quiz_score, 2),
    }


def generate_learning_path(skill_gap):
    """
    Generate rekomendasi learning path untuk skill gap.
    Returns: list of {skill, course_link}
    """
    learning_path = []
    for skill in skill_gap:
        link = course_links.get(
            skill,
            f"https://www.coursera.org/courses?query={quote_plus(skill)}",
        )
        learning_path.append({"skill": skill, "course_link": link})
    return learning_path


def generate_summary(recommended_career, career_match_score, skill_gap):
    """
    Generate interpretasi singkat untuk frontend.
    """
    # Determine level
    if career_match_score >= 80:
        level = "sangat tinggi"
    elif career_match_score >= 60:
        level = "cukup baik"
    elif career_match_score >= 40:
        level = "sedang"
    else:
        level = "masih rendah"

    summary = (
        f"Kamu memiliki kecocokan {career_match_score:.0f}% ({level}) "
        f"untuk posisi {recommended_career.title()}."
    )

    if skill_gap:
        gap_names = ", ".join(skill_gap[:3])
        remaining = len(skill_gap) - 3
        if remaining > 0:
            summary += (
                f" Skill yang perlu ditingkatkan antara lain: {gap_names}, "
                f"dan {remaining} skill lainnya."
            )
        else:
            summary += f" Skill yang perlu ditingkatkan: {gap_names}."
    else:
        summary += " Semua skill yang dibutuhkan sudah kamu miliki. Hebat!"

    return summary


# ============================================================
# Main API function
# ============================================================
def get_skillmap_result(cv_text, target_job="", quiz_score=80):
    """
    Fungsi utama yang dipanggil oleh API.
    Menggabungkan semua langkah: extract → predict → gap → score → learning path.

    Returns: dict sesuai format response API
    """
    # 1. Extract skills dari CV
    detected_skills = extract_skills_from_cv_text(cv_text)

    # 2. Prediksi karier dari model AI
    try:
        predicted_career, model_confidence = predict_career_from_skills(detected_skills)
    except Exception:
        predicted_career = None
        model_confidence = 0.0

    # 3. Tentukan recommended_career
    target_job_clean = target_job.lower().strip() if target_job else ""

    if target_job_clean and target_job_clean in job_skill_map:
        recommended_career = target_job_clean
    elif predicted_career:
        recommended_career = predicted_career
    else:
        recommended_career = target_job_clean if target_job_clean else "unknown"

    # 4. Detect skill gap berdasarkan recommended_career
    owned_skills, skill_gap, required_skills = detect_skill_gap(
        detected_skills, recommended_career
    )

    # 5. Hitung skor kesiapan
    scores = calculate_readiness_score(
        owned_skills=owned_skills,
        required_skills=required_skills,
        model_career_score=model_confidence,
        quiz_score=quiz_score,
    )

    # 6. Generate learning path
    learning_path = generate_learning_path(skill_gap)

    # 7. Generate summary
    summary = generate_summary(
        recommended_career=recommended_career,
        career_match_score=scores["career_match_score"],
        skill_gap=skill_gap,
    )

    return {
        "detected_skills_from_cv": detected_skills,
        "target_job": target_job_clean if target_job_clean else None,
        "recommended_career": recommended_career,
        "predicted_career_ai": predicted_career,
        "career_match_score": scores["career_match_score"],
        "gap_score": scores["gap_score"],
        "skill_match_score": scores["skill_match_score"],
        "model_career_score": scores["model_career_score"],
        "quiz_score": scores["quiz_score"],
        "skill_dimiliki": owned_skills,
        "skill_gap": skill_gap,
        "learning_path": learning_path,
        "summary": summary,
    }