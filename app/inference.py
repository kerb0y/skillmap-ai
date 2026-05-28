"""
SkillMap AI - Inference Module v3.0
=====================================
Mendukung:
- career_categories (~100 kategori bersih) untuk /jobs & target_job user
- model_classes (subset yang dilatih model, sample >= 5)
- Fallback rule-based skill gap jika target_job tidak ada di model_classes
"""

import json
import pickle
import re
import warnings
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

# career_categories = semua kategori untuk /jobs & target_job lookup
with open(ARTIFACT_DIR / "career_categories.json", "r", encoding="utf-8") as f:
    CAREER_CATEGORIES: list = json.load(f)

# model_classes = subset yang dilatih model (sample >= 5)
_model_classes_path = ARTIFACT_DIR / "model_classes.json"
if _model_classes_path.exists():
    with open(_model_classes_path, "r", encoding="utf-8") as f:
        MODEL_CLASSES: list = json.load(f)
else:
    MODEL_CLASSES: list = sorted(list(label_encoder.classes_))


# ============================================================
# Skill normalization helpers
# ============================================================
SKILL_ALIASES = {
    "problem solving": "problem solving skills",
    "analytic thinking": "analytical skills",
    "data visualization skills( power bi/ tableau )": "data visualization",
    "programming language skills": "programming",
    "logical skills": "logical thinking",
    "work under pressure": "stress management",
}

# Skill 1-2 karakter yang valid (nama teknologi)
KNOWN_SHORT_SKILLS = {
    "c#", "js", "hr", "ml", "ai", "go", "c++", "c++",
}

# Alias skill pendek ke versi lebih panjang
SKILL_EXPAND = {
    "r": "r programming",
    "c": "c programming",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "js": "javascript",
}

INVALID_VALUES = {
    "tidak", "ya", "nan", "na", "not applicable",
    "student (unemployed)", "belum bekerja", "",
}


def normalize_skill(skill: str) -> str:
    skill = str(skill).lower().strip().strip(";,. ")
    # Expand alias pendek ke nama bermakna
    skill = SKILL_EXPAND.get(skill, skill)
    return SKILL_ALIASES.get(skill, skill)


def is_valid_skill(skill: str) -> bool:
    """Validasi skill: min 3 karakter, bukan angka, bukan simbol, bukan noise."""
    s = str(skill).strip().lower()
    # Whitelist 2-char yang valid (nama teknologi) dan special tech names
    if s in KNOWN_SHORT_SKILLS:
        return True
    # Special: c++ / c# are valid tech
    if s in {"c++", "c#"}:
        return True
    if s in INVALID_VALUES:
        return False
    if len(s) < 3:
        return False
    if s.replace(".", "", 1).replace(",", "", 1).isdigit():
        return False
    # Harus punya minimal 2 huruf alphabet
    alpha_count = sum(1 for c in s if c.isalpha())
    if alpha_count < 2:
        return False
    return True


def normalize_skill_list(skills: list) -> list:
    """Normalize, expand, dan filter skill list."""
    result = []
    for s in skills:
        normalized = normalize_skill(str(s))
        if is_valid_skill(normalized):
            result.append(normalized)
    return sorted(list(set(result)))


# ============================================================
# Target job normalization (user input → career_category)
# ============================================================
def normalize_target_job(target_job: str) -> str | None:
    """
    Normalize input user ke career_category yang dikenal sistem.
    Lookup order: exact → substring → word-overlap → keyword fallback.
    """
    if not target_job:
        return None
    raw = target_job.lower().strip()
    if not raw:
        return None

    # 1. Exact match ke CAREER_CATEGORIES
    if raw in CAREER_CATEGORIES:
        return raw

    # 2. Substring match
    for cat in CAREER_CATEGORIES:
        if cat in raw or raw in cat:
            return cat

    # 3. Word overlap
    raw_words = set(raw.split())
    best_cat, best_score = None, 0
    for cat in CAREER_CATEGORIES:
        cat_words = set(cat.split())
        overlap = len(raw_words & cat_words)
        if overlap > best_score:
            best_score = overlap
            best_cat = cat
    if best_score >= 2:
        return best_cat
    # Single word match but only for distinct enough words
    if best_score == 1 and best_cat:
        matched_word = list(raw_words & set(best_cat.split()))
        if matched_word and len(matched_word[0]) >= 5:
            return best_cat

    # 4. Keyword-based fallback mapping
    kw_map = [
        (["software", "programmer", "backend", "frontend", "fullstack"], "software engineer"),
        (["web", "frontend", "react", "angular", "vue", "html", "css"], "web developer"),
        (["mobile", "android", "ios", "flutter", "react native"], "mobile developer"),
        (["data analyst", "analytics", "business intelligence"], "data analyst"),
        (["data scientist"], "data scientist"),
        (["data engineer", "etl", "pipeline", "spark", "hadoop"], "data engineer"),
        (["machine learning", "deep learning", "nlp", "computer vision"], "machine learning engineer"),
        (["ai engineer", "artificial intelligence engineer"], "ai engineer"),
        (["devops", "ci/cd", "jenkins", "ansible"], "devops engineer"),
        (["cloud", "aws", "gcp", "azure", "cloud architect"], "cloud engineer"),
        (["network", "networking", "cisco", "ccna"], "network engineer"),
        (["security", "cyber", "infosec", "soc", "penetration"], "cyber security analyst"),
        (["ui", "ux", "figma", "user interface", "user experience"], "ui ux designer"),
        (["product manager", "product owner", "product management"], "product manager"),
        (["project manager", "project management", "scrum master", "agile"], "project manager"),
        (["business analyst", "requirements", "use case"], "business analyst"),
        (["qa", "quality assurance", "testing", "tester"], "quality assurance engineer"),
        (["quality control", "qc", "inspection"], "quality control"),
        (["it support", "helpdesk", "desktop support", "tech support"], "it support"),
        (["embedded", "firmware", "vlsi", "plc", "microcontroller"], "embedded engineer"),
        (["erp", "sap", "oracle erp"], "erp consultant"),
        (["solutions architect", "enterprise architect"], "solutions architect"),
        (["game", "unity", "unreal", "gaming"], "game developer"),
        (["database admin", "dba", "sql server", "oracle"], "database administrator"),
        (["technical writer", "documentation"], "technical writer"),
        (["digital marketing", "seo", "sem", "google ads", "ppc"], "digital marketing"),
        (["social media", "instagram", "tiktok", "twitter", "content strategy"], "social media specialist"),
        (["content creator", "youtuber", "content creation"], "content creator"),
        (["copywriter", "copy writing", "advertising copy"], "copywriter"),
        (["graphic", "illustrator", "photoshop", "canva", "coreldraw"], "graphic designer"),
        (["video editor", "premiere", "after effects", "vfx", "animation"], "video editor"),
        (["journalist", "reporter", "editor", "news"], "journalist"),
        (["writer", "author", "blogger"], "writer"),
        (["translator", "interpreter", "localization"], "translator"),
        (["event", "exhibition", "conference planner"], "event planner"),
        (["photographer", "photojournalist", "videographer"], "photographer"),
        (["public relation", "pr ", "communications", "media relation"], "public relations"),
        (["marketing manager", "brand manager", "marketing staff"], "marketing staff"),
        (["sales manager", "sales director", "area manager"], "sales manager"),
        (["sales", "account executive", "territory", "field sales"], "sales executive"),
        (["customer service", "customer care", "customer support"], "customer service"),
        (["call center", "telecaller", "telemarketer"], "call center agent"),
        (["store manager", "branch manager", "outlet manager"], "store manager"),
        (["retail", "store staff"], "retail staff"),
        (["cashier", "teller"], "cashier"),
        (["key account", "account manager"], "account executive"),
        (["business development", "bizdev"], "business development"),
        (["accountant", "accounting", "bookkeeping", "cpa"], "accountant"),
        (["financial analyst", "investment banking", "valuation"], "financial analyst"),
        (["tax", "pajak", "taxation", "fiscal"], "tax staff"),
        (["audit", "internal control"], "auditor"),
        (["finance", "treasury", "keuangan"], "finance staff"),
        (["banking", "bank officer", "credit analyst"], "banking staff"),
        (["insurance", "asuransi"], "insurance agent"),
        (["investment", "stock", "portfolio", "fund"], "investment analyst"),
        (["hr", "human resource", "people", "hrd"], "human resources"),
        (["recruiter", "talent acquisition", "headhunter"], "recruiter"),
        (["training", "learning development", "l&d"], "training staff"),
        (["admin", "administrator", "office admin"], "admin officer"),
        (["secretary", "personal assistant", "pa "], "secretary"),
        (["receptionist", "front desk"], "receptionist"),
        (["office staff", "office support"], "office staff"),
        (["civil engineer", "structural", "geotechnical"], "civil engineer"),
        (["mechanical engineer", "hvac", "piping"], "mechanical engineer"),
        (["electrical engineer", "power system", "control system"], "electrical engineer"),
        (["industrial engineer", "process engineer", "lean"], "industrial engineer"),
        (["architect", "interior design", "spatial"], "architect"),
        (["drafter", "cad", "autocad"], "drafter"),
        (["surveyor", "quantity surveyor"], "surveyor"),
        (["technician", "lab technician", "field technician"], "technician"),
        (["automotive", "car mechanic", "service advisor"], "automotive technician"),
        (["production", "manufacturing", "plant"], "production staff"),
        (["manufacturing staff", "factory"], "manufacturing staff"),
        (["safety", "hse", "k3"], "safety officer"),
        (["operations", "operational"], "operations staff"),
        (["logistics", "supply chain logistics", "freight", "shipping"], "logistics staff"),
        (["warehouse", "gudang", "stock"], "warehouse staff"),
        (["supply chain", "scm"], "supply chain staff"),
        (["procurement", "purchasing", "buyer", "sourcing"], "procurement staff"),
        (["teacher", "guru", "educator"], "teacher"),
        (["lecturer", "dosen", "professor"], "lecturer"),
        (["tutor", "teaching assistant", "private teacher"], "tutor"),
        (["researcher", "research", "scientist"], "researcher"),
        (["doctor", "physician", "medical doctor", "dentist"], "doctor"),
        (["nurse", "nursing", "perawat"], "nurse"),
        (["pharmacist", "pharmacy", "apoteker"], "pharmacist"),
        (["lab analyst", "laboratory", "microbiologist"], "laboratory analyst"),
        (["healthcare", "clinical", "medical staff"], "healthcare staff"),
        (["legal", "lawyer", "attorney", "advocate", "law"], "legal staff"),
        (["paralegal", "legal assistant"], "paralegal"),
        (["company secretary", "corporate secretary"], "company secretary"),
        (["hotel", "hospitality", "front office hotel"], "hospitality staff"),
        (["chef", "cook", "culinary", "pastry"], "chef / cook"),
        (["barista", "coffee", "cafe"], "barista"),
        (["tourism", "travel", "tour guide"], "tourism staff"),
        (["entrepreneur", "founder", "startup", "self employed"], "entrepreneur"),
        (["general manager", "gm ", "country manager"], "general manager"),
        (["operations manager", "operation head"], "operations manager"),
        (["management trainee", "graduate trainee"], "management trainee"),
        (["business development", "bd manager"], "business development"),
    ]

    for keywords, category in kw_map:
        if any(kw in raw for kw in keywords):
            if category in CAREER_CATEGORIES:
                return category

    return None


# ============================================================
# Core inference functions
# ============================================================
def extract_skills_from_cv_text(cv_text: str) -> list:
    """Ekstrak skill dari teks CV berdasarkan known_skills list."""
    cv_text = cv_text.lower()
    detected = []
    for skill in known_skills:
        skill_lower = skill.lower().strip()
        if not is_valid_skill(skill_lower):
            continue
        pattern = r"\b" + re.escape(skill_lower) + r"\b"
        if re.search(pattern, cv_text):
            detected.append(skill_lower)
    return normalize_skill_list(detected)


def predict_career_from_skills(user_skills: list) -> tuple:
    """
    Prediksi career_category menggunakan model AI.
    Model hanya bisa prediksi MODEL_CLASSES (subset dari CAREER_CATEGORIES).
    Returns: (career_category, confidence_score_0_to_100)
    """
    user_skills = normalize_skill_list(user_skills)
    if not user_skills:
        return None, 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        user_vector = mlb.transform([user_skills])
    prediction = model.predict(user_vector, verbose=0)
    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100

    career_category = label_encoder.inverse_transform([predicted_index])[0]

    if career_category not in CAREER_CATEGORIES:
        return None, 0.0

    return career_category, round(confidence, 2)


def detect_skill_gap(user_skills: list, target_career: str) -> tuple:
    """
    Deteksi skill gap antara skill user dan required skills career category.
    Lookup dari job_skill_map (mencakup semua career_categories termasuk rule-based).
    Returns: (owned_skills, skill_gap, required_skills)
    """
    user_skills = normalize_skill_list(user_skills)
    target_career = target_career.lower().strip()

    required_skills = job_skill_map.get(target_career, [])
    required_skills = normalize_skill_list(required_skills)

    owned_skills = [s for s in required_skills if s in user_skills]
    skill_gap = [s for s in required_skills if s not in user_skills]

    return owned_skills, skill_gap, required_skills


def calculate_readiness_score(
    owned_skills: list,
    required_skills: list,
    model_career_score: float,
    quiz_score: float,
) -> dict:
    """
    Formula: career_match = 0.60*skill_match + 0.25*model_score + 0.15*quiz
    """
    if len(required_skills) > 0:
        skill_match_score = (len(owned_skills) / len(required_skills)) * 100
    else:
        skill_match_score = 0.0

    skill_match_score = min(max(skill_match_score, 0), 100)
    model_career_score = min(max(model_career_score, 0), 100)
    quiz_score = min(max(quiz_score, 0), 100)

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


def generate_learning_path(skill_gap: list) -> list:
    learning_path = []
    for skill in skill_gap:
        link = course_links.get(
            skill,
            f"https://www.coursera.org/courses?query={quote_plus(skill)}",
        )
        learning_path.append({"skill": skill, "course_link": link})
    return learning_path


def generate_summary(
    recommended_career: str,
    career_match_score: float,
    skill_gap: list,
    is_model_predicted: bool = True,
) -> str:
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

    if not is_model_predicted:
        summary += " (Skor berbasis skill matching karena karier ini belum ada di data training model.)"

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
def get_skillmap_result(
    cv_text: str,
    target_job: str = "",
    quiz_score: float = 80,
) -> dict:
    """
    Pipeline utama: extract → normalize_target_job → predict AI → gap → score → path → summary

    Catatan penting:
    - CAREER_CATEGORIES: semua ~100 kategori bersih (untuk /jobs & target_job user)
    - MODEL_CLASSES: subset yang dilatih model (sample >= 5)
    - Jika target_job ada di CAREER_CATEGORIES tapi bukan MODEL_CLASSES:
      → tetap hitung skill_gap dari job_skill_map (rule-based)
      → model_career_score = 0 (model tidak bisa prediksi ini)
    """
    # 1. Extract skills dari CV
    detected_skills = extract_skills_from_cv_text(cv_text)

    # 2. Normalize target_job → career_category
    normalized_target = normalize_target_job(target_job)

    # 3. AI prediction
    try:
        predicted_career, model_confidence = predict_career_from_skills(detected_skills)
    except Exception:
        predicted_career = None
        model_confidence = 0.0

    # 4. Tentukan recommended_career
    #    Priority: normalized_target (jika ada di job_skill_map) → predicted_career → fallback
    if normalized_target and normalized_target in job_skill_map:
        recommended_career = normalized_target
        # Jika target_job ada di MODEL_CLASSES, gunakan model confidence
        # Jika tidak, model_career_score = 0 (fallback rule-based)
        if normalized_target not in MODEL_CLASSES:
            effective_model_score = 0.0
            is_model_predicted = False
        else:
            effective_model_score = model_confidence
            is_model_predicted = True
    elif predicted_career and predicted_career in job_skill_map:
        recommended_career = predicted_career
        effective_model_score = model_confidence
        is_model_predicted = True
    else:
        recommended_career = normalized_target or predicted_career or "unknown"
        effective_model_score = 0.0
        is_model_predicted = False

    # 5. Skill gap
    owned_skills, skill_gap, required_skills = detect_skill_gap(
        detected_skills, recommended_career
    )

    # 6. Score
    scores = calculate_readiness_score(
        owned_skills=owned_skills,
        required_skills=required_skills,
        model_career_score=effective_model_score,
        quiz_score=quiz_score,
    )

    # 7. Learning path & summary
    learning_path = generate_learning_path(skill_gap)
    summary = generate_summary(
        recommended_career=recommended_career,
        career_match_score=scores["career_match_score"],
        skill_gap=skill_gap,
        is_model_predicted=is_model_predicted,
    )

    return {
        "detected_skills_from_cv": detected_skills,
        "target_job": normalized_target if normalized_target else (target_job.strip() or None),
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