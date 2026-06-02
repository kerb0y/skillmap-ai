"""
SkillMap AI - Inference Module v3.1
=====================================
Perbaikan v3.1:
- extract_skills_from_cv_text: sort by length desc, cegah sub-phrase shadowing
- SKILL_ALIASES diperluas (microsoft excel, warehouse management, dll)
- KNOWN_SKILLS_EXTRA: skills penting yang wajib ada, langsung hardcode di modul
- recommend_career_rule_based: hybrid keyword (ID+EN) + skill overlap
- get_skillmap_result: 70% rule-based + 30% model saat target_job kosong
- raw_model_prediction dipisah dari recommended_career
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
    _known_skills_raw = json.load(f)

with open(ARTIFACT_DIR / "career_categories.json", "r", encoding="utf-8") as f:
    CAREER_CATEGORIES: list = json.load(f)

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
    # Normalisasi umum
    "problem solving": "problem solving skills",
    "analytic thinking": "analytical skills",
    "data visualization skills( power bi/ tableau )": "data visualization",
    "programming language skills": "programming",
    "logical skills": "logical thinking",
    "work under pressure": "stress management",
    # Microsoft Office aliases
    "microsoft excel": "excel",
    "ms excel": "excel",
    "ms office": "microsoft office",
    "microsoft office word": "microsoft office",
    "ms word": "microsoft office",
    "ms powerpoint": "presentation skills",
    # Logistik & gudang
    "warehouse management system": "warehouse management",
    "wms": "warehouse management",
    "stock management": "inventory management",
    "stock control": "inventory management",
    "inventory control": "inventory management",
    # Supply chain
    "supply chain management": "supply chain",
    "scm": "supply chain",
    # Keuangan
    "financial management": "finance related skills",
    "financial reporting": "reporting",
    # Umum
    "interpersonal communication": "interpersonal skills",
    "time management skills": "time management",
    "communication": "communication skills",
    "analytical thinking": "analytical skills",
    # ── v3.2: Alias bahasa Indonesia → English (dari CSV3) ──────
    "analisis data":     "analytical skills",
    "pemasaran digital": "digital marketing",
    "kepemimpinan":      "leadership",
    "komunikasi":        "communication skills",
    "manajemen proyek":  "project management",
    "riset":             "research skills",
    "desain grafis":     "designing skills",
    # ── v3.2: Alias kehutanan/lingkungan ────────────────────────
    "kehutanan":            "forestry",
    "hutan":               "forestry",
    "konservasi":          "conservation",
    "pemetaan":            "mapping",
    "survei lapangan":     "field survey",
    "pengamatan lapangan": "field survey",
    "vegetasi":            "vegetation analysis",
    "biodiversitas":       "biodiversity",
    "keberlanjutan":       "sustainability",
    "perubahan iklim":     "climate change",
    "analisis lingkungan": "environmental analysis",
    "penginderaan jauh":   "spatial analysis",
}

# Skill 2-char valid (nama teknologi)
KNOWN_SHORT_SKILLS = {"c#", "js", "hr", "ml", "ai", "go", "c++"}

# Expand alias 1-char ke nama bermakna
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

# ── Skills penting yang dijamin ada di extraction ──────────────────────────────
# Skills ini sering ada di CV non-IT tapi mungkin tidak ada di known_skills.json
# (karena training data bias ke IT). Hardcode di sini sebagai fallback.
KNOWN_SKILLS_EXTRA = {
    # Logistics / Warehouse / Supply Chain
    "logistics", "inventory management", "warehouse management",
    "supply chain", "shipping", "delivery", "distribution",
    "forklift operation", "stock opname", "packing",
    # Admin / Office
    "data entry", "filing", "scheduling", "time management",
    "attention to detail", "organizational skills", "microsoft office",
    "typing", "multitasking", "phone etiquette",
    # Finance / Accounting
    "accounting skills", "finance related skills", "budgeting",
    "reporting", "tax knowledge", "compliance", "auditing",
    "financial analysis", "taxation",
    # Sales / Customer
    "sales", "negotiation skills", "customer service", "crm",
    "product knowledge", "relationship management",
    # HR / Training
    "recruitment", "interviewing", "people management",
    "curriculum development", "facilitation",
    # Operations / Production
    "quality control", "quality management", "lean manufacturing",
    "production planning", "safety management", "regulatory compliance",
    "process improvement",
    # Healthcare
    "nursing", "patient care", "medical knowledge", "empathy",
    "clinical skills", "patient counseling", "lab techniques",
    # Legal
    "legal knowledge", "legal research", "compliance",
    # Creative / Marketing
    "copywriting", "content creation", "social media marketing",
    "seo", "google analytics", "creativity", "photography",
    "video editing", "photoshop", "illustrator", "figma",
    # Engineering Non-IT
    "autocad", "technical drawing", "circuit design",
    "electrical engineering", "mechanical skills", "surveying",
    # Education
    "teaching", "subject knowledge", "curriculum development",
    "classroom management", "mentoring",
    # General soft skills
    "adaptability", "critical thinking", "resilience",
    "stakeholder management", "vendor management", "procurement",
    "market research", "presentation skills", "leadership",
    "teamwork", "interpersonal skills",
    # ─ v3.2: Kehutanan/Lingkungan ────────────────────────────
    "forestry", "forest management", "conservation", "biodiversity",
    "field survey", "gis", "mapping", "vegetation analysis",
    "environmental analysis", "environmental monitoring",
    "regulation knowledge", "sustainability", "ecosystem monitoring",
    "community engagement", "qgis", "gps", "spatial analysis",
    "climate change", "documentation",
    # ─ v3.2: Animator/Diplomat/Consultant ─────────────────────
    "animation", "3d modeling", "cultural knowledge", "language skills",
    "storytelling", "after effects", "business knowledge",
    "research skills", "designing skills",
}

# Gabungkan known_skills dari JSON + EXTRA, lalu sort panjang descending
# (panjang pertama agar "inventory management" cocok sebelum "management")
known_skills: list = sorted(
    list(set([s.lower().strip() for s in _known_skills_raw] + list(KNOWN_SKILLS_EXTRA))),
    key=len,
    reverse=True,  # ← kunci: panjang dulu agar multi-word match sebelum sub-word
)


# ============================================================
# Skill validation & normalization
# ============================================================
def normalize_skill(skill: str) -> str:
    skill = str(skill).lower().strip().strip(";,. ")
    skill = SKILL_EXPAND.get(skill, skill)
    return SKILL_ALIASES.get(skill, skill)


def is_valid_skill(skill: str) -> bool:
    """Min 3 karakter, bukan angka/simbol/noise. Whitelist c++/c#."""
    s = str(skill).strip().lower()
    if s in KNOWN_SHORT_SKILLS or s in {"c++", "c#"}:
        return True
    if s in INVALID_VALUES:
        return False
    if len(s) < 3:
        return False
    if s.replace(".", "", 1).replace(",", "", 1).isdigit():
        return False
    if sum(1 for c in s if c.isalpha()) < 2:
        return False
    return True


def normalize_skill_list(skills: list) -> list:
    """Normalize + filter + deduplicate skill list."""
    result = []
    for s in skills:
        normalized = normalize_skill(str(s))
        if is_valid_skill(normalized):
            result.append(normalized)
    return sorted(list(set(result)))


# ============================================================
# Skill extraction dari CV text
# ============================================================
def extract_skills_from_cv_text(cv_text: str) -> list:
    """
    Ekstrak skill dari teks CV.
    - known_skills sudah di-sort panjang descending → phrase panjang match duluan
    - Lacak matched_positions untuk cegah sub-word shadowing
      (contoh: "inventory management" match → "management" tidak di-add lagi)
    """
    cv_lower = cv_text.lower()
    detected = []
    matched_positions: set = set()

    for skill in known_skills:
        skill_norm = normalize_skill(skill)
        if not is_valid_skill(skill_norm):
            continue
        pattern = r"\b" + re.escape(skill_norm) + r"\b"
        for match in re.finditer(pattern, cv_lower):
            span = set(range(match.start(), match.end()))
            # Hanya tambahkan jika span ini belum di-cover oleh phrase yang lebih panjang
            if not span & matched_positions:
                detected.append(skill_norm)
                matched_positions |= span
                break  # satu match per skill cukup

    return normalize_skill_list(detected)


# ============================================================
# Career keyword map (bilingual ID+EN) untuk rule-based matching
# ============================================================
CAREER_KEYWORD_MAP: dict = {
    # ── Logistics / Warehouse / Supply Chain ───────────────────
    "logistics staff": [
        "logistik", "logistics", "pengiriman", "shipment", "delivery",
        "distribusi", "distribution", "ekspedisi", "freight", "cargo",
        "kurir", "courier", "stok", "gudang", "packing", "supply chain",
    ],
    "warehouse staff": [
        "gudang", "warehouse", "stok", "stock opname", "inventory",
        "packing", "barang masuk", "barang keluar", "keluar masuk",
        "penerimaan barang", "receiving", "forklift", "pergudangan",
    ],
    "supply chain staff": [
        "supply chain", "rantai pasok", "scm", "procurement",
        "purchasing", "logistics", "distribution", "inventory control",
        "vendor", "sourcing",
    ],
    "procurement staff": [
        "procurement", "pengadaan", "purchasing", "pembelian", "buyer",
        "sourcing", "tender", "vendor", "supplier",
    ],
    "operations staff": [
        "operasional", "operations", "operational", "proses bisnis",
        "business process", "back office",
    ],
    # ── Finance / Accounting ───────────────────────────────────
    "accountant": [
        "akuntan", "accountant", "akuntansi", "accounting",
        "pembukuan", "bookkeeping", "jurnal", "ledger", "cpa",
    ],
    "finance staff": [
        "keuangan", "finance", "finansial", "treasury", "anggaran",
        "budgeting", "cash flow", "arus kas",
    ],
    "tax staff": [
        "pajak", "tax", "taxation", "perpajakan", "pph", "ppn",
        "spt", "fiscal", "brevet",
    ],
    "auditor": [
        "audit", "auditor", "internal audit", "pemeriksaan",
        "internal control", "risk assessment",
    ],
    "financial analyst": [
        "financial analyst", "analis keuangan", "valuation",
        "investment banking", "portofolio",
    ],
    "banking staff": [
        "bank", "banking", "perbankan", "teller", "kredit", "credit",
        "bank officer", "dana",
    ],
    "insurance agent": [
        "asuransi", "insurance", "premi", "klaim", "claim", "underwriting",
    ],
    # ── HR / Admin ─────────────────────────────────────────────
    "human resources": [
        "hrd", "hr", "human resource", "sumber daya manusia", "sdm",
        "rekrutmen", "recruitment", "personalia", "payroll", "penggajian",
    ],
    "recruiter": [
        "rekrutmen", "recruitment", "headhunter", "talent acquisition",
        "sourcing kandidat", "linkedin recruiter",
    ],
    "admin officer": [
        "admin", "administrasi", "administration", "kesekretariatan",
        "surat menyurat", "office administration", "back office",
    ],
    "secretary": [
        "sekretaris", "secretary", "personal assistant", "asisten pribadi",
        "pa ", "personal assistant",
    ],
    "receptionist": [
        "resepsionis", "receptionist", "front desk", "front office",
        "penerima tamu",
    ],
    # ── Sales / Marketing / Customer ───────────────────────────
    "sales executive": [
        "sales", "penjualan", "salesperson", "wiraniaga", "agen penjualan",
        "account executive", "field sales", "territory",
    ],
    "sales manager": [
        "sales manager", "manajer penjualan", "sales director",
        "area manager", "regional manager",
    ],
    "marketing staff": [
        "marketing", "pemasaran", "brand", "promosi", "campaign",
        "riset pasar", "market research",
    ],
    "digital marketing": [
        "digital marketing", "pemasaran digital", "seo", "sem",
        "google ads", "social media ads", "email marketing", "ppc",
    ],
    "social media specialist": [
        "social media", "media sosial", "instagram", "tiktok", "twitter",
        "facebook", "konten media sosial", "content strategy",
    ],
    "customer service": [
        "customer service", "layanan pelanggan", "cs", "customer care",
        "customer support", "keluhan pelanggan", "complaint handling",
    ],
    "call center agent": [
        "call center", "contact center", "telemarketing", "telecaller",
        "inbound", "outbound",
    ],
    "business development": [
        "business development", "pengembangan bisnis", "bd", "bizdev",
        "kemitraan", "partnership",
    ],
    # ── Creative / Media ───────────────────────────────────────
    "graphic designer": [
        "desainer grafis", "graphic design", "graphic designer",
        "photoshop", "illustrator", "canva", "coreldraw", "desain visual",
    ],
    "video editor": [
        "video editor", "editor video", "premiere pro", "after effects",
        "motion graphics", "videografi", "videography",
    ],
    "content creator": [
        "content creator", "kreator konten", "youtuber", "vlogger",
        "konten digital", "content creation",
    ],
    "copywriter": [
        "copywriter", "penulis iklan", "copywriting", "advertising copy",
        "tagline", "naskah iklan",
    ],
    "journalist": [
        "jurnalis", "journalist", "wartawan", "reporter", "redaksi",
        "media", "berita", "news",
    ],
    "photographer": [
        "fotografer", "photographer", "fotografi", "photography",
        "studio foto", "photo editing",
    ],
    "public relations": [
        "public relation", "humas", "pr ", "komunikasi publik",
        "media relation", "press release", "corporate communication",
    ],
    # ── Engineering Non-IT ─────────────────────────────────────
    "civil engineer": [
        "teknik sipil", "civil engineer", "konstruksi", "construction",
        "struktur", "structural", "jembatan", "gedung",
    ],
    "mechanical engineer": [
        "teknik mesin", "mechanical engineer", "mesin", "hvac",
        "piping", "maintenance mekanik",
    ],
    "electrical engineer": [
        "teknik elektro", "electrical engineer", "listrik", "power system",
        "plc", "panel", "instalasi listrik",
    ],
    "industrial engineer": [
        "teknik industri", "industrial engineer", "lean", "six sigma",
        "proses produksi", "efisiensi",
    ],
    "architect": [
        "arsitek", "architect", "arsitektur", "architecture",
        "interior design", "desain bangunan",
    ],
    # ── Quality / Production ───────────────────────────────────
    "quality control": [
        "quality control", "qc", "kontrol kualitas", "inspeksi",
        "inspection", "pengujian produk",
    ],
    "quality assurance engineer": [
        "quality assurance", "qa", "jaminan kualitas",
        "testing", "pengujian", "tester",
    ],
    "production staff": [
        "produksi", "production", "pabrik", "manufaktur",
        "lini produksi", "assembly",
    ],
    "safety officer": [
        "k3", "hse", "keselamatan kerja", "safety", "kesehatan kerja",
        "health safety environment", "ahli k3",
    ],
    # ── IT (tetap ada sebagai pembanding) ──────────────────────
    "software engineer": [
        "software engineer", "programmer", "backend", "frontend",
        "fullstack", "developer", "coding", "ngoding",
    ],
    "data analyst": [
        "data analyst", "analis data", "business intelligence",
        "bi analyst", "reporting analyst",
    ],
    "web developer": [
        "web developer", "web programming", "frontend developer",
        "backend developer", "website",
    ],
    "data scientist": [
        "data scientist", "ilmu data", "machine learning engineer",
    ],
    # ── Education ──────────────────────────────────────────────
    "teacher": [
        "guru", "teacher", "pengajar", "pendidik", "mengajar",
        "sekolah", "school",
    ],
    "lecturer": [
        "dosen", "lecturer", "professor", "akademisi", "universitas",
        "perguruan tinggi",
    ],
    "tutor": [
        "tutor", "les privat", "bimbel", "bimbingan belajar",
        "private teacher",
    ],
    # ── Healthcare ─────────────────────────────────────────────
    "nurse": [
        "perawat", "nurse", "nursing", "keperawatan", "rs ",
        "rumah sakit", "klinik", "pasien",
    ],
    "doctor": [
        "dokter", "doctor", "physician", "medis", "kedokteran",
        "klinik", "praktek",
    ],
    "pharmacist": [
        "apoteker", "pharmacist", "apotek", "pharmacy", "obat",
        "farmasi",
    ],
    "healthcare staff": [
        "tenaga kesehatan", "healthcare", "medis", "klinik",
        "hospital", "rumah sakit",
    ],
    # ── Hospitality / F&B ──────────────────────────────────────
    "chef / cook": [
        "chef", "cook", "koki", "masak", "kuliner", "dapur",
        "pastry", "restoran",
    ],
    "barista": [
        "barista", "kopi", "coffee", "cafe", "kedai kopi",
    ],
    "hospitality staff": [
        "hospitality", "hotel", "perhotelan", "front office hotel",
        "akomodasi",
    ],
    # ── Legal ──────────────────────────────────────────────────
    "legal staff": [
        "hukum", "legal", "lawyer", "pengacara", "advokat",
        "kontrak", "perjanjian",
    ],
    # ── Baru v3.2: dari CSV3 ───────────────────────────────────
    "animator": [
        "animasi", "animation", "animator", "3d", "motion graphics",
        "vfx", "rigging", "rendering", "character design", "storyboard",
    ],
    "diplomat": [
        "diplomat", "diplomasi", "hubungan internasional", "kedutaan",
        "kementerian luar negeri", "embassy", "konsulat", "foreign affairs",
        "bilateral", "multilateral",
    ],
    "business consultant": [
        "konsultan", "consultant", "advisory", "management consulting",
        "business advisory", "konsultasi bisnis", "strategic consulting",
    ],
    # ── Baru v3.2: Kehutanan/Lingkungan ────────────────────────
    "forestry officer": [
        "kehutanan", "forestry", "hutan", "silvicultur", "perhutani",
        "klhk", "kementerian lingkungan", "hutan tanaman", "hti",
        "reboisasi", "konservasi hutan",
    ],
    "environmental officer": [
        "lingkungan hidup", "environmental", "amdal", "reklamasi",
        "pencemaran", "teknik lingkungan", "analisis dampak",
        "kementerian lingkungan hidup", "lingkungan hidup", "ekologi",
    ],
    "conservation officer": [
        "konservasi", "conservation", "biodiversitas", "biodiversity",
        "vegetasi", "ekosistem", "taman nasional", "suaka margasatwa",
        "keanekaragaman hayati", "satwa liar",
    ],
    "sustainability officer": [
        "keberlanjutan", "sustainability", "perubahan iklim", "climate change",
        "emisi", "carbon", "net zero", "esg", "lingkungan berkelanjutan",
        "sdgs", "green",
    ],
    "gis analyst": [
        "gis", "qgis", "arcgis", "pemetaan", "mapping", "spatial",
        "remote sensing", "penginderaan jauh", "sig ", "sistem informasi geografis",
        "kartografi", "geospasial",
    ],
    "field officer": [
        "lapangan", "field", "survei lapangan", "field survey",
        "pengamatan lapangan", "monitoring lapangan", "petugas lapangan",
        "field monitoring", "enumerator",
    ],
}


# ============================================================
# Rule-based career recommender (hybrid keyword + skill overlap)
# ============================================================
def recommend_career_rule_based(
    cv_text: str,
    detected_skills: list,
) -> tuple:
    """
    Rekomendasikan career berdasarkan:
    1. Keyword matching (bilingual ID+EN) dari CAREER_KEYWORD_MAP
    2. Skill overlap dengan job_skill_map

    Returns: (career_category, score_0_to_100)
    """
    cv_lower = cv_text.lower()
    detected_set = set(detected_skills)
    scores: dict = {}

    for career in CAREER_CATEGORIES:
        # ── (1) Keyword score ──────────────────────────────────
        keywords = CAREER_KEYWORD_MAP.get(career, [career])
        kw_hits = sum(1 for kw in keywords if kw in cv_lower)
        keyword_score = min(kw_hits / max(len(keywords), 1), 1.0)

        # ── (2) Skill overlap score ────────────────────────────
        required = job_skill_map.get(career, [])
        if required:
            overlap = sum(1 for s in required if s in detected_set)
            skill_score = overlap / len(required)
        else:
            skill_score = 0.0

        # ── Combined: keyword lebih dominan saat target_job kosong ──
        combined = (0.55 * keyword_score) + (0.45 * skill_score)
        scores[career] = combined

    if not scores:
        return None, 0.0

    best_career = max(scores, key=scores.get)
    best_score = scores[best_career] * 100
    return best_career, round(best_score, 2)


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

    # 1. Exact match
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
    if best_score == 1 and best_cat:
        matched_word = list(raw_words & set(best_cat.split()))
        if matched_word and len(matched_word[0]) >= 5:
            return best_cat

    # 4. Keyword-based fallback mapping (untuk input user saat target_job diisi)
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
        # ─ v3.2 new categories ────────────────────────────────────
        (["animator", "animasi", "3d artist", "motion graphic"], "animator"),
        (["diplomat", "diplomasi", "foreign affairs", "hubungan internasional"], "diplomat"),
        (["konsultan", "consultant", "business consultant", "advisory"], "business consultant"),
        (["kehutanan", "forestry", "perhutani", "hutan"], "forestry officer"),
        (["lingkungan hidup", "environmental", "amdal", "teknik lingkungan"], "environmental officer"),
        (["konservasi", "conservation", "taman nasional", "biodiversitas"], "conservation officer"),
        (["sustainability", "keberlanjutan", "climate change", "perubahan iklim"], "sustainability officer"),
        (["gis", "qgis", "arcgis", "pemetaan", "geospasial"], "gis analyst"),
        (["field officer", "petugas lapangan", "survei lapangan", "enumerator"], "field officer"),
    ]

    for keywords, category in kw_map:
        if any(kw in raw for kw in keywords):
            if category in CAREER_CATEGORIES:
                return category

    return None


# ============================================================
# Core inference functions
# ============================================================
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
    recommendation_source: str = "hybrid",
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

    if recommendation_source == "rule_based":
        summary += " (Rekomendasi berbasis skill & keyword matching.)"
    elif recommendation_source == "model":
        summary += " (Rekomendasi dari model AI.)"

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
    Pipeline utama v3.1:

    A. target_job DIISI:
       → normalize ke career_category
       → gunakan sebagai recommended_career
       → AI prediction sebagai informasi tambahan

    B. target_job KOSONG:
       → jalankan recommend_career_rule_based (keyword + skill overlap, 55/45)
       → jalankan model AI prediction
       → hybrid: jika rule_based_score >= 20%, gunakan rule_based sebagai recommended
       → final_model_score = 0.70 * rule_score + 0.30 * model_confidence
       → raw_model_prediction tetap ditampilkan tapi bukan main output
    """
    quiz_score = min(max(float(quiz_score), 0), 100)

    # 1. Extract skills dari CV
    detected_skills = extract_skills_from_cv_text(cv_text)

    # 2. Normalize target_job → career_category
    normalized_target = normalize_target_job(target_job)

    # 3. AI prediction (selalu jalan sebagai info tambahan)
    try:
        raw_model_prediction, model_confidence = predict_career_from_skills(detected_skills)
    except Exception:
        raw_model_prediction = None
        model_confidence = 0.0

    # ── Jalur A: target_job diisi ──────────────────────────────
    if normalized_target and normalized_target in job_skill_map:
        recommended_career = normalized_target
        recommendation_source = "user_input"
        if normalized_target in MODEL_CLASSES:
            effective_model_score = model_confidence
        else:
            effective_model_score = 0.0

    # ── Jalur B: target_job kosong → hybrid recommendation ────
    else:
        rule_career, rule_score = recommend_career_rule_based(cv_text, detected_skills)

        # Threshold: rule_based harus minimal 15% confident untuk menang atas model
        RULE_THRESHOLD = 15.0

        if rule_career and rule_score >= RULE_THRESHOLD:
            recommended_career = rule_career
            recommendation_source = "rule_based"
            # Jika model setuju dengan rule → boost score
            if raw_model_prediction == rule_career:
                effective_model_score = (0.70 * rule_score) + (0.30 * model_confidence)
                recommendation_source = "hybrid"
            else:
                # Rule menang, model disagreed → gunakan rule_score sebagai model proxy
                effective_model_score = rule_score * 0.70
        elif raw_model_prediction and raw_model_prediction in job_skill_map:
            # Rule tidak confident → fallback ke model
            recommended_career = raw_model_prediction
            recommendation_source = "model"
            effective_model_score = model_confidence
        else:
            recommended_career = rule_career or "unknown"
            recommendation_source = "rule_based"
            effective_model_score = 0.0

    # 4. Skill gap
    owned_skills, skill_gap, required_skills = detect_skill_gap(
        detected_skills, recommended_career
    )

    # 5. Score
    scores = calculate_readiness_score(
        owned_skills=owned_skills,
        required_skills=required_skills,
        model_career_score=effective_model_score,
        quiz_score=quiz_score,
    )

    # 6. Learning path & summary
    learning_path = generate_learning_path(skill_gap)
    summary = generate_summary(
        recommended_career=recommended_career,
        career_match_score=scores["career_match_score"],
        skill_gap=skill_gap,
        recommendation_source=recommendation_source,
    )

    return {
        "detected_skills_from_cv": detected_skills,
        "target_job": normalized_target if normalized_target else (target_job.strip() or None),
        "recommended_career": recommended_career,
        "recommendation_source": recommendation_source,
        "raw_model_prediction": raw_model_prediction,
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