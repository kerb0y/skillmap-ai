"""
SkillMap AI - Training Script v3.0
====================================
Dual-dataset, expanded career taxonomy (100+ categories),
pisah career_categories (untuk /jobs) vs model_classes (untuk training),
rule-based skill map fallback untuk coverage penuh.

Jalankan:
    python train_model.py
"""

import json
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer


# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATASET1_PATH = BASE_DIR / "combined_career_recommender_clean.csv"
DATASET2_PATH = BASE_DIR / "career_recommender_indonesia_cleaned.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)


# ============================================================
# 1.  Expanded Career Taxonomy  (raw → career_category)
#     Coverage: IT + non-IT, ~100+ career categories
# ============================================================
CAREER_TAXONOMY = {
    # ─── IT / Software ────────────────────────────────────────
    "software engineer": [
        "software engineer", "software developer", "computer software engineer",
        "insinyur perangkat lunak", "teknologi specialist",
        "software development engineer", "software systems engineer",
        "entry level software engineer", "software engineer 1",
        "junior software engineer", "senior software engineer",
        "software programmer", "specialist programmer",
        "associate software engineer", "systems engineer",
        "system engineer", "associate engineer",
    ],
    "web developer": [
        "web developer", "pengembang web", "front end developer",
        "front end java developer", "back end developer", "back-end developer",
        "full stack developer", "java full stack developer",
        "dot net developer", "asp.net developer", "php developer",
        "crm technical developer", "sql developer", "applications developer",
        "application developer", "application development analyst",
        "web developer and digital marketer", "programmer analyst",
        "java developer", "python developer",
    ],
    "mobile developer": [
        "mobile application developer", "mobile applications developer",
        "ios developer", "android developer", "flutter developer",
    ],
    "data analyst": [
        "data analyst", "data analysis", "senior pmo - data analyst",
        "business intelligence analyst", "crm business analyst",
        "e-commerce analyst", "keuangan specialist",
        "statistical programmer", "process analyst", "mis executive",
        "mis", "sr. executive in data management",
    ],
    "data scientist": [
        "data scientist", "data scientist-intern", "associate data scientist",
        "researcher", "research fellow", "data science and game design trainer",
    ],
    "data engineer": [
        "data engineer", "database manager", "data architect",
        "pengembang database", "sr. executive in data management",
    ],
    "database administrator": [
        "database administrator", "lotus notes admin",
        "database manager",
    ],
    "machine learning engineer": [
        "machine learning engineer", "ml engineer", "mechine learning engineer",
        "product engineer (ai/ml)", "rpa developer",
        "video labeling",
    ],
    "ai engineer": [
        "ai engineer", "artificial intelligence engineer",
        "computer vision engineer", "nlp engineer",
    ],
    "devops engineer": [
        "devops engineer", "associate engineer - devops",
        "aws developer", "aws architect", "system administrator",
        "linux administrator", "associate engineer - controls & software",
    ],
    "cloud engineer": [
        "cloud engineer", "cloud architect", "cloud consultant",
        "aws engineer", "azure engineer", "gcp engineer",
    ],
    "it support": [
        "it support", "desktop support engineer",
        "technical services/help desk/tech support", "dukungan teknis",
        "tech support", "technical associate",
        "technical engineer", "it operations", "it engineer",
    ],
    "network engineer": [
        "network engineer", "network analyst", "insinyur jaringan",
        "network security engineer", "network administrator",
    ],
    "cyber security analyst": [
        "cyber security analyst", "cyber security engineer",
        "information security analyst", "information secuirty analyst",
        "network security administrator", "administrator keamanan sistem",
        "information security analyst", "information technology auditor",
        "network security engineer",
    ],
    "ui ux designer": [
        "ui ux designer", "ux designer", "desainer ux", "design & ux",
        "desain specialist", "user experience designer",
        "user interface designer", "interface designer",
    ],
    "embedded engineer": [
        "embedded system engineer", "embedded developer", "vlsi developer",
        "vlsi digital design", "hardware engineer", "hardware design engineer",
        "hardware pcb design engineer", "instrumentation and control engineer",
        "plant instrumentation engineer", "electronics engineer",
        "engineer-power electronics",
    ],
    "quality assurance engineer": [
        "quality assurance", "qa analyst", "qa specialist",
        "quality analyst", "quality assurance associate",
        "quality assurances engineer", "quality engineer",
        "software quality assurance (qa) / testing",
        "software tester", "software testing", "testing engineer",
        "test analyst", "test engineer", "graduate quality engineer",
        "qc engineer", "trainee consultant - quality",
    ],
    "game developer": [
        "game programmer", "game developer", "game designer",
    ],
    "solutions architect": [
        "solutions architect", "it architect", "enterprise architect",
        "technical architect",
    ],
    "erp consultant": [
        "sap consultant", "erp implementation", "plm consultant",
        "plm engineer", "functional consultant",
        "associate it consultant - erp", "crm consultant",
    ],
    "technical writer": [
        "technical writer", "documentation specialist",
    ],
    # ─── Business / Management ────────────────────────────────
    "product manager": [
        "product manager", "product specialist",
        "product development associate",
    ],
    "project manager": [
        "project manager", "project engineer", "project coordinator",
        "project associate", "planning manager", "manajer proyek",
        "senior pmo", "program manager", "scrum master",
        "associate scm",
    ],
    "business analyst": [
        "business analyst", "business systems analyst",
        "bisnis specialist", "systems analyst",
        "associate consultant", "management trainee",
        "business trainee", "associate professional",
        "it consultant",
    ],
    "operations manager": [
        "operation manager", "operations management",
        "regional operations manager", "general manager",
        "information technology manager",
    ],
    "business development": [
        "business development manager", "business development associate",
        "business developer", "business consultant",
        "buisness devlopment excutive",
    ],
    "management trainee": [
        "management trainee", "graduate management trainee",
        "business trainee", "associate trainee",
    ],
    "entrepreneur": [
        "entrepreneur", "self employed", "managing director",
        "startup founder", "ceo", "co-founder",
    ],
    "general manager": [
        "general manager", "senior manager", "director",
        "national head", "country head",
    ],
    # ─── Marketing / Creative ─────────────────────────────────
    "digital marketing": [
        "digital marketing", "digital/social media marketer",
        "marketing executive", "seo specialist",
        "web developer and digital marketer",
        "digital marketing executive",
    ],
    "marketing staff": [
        "marketing manager", "marketing guy", "i m a marketing head",
        "marketing staff", "brand manager",
    ],
    "social media specialist": [
        "social media executive", "social media specialist",
        "social media marketer", "social media manager",
        "community manager",
    ],
    "content creator": [
        "content creator", "content writer",
        "content writer and creative strategist",
        "maths content developer",
    ],
    "copywriter": [
        "copywriter", "advertising copywriter",
    ],
    "public relations": [
        "public relations", "pr specialist",
        "communications specialist", "media relations",
        "publishing executive",
    ],
    "graphic designer": [
        "graphic designer", "graphics designer", "creative designer",
        "clothes designer",
    ],
    "video editor": [
        "video editor", "vfx artist", "vfx designer",
        "vfx production assistant", "animator",
        "video producer",
    ],
    "journalist": [
        "journalist", "crime reporter", "sub editor",
        "sub-editor", "junior sub editor", "senior editor",
        "news reporter", "mass communication",
    ],
    "writer": [
        "writer", "technical writer", "transcriber",
        "script editor", "author",
    ],
    "translator": [
        "translator", "interpreter",
    ],
    "event planner": [
        "event planner", "event organizer", "event manager",
        "event coordinator",
    ],
    "photographer": [
        "photographer", "photojournalist", "videographer",
    ],
    # ─── Sales / Customer ─────────────────────────────────────
    "sales executive": [
        "sales executive", "sale executive", "sales excellent",
        "field sales officer", "telecaller", "tele-caller",
        "telecaller,sale exicutive",
        "medical and sales reprasentative", "medical representative",
        "sales service engineer", "sales",
        "senior territory sales manager", "sr. territory sales manager",
    ],
    "sales manager": [
        "sales manager", "regional sales manager",
        "sales director", "area sales manager",
    ],
    "account executive": [
        "key account manager", "account executive",
        "account manager",
    ],
    "business development": [
        "business development executive",
    ],
    "customer service": [
        "customer service", "customer care",
        "customer operation associate",
        "flipkart customer care executive",
        "relationships manager", "relationship manager",
    ],
    "call center agent": [
        "call center agent", "tele-caller", "telecaller",
        "telemarketer", "customer support executive",
    ],
    "retail staff": [
        "retail manager", "store assistant",
        "store keeper",
    ],
    "store manager": [
        "store manager", "branch manager",
    ],
    "cashier": [
        "cashier", "teller",
    ],
    # ─── Finance / Accounting ─────────────────────────────────
    "accountant": [
        "accountant", "accountent", "school accountant",
        "chartered accountant", "computerized accounting",
        "office assistant and acoounts",
    ],
    "finance staff": [
        "finance manager", "asst finance manager",
        "finance consultant", "finance staff",
    ],
    "financial analyst": [
        "financial analyst", "investment banking analyst",
        "investment banking associate", "risk analyst",
        "pharma benefit analyst",
    ],
    "tax staff": [
        "tax consultant", "tax staff", "taxation officer",
    ],
    "auditor": [
        "auditor", "internal auditor", "external auditor",
        "information technology auditor",
    ],
    "banking staff": [
        "banking officer", "relationship manager at icici",
        "team lead in mortgage banking",
    ],
    "insurance agent": [
        "insurance agent", "life insurance", "insurance consultant",
    ],
    "investment analyst": [
        "investment analyst", "stock trader",
        "financial advisor",
    ],
    # ─── HR / Admin ───────────────────────────────────────────
    "human resources": [
        "hr", "hr executive", "hr admin", "hr assistant",
        "human resources assistant", "human resource manager",
    ],
    "recruiter": [
        "talent acquisition executive", "technical recruiter",
        "senior technical recruiter", "recruitment specialist",
        "talent acquisition specialist",
    ],
    "training staff": [
        "training and development", "traning and development",
        "learning & development",
    ],
    "admin officer": [
        "admin", "administrator", "administrator portal",
        "administrator keamanan sistem", "back office executive",
        "associate operations processor", "office administrator",
    ],
    "secretary": [
        "secretary", "personal assistant", "personal advisory",
        "executive assistant", "company secretary",
    ],
    "office staff": [
        "office staff", "office assistant",
        "receptionist in coaching center",
    ],
    "receptionist": [
        "receptionist", "front desk officer", "front office staff",
    ],
    # ─── Engineering Non-IT ───────────────────────────────────
    "civil engineer": [
        "civil engineer", "site civil engineer", "site engineer",
        "structural engineer", "civil & structural engineer",
        "civil design engineer", "sub engineer", "junior architect",
        "gis engineer",
    ],
    "mechanical engineer": [
        "mechanical engineer", "mechanical design engineer",
        "mechanical product developer", "mechanical supervisor",
        "maintenance engineer", "maintenance engineer-mechanical",
        "machine operator",
    ],
    "electrical engineer": [
        "electrical engineer", "electrician",
        "engineer-power electronics", "maintenance engineer-electrical",
        "sales service engineer-electrical",
    ],
    "industrial engineer": [
        "industrial engineer", "process engineer",
        "instrumentation and control engineer",
    ],
    "architect": [
        "architect", "junior architect", "interior designer",
        "junior interior designer",
    ],
    "drafter": [
        "drafter", "cad designer", "cad engineer",
    ],
    "surveyor": [
        "surveyor", "quantity surveyor",
    ],
    "technician": [
        "technician", "lab technician",
        "lab technician -food safety & quality",
    ],
    "automotive technician": [
        "automotive technician", "service advisor",
        "motor controller",
    ],
    # ─── Quality / Production / Manufacturing ─────────────────
    "quality control": [
        "quality control engineer", "quality control",
        "graduate quality engineer",
    ],
    "production staff": [
        "production engineer", "production supervisor",
        "production manager", "production specialist",
        "production field", "manufacturing engineer",
    ],
    "manufacturing staff": [
        "manufacturing staff", "machine operator",
        "process engineer", "process operations engineer",
    ],
    "safety officer": [
        "safety officer", "hse officer", "k3 officer",
    ],
    # ─── Operations / Logistics ───────────────────────────────
    "operations staff": [
        "operations specialist", "supervisor",
        "team leader", "senior associate",
    ],
    "logistics staff": [
        "logistics", "supply chain",
        "logistik", "ups supervisor",
    ],
    "warehouse staff": [
        "warehouse staff", "store keeper",
        "inventory staff",
    ],
    "supply chain staff": [
        "supply chain management", "associate scm",
        "material planner",
    ],
    "procurement staff": [
        "procurement engineer", "purchase executive",
        "ast.purchase manager", "sourcing specialist",
    ],
    "purchasing staff": [
        "purchasing staff", "purchasing officer",
        "buyer",
    ],
    # ─── Education / Research ─────────────────────────────────
    "teacher": [
        "teacher", "teaching", "dance teacher",
        "teaching experience", "teaching job",
        "swim coach", "swimming coach",
    ],
    "lecturer": [
        "lecturer", "assistant professor",
        "professor", "associate professor",
    ],
    "tutor": [
        "tutor", "teaching assistant",
        "teaching assistant-biotechnology",
        "python instructor",
    ],
    "researcher": [
        "researcher", "research associate",
        "research fellow", "scientist",
        "biocurator", "r&d engineer",
    ],
    # ─── Healthcare / Science ─────────────────────────────────
    "doctor": [
        "doctor", "medical practitioner",
        "dental surgeon", "medical posting",
        "kesehatan specialist",
    ],
    "nurse": [
        "nurse", "nursing staff",
    ],
    "pharmacist": [
        "pharmacist", "pharmacy staff",
        "chemist in pharmaceutical",
    ],
    "laboratory analyst": [
        "lab analyst", "laboratory analyst",
        "lab technician", "biocurator",
        "scientist in downstream protein purification",
        "senior chemist",
    ],
    "healthcare staff": [
        "healthcare staff", "clinical research associate",
        "dietician", "psychologist",
        "physical therapist",
    ],
    # ─── Legal / Government ───────────────────────────────────
    "legal staff": [
        "legal associate", "law intern", "lawyer",
        "lawyer -ipr", "lawyer-forensic investigations",
        "litigation", "litigation lawyer",
        "corporate and disputes lawyer",
    ],
    "paralegal": [
        "paralegal", "legal assistant",
    ],
    "company secretary": [
        "company secretary",
    ],
    # ─── Hospitality / F&B ────────────────────────────────────
    "hospitality staff": [
        "hospitality staff", "hotel staff", "front office",
        "housekeeping", "hotel receptionist",
    ],
    "chef / cook": [
        "chef", "cook", "sous chef", "pastry chef",
    ],
    "barista": [
        "barista", "barista trainer",
    ],
    "tourism staff": [
        "tourism staff", "travel consultant", "tour guide",
    ],
}

# ============================================================
# 2.  Rule-based skill map untuk coverage /jobs yang lebih luas
#     (untuk categories dengan sample kecil atau tidak ada di dataset)
# ============================================================
RULE_BASED_SKILL_MAP = {
    "cashier": ["communication skills", "numerical skills", "customer service", "cash handling", "pos system"],
    "receptionist": ["communication skills", "microsoft office", "customer service", "scheduling", "phone etiquette"],
    "secretary": ["communication skills", "microsoft office", "time management", "organizational skills", "scheduling"],
    "office staff": ["microsoft office", "communication skills", "data entry", "organizational skills", "filing"],
    "retail staff": ["customer service", "communication skills", "sales", "product knowledge", "inventory management"],
    "store manager": ["leadership", "sales", "inventory management", "customer service", "team management"],
    "call center agent": ["communication skills", "active listening", "problem solving skills", "customer service", "crm"],
    "account executive": ["sales", "negotiation skills", "communication skills", "crm", "relationship management"],
    "business development": ["sales", "communication skills", "negotiation skills", "market research", "presentation skills"],
    "insurance agent": ["sales", "communication skills", "financial knowledge", "negotiation skills", "customer service"],
    "investment analyst": ["financial analysis", "excel", "analytical skills", "market research", "valuation"],
    "banking staff": ["finance related skills", "communication skills", "excel", "customer service", "compliance"],
    "tax staff": ["accounting skills", "taxation", "excel", "analytical skills", "compliance"],
    "auditor": ["accounting skills", "analytical skills", "excel", "risk management skills", "compliance"],
    "financial analyst": ["financial analysis", "excel", "analytical skills", "sql", "data visualization"],
    "management trainee": ["communication skills", "analytical skills", "leadership", "problem solving skills", "teamwork"],
    "training staff": ["communication skills", "presentation skills", "curriculum design", "facilitation", "analytical skills"],
    "recruiter": ["communication skills", "hr", "interviewing", "linkedin", "sourcing"],
    "general manager": ["leadership", "strategic planning", "communication skills", "financial management", "people management"],
    "operations manager": ["leadership", "operations management", "analytical skills", "problem solving skills", "communication skills"],
    "entrepreneur": ["leadership", "business knowledge", "communication skills", "problem solving skills", "financial management"],
    "translator": ["language skills", "communication skills", "writing skills", "cultural knowledge", "research skills"],
    "photographer": ["photography", "photo editing", "creative thinking", "lightroom", "composition"],
    "video editor": ["video editing", "premiere pro", "after effects", "creative thinking", "storytelling"],
    "event planner": ["communication skills", "project management", "organizational skills", "creativity", "vendor management"],
    "copywriter": ["writing skills", "creativity", "communication skills", "seo", "content strategy"],
    "journalist": ["writing skills", "communication skills", "research skills", "interviewing", "analytical skills"],
    "writer": ["writing skills", "creativity", "communication skills", "research skills", "storytelling"],
    "public relations": ["communication skills", "writing skills", "media relations", "social media", "crisis management"],
    "marketing staff": ["marketing", "communication skills", "analytical skills", "excel", "presentation skills"],
    "technical writer": ["writing skills", "communication skills", "technical knowledge", "documentation", "editing"],
    "erp consultant": ["sap", "erp", "business analysis", "communication skills", "project management"],
    "solutions architect": ["system design", "communication skills", "cloud computing", "programming", "technical consulting"],
    "ai engineer": ["machine learning skills", "python", "deep learning", "pytorch", "tensorflow"],
    "game developer": ["programming", "game development", "unity", "c++", "creative thinking"],
    "technical writer": ["writing skills", "documentation", "technical knowledge"],
    "architect": ["autocad", "design", "architectural design", "presentation skills", "project management"],
    "drafter": ["autocad", "cad", "technical drawing", "attention to detail"],
    "surveyor": ["surveying", "analytical skills", "technical drawing", "gis"],
    "technician": ["technical skills", "problem solving skills", "maintenance", "troubleshooting"],
    "automotive technician": ["mechanical skills", "diagnosis", "automotive repair", "technical skills"],
    "industrial engineer": ["analytical skills", "process improvement", "lean manufacturing", "autocad", "project management"],
    "safety officer": ["safety management", "risk assessment", "regulatory compliance", "communication skills", "analytical skills"],
    "manufacturing staff": ["manufacturing", "production planning", "quality control", "lean manufacturing"],
    "warehouse staff": ["inventory management", "logistics", "material handling", "organizational skills"],
    "supply chain staff": ["supply chain", "logistics", "analytical skills", "excel", "vendor management"],
    "purchasing staff": ["procurement", "negotiation skills", "vendor management", "analytical skills", "excel"],
    "teacher": ["communication skills", "teaching", "curriculum development", "patience", "subject knowledge"],
    "lecturer": ["research", "communication skills", "teaching", "academic writing", "subject expertise"],
    "tutor": ["communication skills", "teaching", "patience", "subject knowledge"],
    "researcher": ["research skills", "analytical skills", "writing skills", "data analysis", "critical thinking"],
    "doctor": ["medical knowledge", "diagnostic skills", "communication skills", "empathy", "clinical skills"],
    "nurse": ["nursing", "patient care", "communication skills", "medical knowledge", "empathy"],
    "pharmacist": ["pharmacy", "medical knowledge", "analytical skills", "communication skills"],
    "laboratory analyst": ["analytical skills", "lab techniques", "data analysis", "scientific writing"],
    "healthcare staff": ["medical knowledge", "communication skills", "patient care", "empathy"],
    "legal staff": ["legal knowledge", "analytical skills", "writing skills", "research skills", "communication skills"],
    "paralegal": ["legal research", "writing skills", "organizational skills", "analytical skills"],
    "company secretary": ["legal knowledge", "compliance", "communication skills", "organizational skills"],
    "hospitality staff": ["customer service", "communication skills", "teamwork", "problem solving skills"],
    "chef / cook": ["culinary skills", "time management", "creativity", "teamwork", "food safety"],
    "barista": ["customer service", "coffee knowledge", "communication skills", "teamwork"],
    "tourism staff": ["communication skills", "customer service", "language skills", "tourism knowledge"],
    "electrical engineer": ["electrical engineering", "circuit design", "autocad", "problem solving skills", "project management"],
    "drafter": ["autocad", "technical drawing", "design", "attention to detail"],
    "surveyor": ["surveying", "gis", "analytical skills", "technical skills"],
    "quality control": ["quality management", "analytical skills", "attention to detail", "problem solving skills"],
    "production staff": ["production planning", "manufacturing", "quality control", "lean manufacturing"],
}

# ============================================================
# 3.  Full list of career_categories (untuk /jobs)
#     Ini lebih luas dari model_classes
# ============================================================
ALL_CAREER_CATEGORIES = sorted(list(CAREER_TAXONOMY.keys()))

# ============================================================
# 4.  Text cleaning helpers
# ============================================================
SKILL_ALIASES = {
    "problem solving": "problem solving skills",
    "analytic thinking": "analytical skills",
    "data visualization skills( power bi/ tableau )": "data visualization",
    "programming language skills": "programming",
    "logical skills": "logical thinking",
    "work under pressure": "stress management",
    "language skills": "communication skills",
}

INVALID_SKILLS = {
    "tidak", "ya", "na", "nan", "not applicable",
    "belum bekerja", "student (unemployed)", "belum", "",
}

# Skill yang boleh pendek (2 char) karena ini nama teknologi valid
KNOWN_SHORT_SKILLS = {
    "c#", "js", "hr", "ml", "ai", "go", "c++",
}

# Ganti alias skill pendek ke versi lebih panjang yang bermakna
SKILL_EXPAND = {
    "r": "r programming",
    "c": "c programming",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "js": "javascript",
}

INVALID_JOB_PATTERNS = {
    "not working", "nothing", "previously worked", "still working",
    "housewife", "fruit business", "dancing", "general masdor",
    "gatemen", "miritary soldier", "loco pilot", "student onboarding",
    "worked in research institute", "young horticulture expert",
    "salem farmers producer", "listing, marketting",
    "i m a marketing head", "i was working as",
    "jumio document", "tata projects", "zigna ai",
    "mahendra next wealth", "mahindra and mahindra financial",
    "apprentice dp world", "felix healtcare", "cultfit",
    "articleship", "traniee", "tri", "pat", "get", "cs", "cis",
    "ba", "det", "se", "pc", "not applicable", "belum bekerja",
    "student (unemployed)", "currently", "pursuing", "studying",
    "diploma", "bachelor", "master", "m.tech", "m.sc", "m.com",
    "b.tech", "bba", "mba", "ma ", "pgdm", "b pharmacy",
    "commerce", "engineering degree", "computer science",
    "journalism & mass", "aeronautical engineering",
    "govt job", "miritary",
}


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("Â", "").replace("â€™", "'").replace("â€˜", "'")
    text = text.replace("Ã‰", "").replace("ÃŠ", "").replace("\r", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_skill(skill):
    skill = clean_text(skill).lower().strip().strip(";,. ")
    # Expand pendek alias ke versi bermakna
    skill = SKILL_EXPAND.get(skill, skill)
    return SKILL_ALIASES.get(skill, skill)


def is_valid_skill(skill: str) -> bool:
    """Validasi skill: min 3 karakter, bukan angka, bukan simbol, bukan noise."""
    s = str(skill).strip().lower()
    # Whitelist untuk 2-char dan special tech names yang valid
    if s in KNOWN_SHORT_SKILLS:
        return True
    if s in {"c++", "c#"}:
        return True
    # Filter noise
    if s in INVALID_SKILLS:
        return False
    if len(s) < 3:
        return False
    if s.replace(".", "", 1).replace(",", "", 1).isdigit():
        return False
    # Murni simbol / non-alphabetic
    alpha_count = sum(1 for c in s if c.isalpha())
    if alpha_count < 2:
        return False
    return True


def split_skills(raw_skills):
    """Split dan validasi raw skill string dari dataset."""
    if not isinstance(raw_skills, str) or raw_skills.strip() == "":
        return []
    parts = re.split(r"[;,\n]+", raw_skills)
    skills = []
    for part in parts:
        s = normalize_skill(part.strip())
        if is_valid_skill(s):
            skills.append(s)
    return list(set(skills))


def clean_job(job):

    job = clean_text(job).lower().strip().strip(";,. \"'")
    return job


def is_invalid_job(raw_job):
    raw = raw_job.lower().strip()
    for pattern in INVALID_JOB_PATTERNS:
        if pattern in raw:
            return True
    if len(raw) <= 2:
        return True
    return False


def normalize_job_to_category(raw_job):
    """
    Map raw job title → career_category.
    Returns: category string or None.
    """
    if is_invalid_job(raw_job):
        return None

    raw = raw_job.lower().strip()

    # Exact match on taxonomy values
    for category, keywords in CAREER_TAXONOMY.items():
        if raw in [kw.lower() for kw in keywords]:
            return category

    # Exact match on category name itself
    if raw in CAREER_TAXONOMY:
        return raw

    # Substring match
    for category, keywords in CAREER_TAXONOMY.items():
        for kw in keywords:
            kw_l = kw.lower()
            if kw_l in raw or raw in kw_l:
                return category

    # Keyword fallback
    if any(x in raw for x in ["software", "programmer"]):
        if "web" in raw or "front" in raw or "back" in raw or "full stack" in raw:
            return "web developer"
        if "mobile" in raw or "android" in raw or "ios" in raw:
            return "mobile developer"
        return "software engineer"
    if "data" in raw and "analyst" in raw:
        return "data analyst"
    if "data" in raw and "scientist" in raw:
        return "data scientist"
    if "data" in raw and "engineer" in raw:
        return "data engineer"
    if "machine learning" in raw or raw == "ml":
        return "machine learning engineer"
    if "devops" in raw:
        return "devops engineer"
    if "cloud" in raw and "engineer" in raw:
        return "cloud engineer"
    if "network" in raw and ("engineer" in raw or "analyst" in raw):
        return "network engineer"
    if "security" in raw or "cyber" in raw:
        return "cyber security analyst"
    if "ui" in raw or "ux" in raw or "user experience" in raw:
        return "ui ux designer"
    if "product manager" in raw:
        return "product manager"
    if "project manager" in raw or "project management" in raw:
        return "project manager"
    if "business analyst" in raw:
        return "business analyst"
    if "quality" in raw and ("engineer" in raw or "assurance" in raw or "control" in raw):
        return "quality assurance engineer"
    if "quality control" in raw:
        return "quality control"
    if "it support" in raw or "tech support" in raw or "help desk" in raw:
        return "it support"
    if "embedded" in raw or "vlsi" in raw or "firmware" in raw:
        return "embedded engineer"
    if "digital marketing" in raw:
        return "digital marketing"
    if "social media" in raw:
        return "social media specialist"
    if "content" in raw and "creator" in raw:
        return "content creator"
    if "copywriter" in raw:
        return "copywriter"
    if "graphic" in raw:
        return "graphic designer"
    if "video" in raw and ("editor" in raw or "vfx" in raw):
        return "video editor"
    if "journalist" in raw or "reporter" in raw:
        return "journalist"
    if "sales manager" in raw:
        return "sales manager"
    if "sales" in raw and ("executive" in raw or "officer" in raw or "representative" in raw):
        return "sales executive"
    if "sales" in raw:
        return "sales executive"
    if "customer service" in raw or "customer care" in raw or "customer support" in raw:
        return "customer service"
    if "call center" in raw or "telecaller" in raw or "tele-caller" in raw or "telemarketer" in raw:
        return "call center agent"
    if "accountant" in raw or "accounting" in raw:
        return "accountant"
    if "tax" in raw and ("consultant" in raw or "officer" in raw or "staff" in raw):
        return "tax staff"
    if "audit" in raw:
        return "auditor"
    if "finance manager" in raw or "financial analyst" in raw:
        return "financial analyst"
    if "finance" in raw or "keuangan" in raw:
        return "finance staff"
    if "hr" == raw or "human resource" in raw:
        return "human resources"
    if "recruiter" in raw or "talent acquisition" in raw or "recruitment" in raw:
        return "recruiter"
    if "training" in raw and "development" in raw:
        return "training staff"
    if "civil" in raw and "engineer" in raw:
        return "civil engineer"
    if "mechanical" in raw and "engineer" in raw:
        return "mechanical engineer"
    if "electrical" in raw and "engineer" in raw:
        return "electrical engineer"
    if "industrial" in raw and "engineer" in raw:
        return "industrial engineer"
    if "production" in raw and ("engineer" in raw or "supervisor" in raw or "manager" in raw):
        return "production staff"
    if "manufacturing" in raw:
        return "manufacturing staff"
    if "maintenance" in raw and "engineer" in raw:
        return "mechanical engineer"
    if "safety" in raw and ("officer" in raw or "manager" in raw):
        return "safety officer"
    if "procurement" in raw or "purchasing" in raw:
        return "procurement staff"
    if "supply chain" in raw or "logistics" in raw:
        return "logistics staff"
    if "warehouse" in raw or "gudang" in raw:
        return "warehouse staff"
    if "teacher" in raw or "teaching" in raw:
        return "teacher"
    if "lecturer" in raw or "professor" in raw:
        return "lecturer"
    if "tutor" in raw or "teaching assistant" in raw:
        return "tutor"
    if "researcher" in raw or "research" in raw:
        return "researcher"
    if "doctor" in raw or "medical practitioner" in raw or "dental" in raw:
        return "doctor"
    if "nurse" in raw or "nursing" in raw:
        return "nurse"
    if "pharmacist" in raw or "pharmacy" in raw:
        return "pharmacist"
    if "health" in raw:
        return "healthcare staff"
    if "kesehatan" in raw:
        return "healthcare staff"
    if "lawyer" in raw or "legal" in raw or "advocate" in raw or "litigation" in raw:
        return "legal staff"
    if "admin" in raw or "administrator" in raw:
        return "admin officer"
    if "secretary" in raw or "personal assistant" in raw:
        return "secretary"
    if "receptionist" in raw:
        return "receptionist"
    if "entrepreneur" in raw or "self employed" in raw:
        return "entrepreneur"
    if "database" in raw and "administrator" in raw:
        return "database administrator"
    if "erp" in raw or "sap" in raw:
        return "erp consultant"
    if "architect" in raw and "software" not in raw and "cloud" not in raw:
        return "architect"
    if "solutions architect" in raw or "it architect" in raw:
        return "solutions architect"
    if "game" in raw:
        return "game developer"
    if "public relation" in raw:
        return "public relations"
    if "marketing" in raw:
        return "marketing staff"
    if "event" in raw:
        return "event planner"
    if "chef" in raw or "cook" in raw or "barista" in raw:
        return "chef / cook"
    if "hotel" in raw or "hospitality" in raw:
        return "hospitality staff"
    if "tourism" in raw or "travel" in raw:
        return "tourism staff"
    if "store manager" in raw:
        return "store manager"
    if "retail" in raw:
        return "retail staff"
    if "cashier" in raw or "teller" in raw:
        return "cashier"
    if "operations" in raw and ("manager" in raw or "head" in raw):
        return "operations manager"
    if "operations" in raw or "operation" in raw:
        return "operations staff"
    if "general manager" in raw or "managing director" in raw:
        return "general manager"

    return None


# ============================================================
# 5.  Load & merge both datasets
# ============================================================
print("=" * 60)
print("SkillMap AI - Training Pipeline v3.0 (Dual-Dataset + Expanded Taxonomy)")
print("=" * 60)

print("\n[1/9] Loading datasets...")

# --- CSV1 ---
df1 = pd.read_csv(DATASET1_PATH)
rows_csv1 = len(df1)
print(f"  CSV1 rows: {rows_csv1}")

job_col1 = [c for c in df1.columns if "first" in c.lower() or "job" in c.lower()]
if job_col1:
    df1.rename(columns={job_col1[0]: "Pekerjaan_Pertama"}, inplace=True)
elif len(df1.columns) > 6:
    df1.rename(columns={df1.columns[6]: "Pekerjaan_Pertama"}, inplace=True)

df1 = df1[["Skill", "Pekerjaan_Pertama"]].copy()
df1.columns = ["Skill", "Job_raw"]
df1["source"] = "csv1"

# --- CSV2 ---
# CSV2 kolom Job ada di 'Status_Bekerja' (bukan Pekerjaan_Pertama)
try:
    df2_raw = pd.read_csv(DATASET2_PATH, on_bad_lines="skip", encoding="utf-8")
    rows_csv2_total = len(df2_raw)
    # Filter Status_Bekerja yang valid (bukan invalid values)
    invalid_status = {"Not Applicable", "Tidak", "Ya", "Student (Unemployed)", None, "nan"}
    df2_valid = df2_raw[~df2_raw["Status_Bekerja"].isin(invalid_status)].copy()
    df2_valid = df2_valid[df2_valid["Status_Bekerja"].notna()]
    df2_valid = df2_valid[["Skill", "Status_Bekerja"]].copy()
    df2_valid.columns = ["Skill", "Job_raw"]
    df2_valid["source"] = "csv2"
    rows_csv2_valid = len(df2_valid)
    print(f"  CSV2 rows total : {rows_csv2_total}")
    print(f"  CSV2 valid rows : {rows_csv2_valid} (using Status_Bekerja as job column)")
    use_csv2 = True
except Exception as e:
    print(f"  CSV2 load failed: {e} — using CSV1 only")
    df2_valid = pd.DataFrame(columns=["Skill", "Job_raw", "source"])
    rows_csv2_valid = 0
    rows_csv2_total = 0
    use_csv2 = False

# --- Merge ---
df = pd.concat([df1, df2_valid], ignore_index=True)
rows_combined = len(df)
print(f"\n  Combined rows   : {rows_combined}")


# ============================================================
# 6.  Preprocess skills & map jobs
# ============================================================
print("\n[2/9] Preprocessing...")

df["Skill_clean"] = df["Skill"].apply(split_skills)
df["Job_clean"] = df["Job_raw"].apply(lambda x: clean_job(str(x)))

# Filter baris skill kosong
df = df[df["Skill_clean"].apply(lambda x: len(x) > 0)]
after_skill_filter = len(df)

# Map ke career_category
df["Career_Category"] = df["Job_clean"].apply(normalize_job_to_category)
matched = df["Career_Category"].notna().sum()
dropped = df["Career_Category"].isna().sum()

# Drop yang tidak match
df = df[df["Career_Category"].notna()].copy()

print(f"\n  ======= Preprocessing Report =======")
print(f"  CSV1 rows (raw)           : {rows_csv1}")
print(f"  CSV2 rows (total)         : {rows_csv2_total}")
print(f"  CSV2 valid rows used      : {rows_csv2_valid}")
print(f"  Combined rows             : {rows_combined}")
print(f"  After skill filter        : {after_skill_filter}")
print(f"  Matched to career_category: {matched}")
print(f"  Dropped (no match)        : {dropped}")

print(f"\n  ======= Distribution per Career Category =======")
cat_counts = df["Career_Category"].value_counts()
for cat, cnt in cat_counts.items():
    flag = " [!] SEDIKIT" if cnt < 5 else ""
    print(f"  {cat:<35} : {cnt:>4} samples{flag}")

print(f"\n  Total categories in data  : {df['Career_Category'].nunique()}")
print(f"  Total rows used           : {len(df)}")

all_skills = set()
for skills in df["Skill_clean"]:
    all_skills.update(skills)
print(f"  Unique skills             : {len(all_skills)}")


# ============================================================
# 7.  Build job_skill_map dari data + rule-based fallback
# ============================================================
print("\n[3/9] Building job_skill_map...")

job_skill_counter = {}
for _, row in df.iterrows():
    cat = row["Career_Category"]
    skills = row["Skill_clean"]
    if cat not in job_skill_counter:
        job_skill_counter[cat] = Counter()
    job_skill_counter[cat].update(skills)

# Dari data
job_skill_map = {}
for cat, counter in job_skill_counter.items():
    top = [s for s, _ in counter.most_common(10)]
    if top:
        job_skill_map[cat] = top

# Tambah rule-based untuk categories yang tidak/kurang ada di data
for cat, skills in RULE_BASED_SKILL_MAP.items():
    if cat not in job_skill_map:
        job_skill_map[cat] = skills
    elif len(job_skill_map[cat]) < 5:
        # Merge: data skills + rule-based (deduplicated)
        merged = list(dict.fromkeys(job_skill_map[cat] + skills))[:10]
        job_skill_map[cat] = merged

# Pastikan ALL_CAREER_CATEGORIES ada di job_skill_map
for cat in ALL_CAREER_CATEGORIES:
    if cat not in job_skill_map:
        if cat in RULE_BASED_SKILL_MAP:
            job_skill_map[cat] = RULE_BASED_SKILL_MAP[cat]
        else:
            job_skill_map[cat] = ["communication skills", "problem solving skills"]

print(f"  job_skill_map entries: {len(job_skill_map)}")
print(f"  ALL_CAREER_CATEGORIES: {len(ALL_CAREER_CATEGORIES)}")


# ============================================================
# 8.  Encode data — pisah career_categories vs model_classes
# ============================================================
print("\n[4/9] Encoding & preparing training data...")

# model_classes = categories dengan >= MIN_SAMPLES di dataset
MIN_SAMPLES_FOR_MODEL = 5
cat_counts_all = df["Career_Category"].value_counts()
model_class_list = sorted(cat_counts_all[cat_counts_all >= MIN_SAMPLES_FOR_MODEL].index.tolist())
df_train = df[df["Career_Category"].isin(model_class_list)].copy()

excluded_from_model = df["Career_Category"].nunique() - len(model_class_list)
print(f"  MIN_SAMPLES_FOR_MODEL     : {MIN_SAMPLES_FOR_MODEL}")
print(f"  model_classes             : {len(model_class_list)}")
print(f"  Excluded (low sample)     : {excluded_from_model}")
print(f"  Training rows             : {len(df_train)}")
print(f"  career_categories (/jobs) : {len(ALL_CAREER_CATEGORIES)}")
print(f"\n  Model classes: {model_class_list}")

# MultiLabelBinarizer (fit ALL data untuk full skill coverage)
mlb = MultiLabelBinarizer()
mlb.fit(df["Skill_clean"])
X = mlb.transform(df_train["Skill_clean"])

# LabelEncoder untuk model_classes
le = LabelEncoder()
y_encoded = le.fit_transform(df_train["Career_Category"])

num_skills = X.shape[1]
num_classes = len(le.classes_)

print(f"\n  Input features (skills)   : {num_skills}")
print(f"  Output classes            : {num_classes}")
print(f"  Training samples          : {X.shape[0]}")

y_onehot = tf.keras.utils.to_categorical(y_encoded, num_classes=num_classes)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42, stratify=y_encoded
)
y_test_labels = le.inverse_transform(np.argmax(y_test, axis=1))
print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")


# ============================================================
# 9.  Custom Callback
# ============================================================
class SkillMapTrainingMonitor(tf.keras.callbacks.Callback):
    """
    Custom Callback: track best val_accuracy & print training summary.
    """
    def __init__(self):
        super().__init__()
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        self.history_log = {
            "accuracy": [], "val_accuracy": [],
            "loss": [], "val_loss": [],
        }

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k in self.history_log:
            self.history_log[k].append(logs.get(k, 0))
        val_acc = logs.get("val_accuracy", 0)
        if val_acc > self.best_val_accuracy:
            self.best_val_accuracy = val_acc
            self.best_epoch = epoch + 1
            print(f"  >> [Monitor] New best val_accuracy: {val_acc:.4f} at epoch {epoch + 1}")

    def on_train_end(self, logs=None):
        print("\n" + "=" * 60)
        print("Training Summary (SkillMapTrainingMonitor)")
        print("=" * 60)
        n = len(self.history_log["accuracy"])
        print(f"  Total epochs      : {n}")
        print(f"  Best val_accuracy : {self.best_val_accuracy:.4f}")
        print(f"  Best epoch        : {self.best_epoch}")
        if n:
            print(f"  Final train acc   : {self.history_log['accuracy'][-1]:.4f}")
            print(f"  Final val acc     : {self.history_log['val_accuracy'][-1]:.4f}")
            print(f"  Final train loss  : {self.history_log['loss'][-1]:.4f}")
            print(f"  Final val loss    : {self.history_log['val_loss'][-1]:.4f}")
        print("=" * 60)


# ============================================================
# 10.  Build & Train model (TF Functional API)
# ============================================================
print("\n[5/9] Building model (TensorFlow Functional API)...")

inputs = tf.keras.Input(shape=(num_skills,), name="skill_input")
x = tf.keras.layers.Dense(512, activation="relu", name="hidden_1")(inputs)
x = tf.keras.layers.BatchNormalization(name="bn_1")(x)
x = tf.keras.layers.Dropout(0.4, name="dropout_1")(x)
x = tf.keras.layers.Dense(256, activation="relu", name="hidden_2")(x)
x = tf.keras.layers.BatchNormalization(name="bn_2")(x)
x = tf.keras.layers.Dropout(0.3, name="dropout_2")(x)
x = tf.keras.layers.Dense(128, activation="relu", name="hidden_3")(x)
x = tf.keras.layers.Dropout(0.2, name="dropout_3")(x)
x = tf.keras.layers.Dense(64, activation="relu", name="hidden_4")(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="career_output")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="SkillMap_Career_Model_v3")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

print("\n[6/9] Training model...")
monitor_cb = SkillMapTrainingMonitor()
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=12, restore_best_weights=True
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=70,
    batch_size=32,
    callbacks=[monitor_cb, early_stop, reduce_lr],
    verbose=1,
)


# ============================================================
# 11.  Evaluation
# ============================================================
print("\n[7/9] Evaluation Report...")

train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

y_pred_proba = model.predict(X_test, verbose=0)
y_pred_labels = le.inverse_transform(np.argmax(y_pred_proba, axis=1))

print(f"\n  === Model Evaluation ===")
print(f"  Train Accuracy : {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"  Test  Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"  Test  Loss     : {test_loss:.4f}")

print(f"\n  === Classification Report ===")
print(classification_report(
    y_test_labels, y_pred_labels,
    labels=le.classes_, zero_division=0,
))

pred_dist = Counter(y_pred_labels)
print(f"\n  === Top-10 Predicted (Test Set) ===")
for cat, cnt in pred_dist.most_common(10):
    print(f"  {cat:<35} : {cnt}")


# ============================================================
# 12.  Save all artifacts
# ============================================================
print("\n[8/9] Saving artifacts...")

model.save(ARTIFACT_DIR / "skillmap_model.keras")
print(f"  [OK] skillmap_model.keras")

with open(ARTIFACT_DIR / "mlb.pkl", "wb") as f:
    pickle.dump(mlb, f)
print(f"  [OK] mlb.pkl")

with open(ARTIFACT_DIR / "label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print(f"  [OK] label_encoder.pkl ({num_classes} model classes)")

with open(ARTIFACT_DIR / "job_skill_map.json", "w", encoding="utf-8") as f:
    json.dump(job_skill_map, f, indent=2, ensure_ascii=False)
print(f"  [OK] job_skill_map.json ({len(job_skill_map)} entries)")

known_skills_list = sorted(list(all_skills))
with open(ARTIFACT_DIR / "known_skills.json", "w", encoding="utf-8") as f:
    json.dump(known_skills_list, f, indent=2, ensure_ascii=False)
print(f"  [OK] known_skills.json ({len(known_skills_list)} skills)")

# career_categories.json = ALL categories untuk /jobs
with open(ARTIFACT_DIR / "career_categories.json", "w", encoding="utf-8") as f:
    json.dump(ALL_CAREER_CATEGORIES, f, indent=2, ensure_ascii=False)
print(f"  [OK] career_categories.json ({len(ALL_CAREER_CATEGORIES)} categories)")

# model_classes.json = subset yang dilatih model
with open(ARTIFACT_DIR / "model_classes.json", "w", encoding="utf-8") as f:
    json.dump(sorted(list(le.classes_)), f, indent=2, ensure_ascii=False)
print(f"  [OK] model_classes.json ({num_classes} classes)")

MANUAL_COURSE_LINKS = {
    "python": "https://www.freecodecamp.org/learn/scientific-computing-with-python/",
    "sql": "https://www.w3schools.com/sql/",
    "html": "https://www.w3schools.com/html/",
    "css": "https://www.w3schools.com/css/",
    "java": "https://www.w3schools.com/java/",
    "javascript": "https://www.w3schools.com/js/",
    "excel": "https://support.microsoft.com/en-us/excel",
    "tableau": "https://www.tableau.com/learn/training",
    "communication skills": "https://www.coursera.org/courses?query=communication+skills",
    "problem solving skills": "https://www.coursera.org/courses?query=problem+solving",
    "critical thinking": "https://www.coursera.org/courses?query=critical+thinking",
    "analytical skills": "https://www.coursera.org/courses?query=analytical+skills",
    "data visualization": "https://www.kaggle.com/learn/data-visualization",
    "machine learning skills": "https://www.coursera.org/learn/machine-learning",
    "artificial intelligence": "https://www.coursera.org/learn/ai-for-everyone",
    "programming": "https://www.freecodecamp.org/learn/",
    "c++": "https://www.w3schools.com/cpp/",
    "active listening": "https://www.coursera.org/courses?query=active+listening",
    "business knowledge": "https://www.coursera.org/courses?query=business+fundamentals",
    "leadership": "https://www.coursera.org/courses?query=leadership",
    "negotiation skills": "https://www.coursera.org/courses?query=negotiation",
    "product knowledge": "https://www.coursera.org/courses?query=product+management",
    "finance related skills": "https://www.coursera.org/courses?query=finance",
    "accounting skills": "https://www.coursera.org/courses?query=accounting",
    "sales": "https://www.coursera.org/courses?query=sales",
    "people management": "https://www.coursera.org/courses?query=people+management",
    "r": "https://www.coursera.org/courses?query=r+programming",
    "linux": "https://www.coursera.org/courses?query=linux",
    "cloud computing": "https://www.coursera.org/courses?query=cloud+computing",
    "aws": "https://www.coursera.org/courses?query=aws",
    "php": "https://www.w3schools.com/php/",
    "designing skills": "https://www.coursera.org/courses?query=design+skills",
    "writing skills": "https://www.coursera.org/courses?query=writing+skills",
    "presentation skills": "https://www.coursera.org/courses?query=presentation+skills",
    "risk management skills": "https://www.coursera.org/courses?query=risk+management",
    "editing": "https://www.coursera.org/courses?query=editing",
    "interpersonal skills": "https://www.coursera.org/courses?query=interpersonal+skills",
    "matlab": "https://www.coursera.org/courses?query=matlab",
    "hr": "https://www.coursera.org/courses?query=human+resource+management",
    "social media marketing": "https://www.coursera.org/courses?query=social+media+marketing",
    "project management": "https://www.coursera.org/courses?query=project+management",
    "graphic design": "https://www.coursera.org/courses?query=graphic+design",
    "content creation": "https://www.coursera.org/courses?query=content+creation",
    "digital marketing": "https://www.coursera.org/courses?query=digital+marketing",
    "seo": "https://www.coursera.org/courses?query=seo",
    "figma": "https://www.coursera.org/courses?query=figma",
    "ui ux": "https://www.coursera.org/courses?query=ui+ux+design",
    "docker": "https://www.coursera.org/courses?query=docker",
    "kubernetes": "https://www.coursera.org/courses?query=kubernetes",
    "git": "https://www.coursera.org/courses?query=git",
    "agile": "https://www.coursera.org/courses?query=agile",
    "networking": "https://www.coursera.org/courses?query=networking",
    "cybersecurity": "https://www.coursera.org/courses?query=cybersecurity",
    "financial analysis": "https://www.coursera.org/courses?query=financial+analysis",
    "taxation": "https://www.coursera.org/courses?query=taxation",
    "recruitment": "https://www.coursera.org/courses?query=recruitment",
    "supply chain": "https://www.coursera.org/courses?query=supply+chain+management",
    "procurement": "https://www.coursera.org/courses?query=procurement",
    "teaching": "https://www.coursera.org/courses?query=teaching",
    "medical knowledge": "https://www.coursera.org/courses?query=medical",
    "legal knowledge": "https://www.coursera.org/courses?query=legal",
    "customer service": "https://www.coursera.org/courses?query=customer+service",
    "autocad": "https://www.coursera.org/courses?query=autocad",
    "manufacturing": "https://www.coursera.org/courses?query=manufacturing",
    "quality management": "https://www.coursera.org/courses?query=quality+management",
    "photoshop": "https://www.coursera.org/courses?query=photoshop",
    "video editing": "https://www.coursera.org/courses?query=video+editing",
    "event management": "https://www.coursera.org/courses?query=event+management",
    "logical thinking": "https://www.coursera.org/courses?query=logical+thinking",
    "cash handling": "https://www.coursera.org/courses?query=retail+management",
    "inventory management": "https://www.coursera.org/courses?query=inventory+management",
    "culinary skills": "https://www.coursera.org/courses?query=culinary+skills",
    "nursing": "https://www.coursera.org/courses?query=nursing",
    "pharmacy": "https://www.coursera.org/courses?query=pharmacy",
    "research skills": "https://www.coursera.org/courses?query=research+methods",
    "architecture": "https://www.coursera.org/courses?query=architecture",
}

with open(ARTIFACT_DIR / "course_links.json", "w", encoding="utf-8") as f:
    json.dump(MANUAL_COURSE_LINKS, f, indent=2, ensure_ascii=False)
print(f"  [OK] course_links.json")


# ============================================================
# Done
# ============================================================
print("\n[9/9] Summary")
print("=" * 60)
print("Training pipeline complete! (v3.0)")
print(f"  CSV1 rows (raw)           : {rows_csv1}")
print(f"  CSV2 valid rows used      : {rows_csv2_valid}")
print(f"  Combined rows             : {rows_combined}")
print(f"  Training rows             : {len(df_train)}")
print(f"  Rows dropped (no match)   : {dropped}")
print(f"  career_categories (/jobs) : {len(ALL_CAREER_CATEGORIES)}")
print(f"  model_classes (trained)   : {num_classes}")
print(f"  Skill features            : {num_skills}")
print(f"  Train accuracy            : {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"  Test  accuracy            : {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"  Artifacts                 : {ARTIFACT_DIR}")
print("=" * 60)

print(f"\nAll {len(ALL_CAREER_CATEGORIES)} career categories:")
for cat in ALL_CAREER_CATEGORIES:
    in_model = "[model]" if cat in model_class_list else "[rule-based]"
    print(f"  {in_model:13} {cat}")
