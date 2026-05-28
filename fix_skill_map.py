"""
Regenerate job_skill_map.json dengan CURATED_SKILL_MAP yang override data noisy.
Jalankan: python fix_skill_map.py
"""
import json
from pathlib import Path

ARTIFACT_DIR = Path("artifacts")

# ============================================================
# Curated skill map — ini yang benar-benar override dataset
# Skills dipilih berdasarkan job requirements yang realistis
# ============================================================
CURATED_SKILL_MAP = {
    # ── IT / Software ──────────────────────────────────────────
    "software engineer": [
        "python", "java", "c++", "javascript", "git",
        "sql", "system designing", "problem solving skills",
        "communication skills", "analytical skills",
    ],
    "web developer": [
        "html", "css", "javascript", "python", "git",
        "sql", "php", "react", "communication skills", "problem solving skills",
    ],
    "mobile developer": [
        "java", "kotlin", "swift", "flutter", "react native",
        "python", "git", "problem solving skills", "communication skills", "analytical skills",
    ],
    "data analyst": [
        "python", "sql", "excel", "data visualization", "analytical skills",
        "communication skills", "problem solving skills", "tableau", "r programming",
        "statistical analysis",
    ],
    "data scientist": [
        "python", "machine learning", "sql", "r programming", "data visualization",
        "analytical skills", "statistical analysis", "deep learning",
        "communication skills", "problem solving skills",
    ],
    "data engineer": [
        "python", "sql", "spark", "hadoop", "cloud computing",
        "database design", "etl", "analytical skills",
        "communication skills", "problem solving skills",
    ],
    "machine learning engineer": [
        "machine learning", "python", "deep learning", "tensorflow",
        "pytorch", "sql", "analytical skills", "data visualization",
        "communication skills", "problem solving skills",
    ],
    "ai engineer": [
        "machine learning", "python", "deep learning", "nlp",
        "computer vision", "tensorflow", "pytorch",
        "analytical skills", "communication skills", "problem solving skills",
    ],
    "devops engineer": [
        "linux", "docker", "kubernetes", "git", "cloud computing",
        "ci/cd", "python", "problem solving skills",
        "communication skills", "analytical skills",
    ],
    "cloud engineer": [
        "cloud computing", "aws", "linux", "docker", "kubernetes",
        "python", "networking", "problem solving skills",
        "communication skills", "analytical skills",
    ],
    "it support": [
        "communication skills", "problem solving skills", "networking",
        "windows os", "active listening", "troubleshooting",
        "customer service", "analytical skills", "linux", "microsoft office",
    ],
    "network engineer": [
        "networking", "linux", "cloud computing", "communication skills",
        "problem solving skills", "analytical skills",
        "cisco", "firewall", "tcp/ip", "routing",
    ],
    "cyber security analyst": [
        "networking", "linux", "python", "information security",
        "analytical skills", "problem solving skills",
        "communication skills", "ethical hacking", "risk management skills", "compliance",
    ],
    "database administrator": [
        "sql", "python", "database design", "analytical skills",
        "problem solving skills", "linux", "backup and recovery",
        "performance tuning", "communication skills", "oracle",
    ],
    "ui ux designer": [
        "figma", "designing skills", "creative thinking", "communication skills",
        "analytical skills", "user research", "prototyping",
        "adobe xd", "problem solving skills", "presentation skills",
    ],
    "embedded engineer": [
        "c++", "python", "electronics", "microcontroller",
        "linux", "problem solving skills", "analytical skills",
        "communication skills", "electrical engineering", "circuit design",
    ],
    "quality assurance engineer": [
        "testing", "analytical skills", "communication skills",
        "problem solving skills", "attention to detail",
        "python", "selenium", "documentation", "sql", "agile",
    ],
    "game developer": [
        "c++", "python", "game development", "unity",
        "creative thinking", "problem solving skills",
        "java", "communication skills", "3d modeling", "analytical skills",
    ],
    "solutions architect": [
        "cloud computing", "system designing", "aws", "communication skills",
        "analytical skills", "problem solving skills",
        "python", "presentation skills", "networking", "leadership",
    ],
    "erp consultant": [
        "sap", "sql", "communication skills", "analytical skills",
        "problem solving skills", "business knowledge",
        "presentation skills", "project management", "excel", "documentation",
    ],
    "technical writer": [
        "writing skills", "communication skills", "documentation",
        "analytical skills", "research skills",
        "microsoft office", "attention to detail", "editing", "html", "project management",
    ],
    # ── Business / Management ──────────────────────────────────
    "product manager": [
        "analytical skills", "communication skills", "leadership",
        "problem solving skills", "presentation skills",
        "project management", "agile", "product knowledge", "business knowledge", "stakeholder management",
    ],
    "project manager": [
        "project management", "leadership", "communication skills",
        "analytical skills", "problem solving skills",
        "risk management skills", "agile", "presentation skills", "stakeholder management", "ms project",
    ],
    "business analyst": [
        "analytical skills", "communication skills", "problem solving skills",
        "sql", "excel", "presentation skills",
        "business knowledge", "documentation", "stakeholder management", "agile",
    ],
    "operations manager": [
        "leadership", "analytical skills", "communication skills",
        "problem solving skills", "project management",
        "excel", "business knowledge", "people management", "process improvement", "reporting",
    ],
    "business development": [
        "sales", "communication skills", "negotiation skills",
        "analytical skills", "presentation skills",
        "networking", "market research", "crm", "leadership", "problem solving skills",
    ],
    "management trainee": [
        "communication skills", "analytical skills", "leadership",
        "problem solving skills", "teamwork",
        "excel", "presentation skills", "business knowledge", "adaptability", "interpersonal skills",
    ],
    "entrepreneur": [
        "leadership", "communication skills", "business knowledge",
        "problem solving skills", "analytical skills",
        "financial management", "marketing", "networking", "creativity", "resilience",
    ],
    "general manager": [
        "leadership", "strategic planning", "communication skills",
        "financial management", "people management",
        "analytical skills", "problem solving skills", "business knowledge", "negotiation skills", "reporting",
    ],
    # ── Marketing / Creative ───────────────────────────────────
    "digital marketing": [
        "digital marketing", "seo", "google analytics",
        "communication skills", "analytical skills",
        "content creation", "social media marketing", "excel", "copywriting", "presentation skills",
    ],
    "marketing staff": [
        "communication skills", "analytical skills", "excel",
        "presentation skills", "marketing",
        "creativity", "problem solving skills", "social media", "market research", "reporting",
    ],
    "social media specialist": [
        "social media marketing", "communication skills", "creativity",
        "content creation", "analytical skills",
        "copywriting", "photoshop", "video editing", "presentation skills", "seo",
    ],
    "content creator": [
        "writing skills", "creativity", "communication skills",
        "social media", "content creation",
        "photoshop", "video editing", "research skills", "seo", "analytical skills",
    ],
    "copywriter": [
        "writing skills", "creativity", "communication skills",
        "seo", "editing",
        "research skills", "analytical skills", "content strategy", "storytelling", "attention to detail",
    ],
    "graphic designer": [
        "photoshop", "illustrator", "designing skills", "creativity",
        "communication skills", "figma",
        "adobe xd", "typography", "attention to detail", "presentation skills",
    ],
    "video editor": [
        "video editing", "premiere pro", "after effects",
        "creativity", "communication skills",
        "storytelling", "attention to detail", "color grading", "motion graphics", "analytical skills",
    ],
    "public relations": [
        "communication skills", "writing skills", "media relations",
        "social media", "crisis management",
        "presentation skills", "networking", "analytical skills", "creativity", "research skills",
    ],
    "journalist": [
        "writing skills", "communication skills", "research skills",
        "analytical skills", "interviewing",
        "editing", "storytelling", "attention to detail", "social media", "critical thinking",
    ],
    "writer": [
        "writing skills", "creativity", "communication skills",
        "research skills", "editing",
        "storytelling", "attention to detail", "analytical skills", "seo", "time management",
    ],
    "translator": [
        "language skills", "communication skills", "writing skills",
        "attention to detail", "research skills",
        "cultural knowledge", "time management", "analytical skills", "editing", "microsoft office",
    ],
    "event planner": [
        "communication skills", "project management", "organizational skills",
        "creativity", "vendor management",
        "problem solving skills", "negotiation skills", "budgeting", "leadership", "attention to detail",
    ],
    "photographer": [
        "photography", "photo editing", "creative thinking",
        "communication skills", "lightroom",
        "photoshop", "composition", "attention to detail", "storytelling", "time management",
    ],
    # ── Sales / Customer ───────────────────────────────────────
    "sales executive": [
        "sales", "communication skills", "negotiation skills",
        "product knowledge", "customer service",
        "crm", "problem solving skills", "interpersonal skills", "analytical skills", "presentation skills",
    ],
    "sales manager": [
        "leadership", "sales", "communication skills",
        "negotiation skills", "analytical skills",
        "team management", "crm", "problem solving skills", "presentation skills", "forecasting",
    ],
    "account executive": [
        "sales", "communication skills", "negotiation skills",
        "crm", "relationship management",
        "product knowledge", "problem solving skills", "presentation skills", "analytical skills", "customer service",
    ],
    "customer service": [
        "communication skills", "active listening", "problem solving skills",
        "customer service", "patience",
        "crm", "product knowledge", "interpersonal skills", "empathy", "microsoft office",
    ],
    "call center agent": [
        "communication skills", "active listening", "customer service",
        "problem solving skills", "crm",
        "patience", "typing", "product knowledge", "interpersonal skills", "stress management",
    ],
    "retail staff": [
        "customer service", "communication skills", "sales",
        "product knowledge", "inventory management",
        "cash handling", "teamwork", "problem solving skills", "attention to detail", "adaptability",
    ],
    "store manager": [
        "leadership", "sales", "inventory management",
        "customer service", "team management",
        "problem solving skills", "analytical skills", "communication skills", "budgeting", "reporting",
    ],
    "cashier": [
        "communication skills", "cash handling", "customer service",
        "attention to detail", "numerical skills",
        "pos system", "teamwork", "patience", "honesty", "time management",
    ],
    # ── Finance / Accounting ───────────────────────────────────
    "accountant": [
        "accounting skills", "excel", "analytical skills",
        "finance related skills", "attention to detail",
        "problem solving skills", "communication skills", "tax knowledge", "reporting", "audit",
    ],
    "finance staff": [
        "finance related skills", "excel", "accounting skills",
        "analytical skills", "attention to detail",
        "communication skills", "problem solving skills", "reporting", "budgeting", "sql",
    ],
    "financial analyst": [
        "financial analysis", "excel", "analytical skills",
        "sql", "data visualization",
        "communication skills", "problem solving skills", "forecasting", "reporting", "python",
    ],
    "tax staff": [
        "accounting skills", "taxation", "excel",
        "analytical skills", "attention to detail",
        "communication skills", "problem solving skills", "compliance", "reporting", "finance related skills",
    ],
    "auditor": [
        "accounting skills", "analytical skills", "excel",
        "risk management skills", "attention to detail",
        "communication skills", "problem solving skills", "compliance", "reporting", "documentation",
    ],
    "banking staff": [
        "finance related skills", "communication skills", "excel",
        "customer service", "analytical skills",
        "compliance", "problem solving skills", "attention to detail", "interpersonal skills", "reporting",
    ],
    "insurance agent": [
        "sales", "communication skills", "finance related skills",
        "negotiation skills", "customer service",
        "product knowledge", "interpersonal skills", "problem solving skills", "presentation skills", "analytical skills",
    ],
    "investment analyst": [
        "financial analysis", "excel", "analytical skills",
        "sql", "market research",
        "communication skills", "problem solving skills", "valuation", "presentation skills", "python",
    ],
    # ── HR / Admin ─────────────────────────────────────────────
    "human resources": [
        "communication skills", "interpersonal skills", "people management",
        "analytical skills", "hr",
        "recruitment", "microsoft office", "excel", "problem solving skills", "leadership",
    ],
    "recruiter": [
        "communication skills", "interpersonal skills", "hr",
        "analytical skills", "sourcing",
        "interviewing", "linkedin", "problem solving skills", "networking", "presentation skills",
    ],
    "training staff": [
        "communication skills", "presentation skills", "curriculum design",
        "facilitation", "analytical skills",
        "instructional design", "microsoft office", "problem solving skills", "interpersonal skills", "leadership",
    ],
    "admin officer": [
        "microsoft office", "communication skills", "organizational skills",
        "data entry", "problem solving skills",
        "excel", "attention to detail", "time management", "scheduling", "filing",
    ],
    "secretary": [
        "microsoft office", "communication skills", "organizational skills",
        "scheduling", "time management",
        "attention to detail", "typing", "filing", "interpersonal skills", "confidentiality",
    ],
    "office staff": [
        "microsoft office", "communication skills", "data entry",
        "organizational skills", "attention to detail",
        "filing", "time management", "teamwork", "typing", "scheduling",
    ],
    "receptionist": [
        "communication skills", "customer service", "microsoft office",
        "scheduling", "phone etiquette",
        "organizational skills", "interpersonal skills", "multitasking", "attention to detail", "professional appearance",
    ],
    # ── Engineering Non-IT ─────────────────────────────────────
    "civil engineer": [
        "autocad", "analytical skills", "problem solving skills",
        "communication skills", "project management",
        "structural analysis", "surveying", "construction management", "attention to detail", "microsoft office",
    ],
    "mechanical engineer": [
        "autocad", "analytical skills", "problem solving skills",
        "communication skills", "matlab",
        "solidworks", "manufacturing", "project management", "attention to detail", "technical drawing",
    ],
    "electrical engineer": [
        "electrical engineering", "autocad", "analytical skills",
        "problem solving skills", "communication skills",
        "circuit design", "plc programming", "project management", "attention to detail", "matlab",
    ],
    "industrial engineer": [
        "analytical skills", "problem solving skills", "autocad",
        "communication skills", "lean manufacturing",
        "process improvement", "project management", "statistical analysis", "excel", "attention to detail",
    ],
    "architect": [
        "autocad", "designing skills", "creative thinking",
        "communication skills", "project management",
        "revit", "sketchup", "presentation skills", "attention to detail", "3d modeling",
    ],
    "drafter": [
        "autocad", "technical drawing", "analytical skills",
        "attention to detail", "communication skills",
        "revit", "solidworks", "cad software", "problem solving skills", "microsoft office",
    ],
    "surveyor": [
        "surveying", "analytical skills", "attention to detail",
        "communication skills", "problem solving skills",
        "autocad", "gis", "microsoft office", "technical drawing", "measurement tools",
    ],
    "technician": [
        "technical skills", "problem solving skills", "attention to detail",
        "communication skills", "maintenance",
        "troubleshooting", "analytical skills", "teamwork", "safety awareness", "documentation",
    ],
    "automotive technician": [
        "mechanical skills", "troubleshooting", "attention to detail",
        "communication skills", "problem solving skills",
        "diagnostic tools", "maintenance", "teamwork", "safety awareness", "technical skills",
    ],
    # ── Quality / Production ───────────────────────────────────
    "quality control": [
        "quality management", "analytical skills", "attention to detail",
        "problem solving skills", "communication skills",
        "documentation", "testing", "reporting", "iso standards", "teamwork",
    ],
    "quality assurance engineer": [
        "testing", "analytical skills", "attention to detail",
        "problem solving skills", "communication skills",
        "documentation", "python", "selenium", "sql", "agile",
    ],
    "production staff": [
        "production planning", "analytical skills", "problem solving skills",
        "communication skills", "teamwork",
        "lean manufacturing", "quality control", "attention to detail", "microsoft office", "reporting",
    ],
    "manufacturing staff": [
        "manufacturing", "analytical skills", "problem solving skills",
        "teamwork", "attention to detail",
        "quality control", "lean manufacturing", "safety awareness", "communication skills", "documentation",
    ],
    "safety officer": [
        "safety management", "risk management skills", "analytical skills",
        "communication skills", "problem solving skills",
        "regulatory compliance", "documentation", "training delivery", "attention to detail", "reporting",
    ],
    # ── Operations / Logistics ─────────────────────────────────
    "operations staff": [
        "communication skills", "analytical skills", "problem solving skills",
        "organizational skills", "microsoft office",
        "excel", "teamwork", "attention to detail", "reporting", "time management",
    ],
    "operations manager": [
        "leadership", "analytical skills", "communication skills",
        "problem solving skills", "process improvement",
        "excel", "project management", "people management", "reporting", "business knowledge",
    ],
    "logistics staff": [
        "logistics", "supply chain", "analytical skills",
        "communication skills", "problem solving skills",
        "microsoft office", "excel", "organizational skills", "attention to detail", "reporting",
    ],
    "warehouse staff": [
        "inventory management", "organizational skills", "attention to detail",
        "communication skills", "teamwork",
        "physical fitness", "safety awareness", "microsoft office", "forklift operation", "reporting",
    ],
    "supply chain staff": [
        "supply chain", "analytical skills", "communication skills",
        "excel", "problem solving skills",
        "negotiation skills", "vendor management", "inventory management", "reporting", "attention to detail",
    ],
    "procurement staff": [
        "procurement", "analytical skills", "negotiation skills",
        "communication skills", "vendor management",
        "excel", "problem solving skills", "attention to detail", "sourcing", "market research",
    ],
    "purchasing staff": [
        "procurement", "negotiation skills", "analytical skills",
        "communication skills", "excel",
        "vendor management", "attention to detail", "problem solving skills", "sourcing", "reporting",
    ],
    # ── Education ──────────────────────────────────────────────
    "teacher": [
        "communication skills", "teaching", "subject knowledge",
        "patience", "interpersonal skills",
        "curriculum development", "classroom management", "problem solving skills", "creativity", "adaptability",
    ],
    "lecturer": [
        "communication skills", "teaching", "research skills",
        "subject expertise", "academic writing",
        "curriculum development", "analytical skills", "presentation skills", "mentoring", "problem solving skills",
    ],
    "tutor": [
        "communication skills", "teaching", "subject knowledge",
        "patience", "interpersonal skills",
        "analytical skills", "problem solving skills", "adaptability", "mentoring", "microsoft office",
    ],
    "researcher": [
        "research skills", "analytical skills", "writing skills",
        "communication skills", "critical thinking",
        "data analysis", "python", "sql", "statistical analysis", "problem solving skills",
    ],
    # ── Healthcare ─────────────────────────────────────────────
    "doctor": [
        "medical knowledge", "diagnostic skills", "communication skills",
        "empathy", "clinical skills",
        "problem solving skills", "attention to detail", "patient care", "analytical skills", "decision making",
    ],
    "nurse": [
        "nursing", "patient care", "communication skills",
        "medical knowledge", "empathy",
        "attention to detail", "clinical skills", "teamwork", "problem solving skills", "time management",
    ],
    "pharmacist": [
        "pharmacy", "medical knowledge", "analytical skills",
        "communication skills", "attention to detail",
        "patient counseling", "compliance", "problem solving skills", "teamwork", "microsoft office",
    ],
    "laboratory analyst": [
        "analytical skills", "lab techniques", "data analysis",
        "attention to detail", "scientific writing",
        "communication skills", "problem solving skills", "research skills", "documentation", "quality control",
    ],
    "healthcare staff": [
        "medical knowledge", "communication skills", "patient care",
        "empathy", "teamwork",
        "attention to detail", "problem solving skills", "clinical skills", "time management", "microsoft office",
    ],
    # ── Legal ──────────────────────────────────────────────────
    "legal staff": [
        "legal knowledge", "analytical skills", "writing skills",
        "communication skills", "research skills",
        "attention to detail", "negotiation skills", "problem solving skills", "documentation", "compliance",
    ],
    "paralegal": [
        "legal research", "writing skills", "analytical skills",
        "communication skills", "organizational skills",
        "documentation", "attention to detail", "microsoft office", "problem solving skills", "compliance",
    ],
    "company secretary": [
        "legal knowledge", "compliance", "communication skills",
        "organizational skills", "documentation",
        "microsoft office", "attention to detail", "analytical skills", "problem solving skills", "confidentiality",
    ],
    # ── Hospitality / F&B ──────────────────────────────────────
    "hospitality staff": [
        "customer service", "communication skills", "teamwork",
        "problem solving skills", "attention to detail",
        "interpersonal skills", "adaptability", "time management", "multitasking", "professional appearance",
    ],
    "chef / cook": [
        "culinary skills", "creativity", "time management",
        "teamwork", "attention to detail",
        "food safety", "kitchen management", "problem solving skills", "physical fitness", "leadership",
    ],
    "barista": [
        "customer service", "communication skills", "coffee knowledge",
        "teamwork", "attention to detail",
        "time management", "adaptability", "product knowledge", "cash handling", "interpersonal skills",
    ],
    "tourism staff": [
        "communication skills", "customer service", "language skills",
        "tourism knowledge", "interpersonal skills",
        "problem solving skills", "adaptability", "teamwork", "local knowledge", "presentation skills",
    ],
}

# ============================================================
# Skill validator — harus konsisten dengan inference.py
# ============================================================
KNOWN_SHORT_SKILLS = {"c#", "js", "hr", "ml", "ai", "go"}
INVALID_VALUES = {"tidak", "ya", "nan", "na", "not applicable", ""}


def is_valid_skill(skill: str) -> bool:
    s = str(skill).strip().lower()
    if s in KNOWN_SHORT_SKILLS:
        return True
    if s in INVALID_VALUES:
        return False
    if len(s) < 3:
        return False
    if s.replace(".", "", 1).replace(",", "", 1).isdigit():
        return False
    alpha_count = sum(1 for c in s if c.isalpha())
    if alpha_count < 2:
        return False
    return True


# Load existing job_skill_map
with open(ARTIFACT_DIR / "job_skill_map.json", "r", encoding="utf-8") as f:
    existing = json.load(f)

print("=== Regenerating job_skill_map.json with curated skills ===")
print(f"Original entries: {len(existing)}")

# Override with curated data + validate every skill
updated = dict(existing)
for cat, skills in CURATED_SKILL_MAP.items():
    clean_skills = [s for s in skills if is_valid_skill(s)]
    if len(clean_skills) < len(skills):
        dropped = set(skills) - set(clean_skills)
        print(f"  [WARN] {cat}: dropped noise skills: {dropped}")
    updated[cat] = clean_skills
    if cat in existing:
        print(f"  [OVERRIDE] {cat}: {existing[cat][:3]} -> {clean_skills[:3]}")
    else:
        print(f"  [NEW]      {cat}")

# Also validate all existing rule-based entries (non-curated)
for cat, skills in list(updated.items()):
    if cat not in CURATED_SKILL_MAP:
        clean = [s for s in skills if is_valid_skill(s)]
        if len(clean) != len(skills):
            dropped = set(skills) - set(clean)
            print(f"  [CLEAN existing] {cat}: dropped {dropped}")
        updated[cat] = clean

print(f"\nFinal entries: {len(updated)}")

with open(ARTIFACT_DIR / "job_skill_map.json", "w", encoding="utf-8") as f:
    json.dump(updated, f, indent=2, ensure_ascii=False)

print("\nDone! job_skill_map.json updated.")

# Verify a few
print("\n=== Verify after update ===")
for cat in ["data analyst", "data scientist", "software engineer", "admin officer",
            "finance staff", "sales executive", "healthcare staff", "teacher",
            "graphic designer", "nurse"]:
    print(f"{cat}: {updated.get(cat, ['NOT FOUND'])}")

# Final noise check
all_skills_flat = [s for skills in updated.values() for s in skills]
noise = [s for s in all_skills_flat if not is_valid_skill(s)]
if noise:
    print(f"\n[ERROR] Masih ada noise skills: {set(noise)}")
else:
    print(f"\n[OK] Semua {len(all_skills_flat)} skill entries valid, tidak ada noise.")

