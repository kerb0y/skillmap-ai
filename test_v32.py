"""
SkillMap AI v3.2 - Inference Test Script
Test 5 payload cases for validation.
"""
import sys
sys.path.insert(0, ".")
from app.inference import get_skillmap_result as predict


PASS = "[PASS]"
FAIL = "[FAIL]"


def check(label, result, expected_keywords):
    rec = result["recommended_career"].lower()
    ok = any(kw in rec for kw in expected_keywords)
    status = PASS if ok else FAIL
    print(f"  {status} recommended_career : {result['recommended_career']!r}  (expected one of: {expected_keywords})")
    print(f"       raw_model_prediction: {result['raw_model_prediction']!r}")
    print(f"       recommendation_source: {result['recommendation_source']!r}")
    print(f"       detected_skills    : {result['detected_skills_from_cv']}")
    print(f"       match_score={result['career_match_score']}  gap_score={result['gap_score']}")
    print(f"       skill_gap          : {result['skill_gap'][:5]}")
    print()
    return ok



print("=" * 65)
print("SkillMap AI v3.2 - Inference Validation")
print("=" * 65)
results = []

# ─── TEST 1: CV Kehutanan, no target_job ─────────────────────────
cv1 = (
    "Saya adalah lulusan Kehutanan IPB. Pengalaman saya meliputi survei hutan, "
    "inventarisasi vegetasi, pemetaan kawasan hutan, analisis biodiversitas, "
    "dan konservasi ekosistem. Saya terbiasa menggunakan GIS dan QGIS untuk "
    "pemetaan lapangan dan pengolahan data spasial. Aktif di KLHK."
)
print("=== TEST 1: CV Kehutanan (no target_job) ===")
r1 = predict(cv_text=cv1, target_job="", quiz_score=75)
results.append(check("T1", r1, ["forestry", "conservation", "environmental", "gis"]))

# ─── TEST 2: CV Kehutanan, target_job = forestry officer ─────────
print("=== TEST 2: CV Kehutanan (target_job=forestry officer) ===")
r2 = predict(cv_text=cv1, target_job="forestry officer", quiz_score=75)
results.append(check("T2", r2, ["forestry"]))

# ─── TEST 3: CV Logistik/Gudang ──────────────────────────────────
cv3 = (
    "Saya memiliki pengalaman sebagai staff gudang dan logistik. Saya terbiasa "
    "menerima barang, mengecek stok, melakukan packing, membuat laporan keluar masuk barang, "
    "mengatur pengiriman, dan memastikan data stok sesuai. Skill: inventory management, "
    "warehouse management, logistics, data entry, microsoft excel, attention to detail, "
    "teamwork, time management, problem solving skills."
)
print("=== TEST 3: CV Logistik (no target_job) ===")
r3 = predict(cv_text=cv3, target_job="", quiz_score=80)
results.append(check("T3", r3, ["warehouse", "logistics", "supply chain", "procurement"]))

# ─── TEST 4: CV Data Analyst ─────────────────────────────────────
cv4 = (
    "Saya seorang data analyst dengan pengalaman 3 tahun. Skill: python, sql, "
    "tableau, excel, data visualization, analytical skills, statistical analysis, "
    "machine learning, communication skills, problem solving skills."
)
print("=== TEST 4: CV Data Analyst (target_job=data analyst) ===")
r4 = predict(cv_text=cv4, target_job="data analyst", quiz_score=85)
results.append(check("T4", r4, ["data analyst"]))

# ─── TEST 5: CV Sustainability ────────────────────────────────────
cv5 = (
    "Saya bekerja di bidang keberlanjutan lingkungan. Saya memiliki pengalaman "
    "dalam analisis dampak lingkungan, sustainability reporting, climate change assessment, "
    "ESG compliance, dan community engagement. Terbiasa dengan regulasi AMDAL dan SDGs."
)
print("=== TEST 5: CV Sustainability (no target_job) ===")
r5 = predict(cv_text=cv5, target_job="", quiz_score=70)
results.append(check("T5", r5, ["sustainability", "environmental", "conservation", "gis"]))

# ─── Summary ─────────────────────────────────────────────────────
passed = sum(results)
total = len(results)
print("=" * 65)
print(f"Results: {passed}/{total} passed")
print("=" * 65)
