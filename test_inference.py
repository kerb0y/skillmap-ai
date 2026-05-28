import sys, os
sys.path.insert(0, '.')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from app.inference import extract_skills_from_cv_text, recommend_career_rule_based, get_skillmap_result

cv_logistics = (
    "Saya memiliki pengalaman sebagai staff gudang dan logistik. "
    "Saya terbiasa menerima barang, mengecek stok, melakukan packing, "
    "membuat laporan keluar masuk barang, mengatur pengiriman, dan memastikan data stok sesuai. "
    "Saya memiliki skill inventory management, warehouse management, logistics, data entry, "
    "microsoft excel, attention to detail, teamwork, time management, dan problem solving skills."
)

print("=== Skill Extraction ===")
skills = extract_skills_from_cv_text(cv_logistics)
print("Detected (%d): %s" % (len(skills), skills))

print()
print("=== Rule-based Recommendation ===")
rule_career, rule_score = recommend_career_rule_based(cv_logistics, skills)
print("rule_career: %s, score: %.2f%%" % (rule_career, rule_score))

print()
print("=== Full predict (target_job kosong) ===")
result = get_skillmap_result(cv_logistics, target_job='', quiz_score=80)
print("recommended_career: %s" % result["recommended_career"])
print("recommendation_source: %s" % result["recommendation_source"])
print("raw_model_prediction: %s" % result["raw_model_prediction"])
print("career_match_score: %.2f%%" % result["career_match_score"])
print("skill_dimiliki (%d): %s" % (len(result["skill_dimiliki"]), result["skill_dimiliki"]))
print("skill_gap (%d): %s" % (len(result["skill_gap"]), result["skill_gap"]))
print("summary: %s" % result["summary"])

# Test 8 CV non-IT lainnya
print()
print("=== Multi-CV test (target_job kosong) ===")
test_cases = [
    ("admin", "Background admin staff, terbiasa mengelola dokumen, microsoft office, data entry, scheduling, organizational skills, filing, attention to detail, communication skills."),
    ("finance", "Staff keuangan 3 tahun. accounting skills, excel, finance related skills, budgeting, reporting, analytical skills, attention to detail, communication skills."),
    ("sales", "Sales executive berpengalaman. sales, communication skills, negotiation skills, customer service, product knowledge, crm, presentation skills."),
    ("customer service", "CS di perusahaan retail. customer service, communication skills, active listening, problem solving skills, crm, product knowledge, empathy."),
    ("nurse", "Perawat di rumah sakit. nursing, patient care, communication skills, medical knowledge, empathy, clinical skills, attention to detail, teamwork."),
    ("graphic designer", "Desainer grafis freelance. photoshop, illustrator, figma, creativity, designing skills, communication skills, attention to detail."),
    ("teacher", "Guru SMA 5 tahun. teaching, communication skills, subject knowledge, curriculum development, classroom management, patience, creativity."),
    ("barista", "Barista kafe 2 tahun. customer service, coffee knowledge, communication skills, teamwork, attention to detail, cash handling, time management."),
]

for label, cv in test_cases:
    r = get_skillmap_result(cv, target_job='', quiz_score=75)
    print("  [%s] recommended=%s (src=%s) | raw_model=%s | match=%.0f%%" % (
        label,
        r["recommended_career"],
        r["recommendation_source"],
        r["raw_model_prediction"],
        r["career_match_score"]
    ))
