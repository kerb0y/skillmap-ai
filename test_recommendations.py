"""
Test career_recommendations field output untuk 3 domain berbeda.
"""
import sys, json
sys.path.insert(0, ".")
from app.inference import get_skillmap_result

def show(title, result):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  recommended_career : {result['recommended_career']}")
    recs = result.get("career_recommendations", [])
    print(f"  career_recommendations ({len(recs)} items):")
    for r in recs:
        print(f"    - {r['career']:<35} score={r['score']:>6.2f}  source={r['source']}")
    print(f"  career_match_score : {result['career_match_score']}")
    print(f"  skill_gap          : {result['skill_gap'][:4]}")
    print()

# --- CV Kehutanan ---
cv_kehutanan = (
    "Saya adalah lulusan Kehutanan IPB. Pengalaman saya meliputi survei hutan, "
    "inventarisasi vegetasi, pemetaan kawasan hutan, analisis biodiversitas, "
    "konservasi ekosistem. Saya terbiasa menggunakan GIS dan QGIS untuk "
    "pemetaan lapangan dan pengolahan data spasial. Terbiasa field survey, "
    "monitoring lapangan, ecosystem monitoring, climate change. Aktif di KLHK."
)
show("CV Kehutanan (no target_job)", get_skillmap_result(cv_kehutanan, "", 75))

# --- CV Logistik ---
cv_logistik = (
    "Saya memiliki pengalaman sebagai staff gudang dan logistik. Saya terbiasa "
    "menerima barang, mengecek stok, melakukan packing, laporan keluar masuk barang, "
    "mengatur pengiriman, memastikan data stok sesuai. Skill: inventory management, "
    "warehouse management, logistics, data entry, microsoft excel, attention to detail, "
    "supply chain, teamwork, time management, quality control, production planning."
)
show("CV Logistik/Gudang (no target_job)", get_skillmap_result(cv_logistik, "", 80))

# --- CV Hukum ---
cv_hukum = (
    "Saya seorang lulusan hukum dengan pengalaman sebagai legal staff. "
    "Saya terbiasa membuat kontrak, melakukan legal research, compliance review, "
    "dan perjanjian bisnis. Saya juga berpengalaman dalam legal knowledge, "
    "civil law, analytical skills, dan writing skills. Pernah magang sebagai paralegal."
)
show("CV Hukum (no target_job)", get_skillmap_result(cv_hukum, "", 70))

# --- CV target_job diisi (harus hanya 1 rekomendasi) ---
cv_data = (
    "Saya seorang data analyst. Skill: python, sql, tableau, excel, "
    "data visualization, analytical skills, statistical analysis."
)
show("CV Data Analyst (target_job=data analyst)", get_skillmap_result(cv_data, "data analyst", 85))
