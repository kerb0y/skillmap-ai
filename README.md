# SkillMap AI

> Bagian AI untuk proyek capstone **SkillMap** – Aplikasi yang membantu user memahami potensi karier dan kebutuhan skill enhancement berdasarkan CV.

---

## Deskripsi

SkillMap AI adalah REST API berbasis FastAPI yang:

1. **Mengekstrak skill** dari teks CV user
2. **Memprediksi karier** yang cocok menggunakan model Deep Learning
3. **Menghitung career match score** berdasarkan skill, AI confidence, dan quiz score
4. **Mendeteksi skill gap** terhadap target karier
5. **Merekomendasikan learning path** (kursus) untuk setiap skill yang belum dimiliki

---

## Struktur Proyek

```
skillmap-ai/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application & endpoints
│   └── inference.py         # Logika ekstraksi, prediksi, skor, learning path
├── artifacts/
│   ├── skillmap_model.keras # Model TF hasil training
│   ├── mlb.pkl              # MultiLabelBinarizer (encoding skill)
│   ├── label_encoder.pkl    # LabelEncoder (encoding job/karier)
│   ├── job_skill_map.json   # Mapping job → required skills (430 jobs)
│   ├── course_links.json    # Mapping skill → link kursus
│   └── known_skills.json    # Daftar semua skill yang dikenali (557 skills)
├── train_model.py           # Script training model dari dataset
├── combined_career_recommender_clean.csv  # Dataset utama (22.147 baris)
├── requirements.txt
├── API_DOCS.md              # Dokumentasi endpoint lengkap
└── README.md
```

---

## Tech Stack

| Komponen | Library / Framework |
|---|---|
| API | FastAPI + Uvicorn |
| Model | TensorFlow (Functional API) |
| Encoding | scikit-learn (MLB, LabelEncoder) |
| Data | pandas, numpy |
| Dataset | `combined_career_recommender_clean.csv` |

---

## Setup & Instalasi

### 1. Clone & masuk ke folder

```bash
git clone <repo-url>
cd skillmap-ai
```

### 2. Buat virtual environment & install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Linux/Mac

pip install -r requirements.txt
```

### 3. (Opsional) Retrain model

Artifacts sudah tersedia di folder `artifacts/`. Jalankan ini hanya jika ingin rebuild model dari dataset:

```bash
python train_model.py
```

### 4. Jalankan API

```bash
uvicorn app.main:app --reload
```

API berjalan di: `http://127.0.0.1:8000`  
Swagger UI: `http://127.0.0.1:8000/docs`

---

## Endpoints

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| GET | `/` | Info API |
| GET | `/health` | Health check |
| GET | `/jobs` | Daftar target job yang tersedia |
| GET | `/skills` | Daftar skill yang dikenali sistem |
| POST | `/predict` | Analisis CV & rekomendasi karier |

### Contoh Request `POST /predict`

```json
{
  "cv_text": "Saya memiliki kemampuan Python, SQL, Excel, HTML, CSS, dan problem solving skills.",
  "target_job": "data analyst",
  "quiz_score": 80
}
```

> `target_job` bersifat **opsional**. Jika dikosongkan, sistem menggunakan prediksi AI sebagai rekomendasi karier.  
> `quiz_score` harus berada di rentang **0–100**.

### Contoh Response

```json
{
  "detected_skills_from_cv": ["css", "excel", "html", "problem solving skills", "python", "sql"],
  "target_job": "data analyst",
  "recommended_career": "data analyst",
  "predicted_career_ai": "computer software engineer",
  "career_match_score": 38.04,
  "gap_score": 61.96,
  "skill_match_score": 40.0,
  "model_career_score": 8.16,
  "quiz_score": 80.0,
  "skill_dimiliki": ["excel", "problem solving skills", "python", "sql"],
  "skill_gap": ["analytical skills", "communication skills", "data visualization", "machine learning skills", "programming", "team work"],
  "learning_path": [
    { "skill": "analytical skills", "course_link": "https://www.coursera.org/courses?query=analytical+skills" },
    { "skill": "data visualization", "course_link": "https://www.kaggle.com/learn/data-visualization" }
  ],
  "summary": "Kamu memiliki kecocokan 38% (masih rendah) untuk posisi Data Analyst. Skill yang perlu ditingkatkan antara lain: analytical skills, communication skills, data visualization, dan 3 skill lainnya."
}
```

---

## Formula Scoring

```
career_match_score = (0.60 × skill_match_score) + (0.25 × model_career_score) + (0.15 × quiz_score)
gap_score          = 100 - career_match_score
```

| Komponen | Bobot | Penjelasan |
|---|---|---|
| `skill_match_score` | 60% | % skill user yang cocok dengan required skills target job |
| `model_career_score` | 25% | Confidence model AI terhadap karier yang diprediksi |
| `quiz_score` | 15% | Skor mini-quiz sebagai validasi tambahan |

---

## Model Deep Learning

- **Arsitektur**: TensorFlow Functional API
- **Input**: MultiLabel binary vector dari skill user (557 fitur)
- **Hidden layers**: Dense(256) → BatchNorm → Dropout(0.3) → Dense(128) → Dropout(0.2) → Dense(64)
- **Output**: Softmax (127 kelas karier)
- **Custom Callback**: `SkillMapTrainingMonitor` — tracking best val_accuracy & training summary
- **Dataset training**: 21.394 baris, 127 job class, 557 skill unik

---

## Dokumentasi Lengkap

Lihat [API_DOCS.md](./API_DOCS.md) untuk panduan lengkap termasuk cara testing via Postman.
