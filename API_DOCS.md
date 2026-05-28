# SkillMap AI – API Documentation (v2.0.0)

## Overview

SkillMap AI adalah API yang membantu user memahami potensi karier dan kebutuhan skill enhancement berdasarkan CV. API membaca teks CV, mengekstrak skill, memprediksi rekomendasi karier, menghitung skor kecocokan karier, mendeteksi skill gap, dan memberikan rekomendasi learning path.

## Tech Stack

- **Framework**: FastAPI
- **Model**: TensorFlow (Functional API)
- **Dataset**: `combined_career_recommender_clean.csv`

---

## Setup & Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Training Model

Jalankan training script untuk memproses dataset dan membangun model:

```bash
python train_model.py
```

Output training akan tersimpan di folder `artifacts/`:
- `skillmap_model.keras` – Model deep learning
- `mlb.pkl` – MultiLabelBinarizer
- `label_encoder.pkl` – LabelEncoder
- `job_skill_map.json` – Mapping job → required skills
- `course_links.json` – Mapping skill → course link
- `known_skills.json` – Daftar semua skill yang dikenali

### 3. Menjalankan API

```bash
uvicorn app.main:app --reload
```

API akan berjalan di: `http://127.0.0.1:8000`

Swagger UI: `http://127.0.0.1:8000/docs`

---

## Endpoints

### `GET /`

Info dasar API.

**Response:**
```json
{
  "message": "SkillMap AI API is running",
  "version": "1.0.0",
  "endpoints": ["GET /", "GET /health", "GET /jobs", "GET /skills", "POST /predict"]
}
```

### `GET /health`

Health check.

**Response:**
```json
{
  "status": "healthy",
  "service": "skillmap-ai"
}
```

### `GET /jobs`

Menampilkan daftar semua **career categories** yang tersedia di sistem.
Semua label sudah dinormalisasi dari career taxonomy — bukan raw job title dari dataset.

**Response:**
```json
{
  "total": 33,
  "jobs": [
    "accountant", "admin officer", "business analyst", "civil engineer",
    "content creator", "customer service", "cyber security analyst",
    "data analyst", "data engineer", "data scientist", "devops engineer",
    "digital marketing", "embedded engineer", "entrepreneur",
    "finance staff", "graphic designer", "healthcare staff",
    "human resources", "it support", "legal staff",
    "machine learning engineer", "mechanical engineer", "network engineer",
    "operations staff", "product manager", "project manager",
    "quality assurance engineer", "sales executive", "sales manager",
    "software engineer", "teacher / lecturer", "ui ux designer", "web developer"
  ]
}
```

### `GET /skills`

Menampilkan daftar semua known_skills yang dikenali sistem.

**Response:**
```json
{
  "total": 120,
  "skills": ["python", "sql", "java", "..."]
}
```

### `POST /predict`

Endpoint utama untuk analisis CV.

**Request Body:**
```json
{
  "cv_text": "Saya memiliki kemampuan Python, SQL, Excel, HTML, CSS, dan problem solving skills.",
  "target_job": "data analyst",
  "quiz_score": 80
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `cv_text` | string | Yes | - | Teks CV user (minimal 1 karakter) |
| `target_job` | string | No | `""` | Target karier (opsional). Sistem auto-normalisasi ke career category terdekat. Contoh: "data analyst", "software dev", "web developer" |
| `quiz_score` | float | No | `80` | Skor mini-quiz (0-100) |

**Response:**
```json
{
  "detected_skills_from_cv": ["css", "excel", "html", "problem solving skills", "python", "sql"],
  "target_job": "data analyst",
  "recommended_career": "data analyst",
  "predicted_career_ai": "data analyst",
  "career_match_score": 60.5,
  "gap_score": 39.5,
  "skill_match_score": 37.5,
  "model_career_score": 85.2,
  "quiz_score": 80,
  "skill_dimiliki": ["python", "sql"],
  "skill_gap": ["analytical skills", "c++", "data visualization", "machine learning skills", "programming"],
  "learning_path": [
    {"skill": "analytical skills", "course_link": "https://www.coursera.org/courses?query=analytical+skills"},
    {"skill": "c++", "course_link": "https://www.w3schools.com/cpp/"}
  ],
  "summary": "Kamu memiliki kecocokan 61% (cukup baik) untuk posisi Data Analyst. Skill yang perlu ditingkatkan antara lain: analytical skills, c++, data visualization, dan 2 skill lainnya."
}
```

---

## Scoring Formula

```
career_match_score = (0.60 × skill_match_score) + (0.25 × model_career_score) + (0.15 × quiz_score)
gap_score = 100 - career_match_score
```

| Komponen | Bobot | Penjelasan |
|----------|-------|------------|
| `skill_match_score` | 60% | Persentase skill user yang cocok dengan required skills target job |
| `model_career_score` | 25% | Confidence/probability model AI terhadap karier yang diprediksi |
| `quiz_score` | 15% | Skor mini-quiz sebagai validasi tambahan |

---

## Logika Target Job

- Jika `target_job` diisi dan ada di database → `recommended_career = target_job`
- Jika `target_job` kosong atau tidak ditemukan → `recommended_career = predicted_career_ai`
- Skill gap dihitung berdasarkan `recommended_career`

---

## Project Structure

```
skillmap-ai/
├── app/
│   ├── main.py            # FastAPI application
│   └── inference.py        # Inference functions
├── artifacts/
│   ├── skillmap_model.keras
│   ├── mlb.pkl
│   ├── label_encoder.pkl
│   ├── job_skill_map.json
│   ├── course_links.json
│   └── known_skills.json
├── train_model.py          # Training script
├── combined_career_recommender_clean.csv
├── requirements.txt
├── API_DOCS.md
└── README.md
```

---

## Testing dengan Postman

1. Jalankan API: `uvicorn app.main:app --reload`
2. Buka Postman
3. Buat request baru:
   - Method: `POST`
   - URL: `http://127.0.0.1:8000/predict`
   - Body → raw → JSON:
     ```json
     {
       "cv_text": "Saya memiliki kemampuan Python, SQL, Excel, HTML, CSS, dan problem solving skills.",
       "target_job": "data analyst",
       "quiz_score": 80
     }
     ```
4. Klik Send

### Test tanpa target_job (sistem memilih karier):
```json
{
  "cv_text": "Experienced in Python, machine learning, data visualization, SQL, and statistical analysis.",
  "quiz_score": 75
}
```
