# SkillMap AI

> AI/API service for career recommendation — part of the **SkillMap** capstone project.  
> Reads CV text, detects skills, recommends careers, and computes match/gap scores with a learning path.

**Live API:** https://api-skillmap-ai.up.railway.app  
**Version:** v3.2

---

## Table of Contents

- [Project Overview](#project-overview)
- [Main Features](#main-features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Model & Training Pipeline](#model--training-pipeline)
- [Hybrid Recommendation Approach](#hybrid-recommendation-approach)
- [API Endpoints](#api-endpoints)
- [Predict Request & Response Example](#predict-request--response-example)
- [Output Field Explanation](#output-field-explanation)
- [Artifacts](#artifacts)
- [How to Run Locally](#how-to-run-locally)
- [Deployment](#deployment)
- [Limitations](#limitations)
- [Latest Update — v3.2](#latest-update--v32)
- [Notes for Frontend Integration](#notes-for-frontend-integration)

---

## Project Overview

SkillMap AI is a FastAPI-based REST API that serves as the AI backend for the SkillMap application.
It receives a raw CV text from the frontend or backend service, processes it through a hybrid pipeline,
and returns structured career recommendations along with skill gap analysis and a personalized learning path.

---

## Main Features

| Feature | Description |
|---|---|
| Skill extraction | Detects skills from raw CV text using a curated skill vocabulary (371 skills) |
| Career recommendation | Recommends the most relevant career using a hybrid rule-based + AI approach |
| Multiple career options | Returns a dynamic list of relevant careers via `career_recommendations` |
| Career match score | Quantifies how well the user's skills match the target career (0–100) |
| Gap score | Inverse of match score; represents how much room for improvement exists |
| Skill gap detection | Lists specific skills the user is missing for the target career |
| Learning path | Returns course links for each missing skill |
| Summary | Human-readable summary of the analysis result |

---

## Tech Stack

| Component | Library / Framework |
|---|---|
| API Framework | FastAPI + Uvicorn |
| Deep Learning | TensorFlow (Functional API) |
| Encoding | scikit-learn (MultiLabelBinarizer, LabelEncoder) |
| Data Processing | pandas, numpy |
| Deployment | Railway |

---

## Dataset

SkillMap AI v3.2 uses a **triple-dataset training pipeline**:

| Dataset | File | Rows (raw) | Description |
|---|---|---|---|
| CSV1 | `combined_career_recommender_clean.csv` | 22,147 | Primary dataset — English job titles & skills |
| CSV2 | `career_recommender_indonesia_cleaned.csv` | ~441 valid | Indonesian career data (strict validation applied) |
| CSV3 | `New_Career_Recommendation_Cleaned.csv` | 48,013 total / **4,200 used** | New dataset from the Data Science team — capped at 300 samples per class to prevent dominance |

> **Note on CSV3 capping:** CSV3 contains 48,013 rows across 14 career labels with a relatively uniform skill distribution.
> To prevent this dataset from diluting the existing model signal, a maximum of **300 samples per class** was applied,
> resulting in ~4,200 rows used out of 48,013.

---

## Model & Training Pipeline

### Architecture

- **Framework:** TensorFlow Functional API
- **Input:** Multi-label binary vector of user skills (371 features)
- **Hidden layers:** Dense(256) → BatchNormalization → Dropout(0.3) → Dense(128) → Dropout(0.2) → Dense(64)
- **Output:** Softmax (41 output classes)
- **Custom callback:** `SkillMapTrainingMonitor` — tracks best `val_accuracy` and prints training summary

### Training Stats (v3.2)

| Metric | Value |
|---|---|
| Training rows | ~19,800 |
| Input features | 371 |
| Output classes (model) | 41 |
| Career categories (/jobs) | 109 |
| job_skill_map entries | 109 |
| known_skills | 371 |

### Honest Note on Model Accuracy

The deep learning model is used as **a supporting component**, not as the sole decision maker.
Model-only classification accuracy is currently modest due to:
- Dataset imbalance across 41 career classes
- Overlapping skill distributions between similar careers
- Limited signal from some newer categories (e.g., animator, diplomat, business consultant from CSV3)

The final recommendation quality is strengthened by the hybrid approach described below.

---

## Hybrid Recommendation Approach

The recommendation pipeline combines multiple signals:

```
1. Keyword matching     → scan CV text for career-specific keywords (bilingual ID + EN)
2. Skill overlap        → compare detected skills with job_skill_map required skills
3. Model prediction     → deep learning model provides a supporting career class prediction
4. Hybrid decision      → if rule-based score ≥ 15%, rule-based wins; otherwise fall back to model
```

**Weight for scoring:**

```
career_match_score = (0.60 × skill_match_score) + (0.25 × model_career_score) + (0.15 × quiz_score)
gap_score          = 100 − career_match_score
```

| Component | Weight | Description |
|---|---|---|
| `skill_match_score` | 60% | % of required skills the user already has |
| `model_career_score` | 25% | Model confidence adapted from rule/model hybrid score |
| `quiz_score` | 15% | Optional quiz result (0–100) sent from frontend |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check — returns service status |
| GET | `/jobs` | List of all 109 supported career categories |
| GET | `/skills` | List of all 371 recognized skills |
| GET | `/info` | API version and artifact metadata |
| POST | `/predict` | Main endpoint — analyze CV and return career recommendations |
| GET | `/docs` | Swagger UI (interactive API documentation) |

---

## Predict Request & Response Example

### Request

```http
POST /predict
Content-Type: application/json
```

```json
{
  "cv_text": "Saya lulusan ilmu hukum dengan pengalaman magang di kantor hukum dan bagian legal corporate. Saya terbiasa membantu legal research, membaca peraturan, melakukan contract review sederhana, mengelola dokumen legal, menyusun laporan administrasi legal, membantu legal drafting sederhana, serta memastikan dokumen sesuai dengan regulasi yang berlaku. Saya memiliki skill legal research, contract review, legal administration, regulation knowledge, compliance, document management, microsoft office, reporting, communication skills, attention to detail, critical thinking, problem solving skills, teamwork, dan time management.",
  "target_job": "",
  "quiz_score": 78
}
```

> `target_job` is **optional**. Leave it empty to let the system recommend a career automatically.  
> `quiz_score` must be between **0 and 100**.

### Response (condensed)

```json
{
  "detected_skills_from_cv": [
    "attention to detail", "communication skills", "compliance",
    "critical thinking", "document management", "legal knowledge",
    "problem solving skills", "regulation knowledge", "reporting",
    "teamwork", "time management"
  ],
  "target_job": null,
  "recommended_career": "paralegal",
  "career_recommendations": [
    { "career": "paralegal",   "score": 73.0,  "source": "rule_based" },
    { "career": "legal staff", "score": 49.43, "source": "rule_based" }
  ],
  "recommendation_source": "rule_based",
  "raw_model_prediction": "data analyst",
  "career_match_score": 47.27,
  "gap_score": 52.73,
  "skill_match_score": 55.6,
  "model_career_score": 0.0,
  "quiz_score": 78.0,
  "skill_dimiliki": ["attention to detail", "communication skills", "compliance", "reporting"],
  "skill_gap": ["documentation", "legal research", "microsoft office", "organizational skills"],
  "learning_path": [
    { "skill": "legal research", "course_link": "https://www.coursera.org/courses?query=legal+research" },
    { "skill": "microsoft office", "course_link": "https://www.coursera.org/courses?query=microsoft+office" }
  ],
  "summary": "Based on your CV, the system recommends Paralegal. Your current career match score is 47.27%. Focus on improving: documentation, legal research, microsoft office, and 1 more skill."
}
```

---

## Output Field Explanation

| Field | Type | Description |
|---|---|---|
| `detected_skills_from_cv` | `list[str]` | Skills extracted from the CV text |
| `target_job` | `str \| null` | Normalized target job if provided by the user |
| `recommended_career` | `str` | **Primary output** — the top recommended career |
| `career_recommendations` | `list[object]` | **Dynamic list** of relevant careers with scores and sources |
| `recommendation_source` | `str` | How the recommendation was made (see below) |
| `raw_model_prediction` | `str` | Direct model output — **for debug only**, not the main output |
| `career_match_score` | `float` | Overall match score (0–100) |
| `gap_score` | `float` | Skill gap percentage (100 − career_match_score) |
| `skill_match_score` | `float` | % of required skills the user already has |
| `model_career_score` | `float` | Adapted model confidence contribution |
| `quiz_score` | `float` | Quiz score passed in the request |
| `skill_dimiliki` | `list[str]` | Skills the user already has (matched to target career) |
| `skill_gap` | `list[str]` | Skills the user is missing for the target career |
| `learning_path` | `list[object]` | Course recommendations per missing skill |
| `summary` | `str` | Human-readable summary of the result |

### `career_recommendations` Format

Each item in the list contains:

```json
{
  "career": "paralegal",
  "score": 73.0,
  "source": "rule_based"
}
```

**Rules:**
- The list is **dynamic** — can contain 1, 2, 4, 8, or more items depending on relevance
- Maximum of **8 items** per response
- `recommended_career` is always equal to `career_recommendations[0].career`
- If the user provides a `target_job`, `career_recommendations` will contain exactly **1 item** with `source: "user_input"`
- Only careers with sufficient relevance score are included — irrelevant careers are not forced in

### `recommendation_source` Values

| Value | Meaning |
|---|---|
| `user_input` | User explicitly set `target_job` |
| `rule_based` | Recommendation from keyword + skill matching |
| `hybrid` | Rule-based and model prediction agreed on the same career |
| `model` | Rule-based confidence was low; model prediction was used as fallback |

---

## Artifacts

All artifacts are stored in the `artifacts/` directory and are loaded at API startup.

| File | Description |
|---|---|
| `skillmap_model.keras` | Trained TensorFlow model (Functional API) |
| `mlb.pkl` | MultiLabelBinarizer — encodes skill list into feature vector |
| `label_encoder.pkl` | LabelEncoder — maps model output index to career class name (41 classes) |
| `job_skill_map.json` | Curated mapping of career → required skills (109 entries) |
| `known_skills.json` | Full list of recognized skills (371 skills) |
| `career_categories.json` | All 109 supported career categories for `/jobs` endpoint |
| `model_classes.json` | The 41 career classes the model was trained on |
| `course_links.json` | Mapping of skill → course URL for learning path generation |

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/kerb0y/skillmap-ai
cd skillmap-ai
```

### 2. Create a virtual environment and install dependencies

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. (Optional) Retrain the model

Artifacts are already committed to the repository. Only run this if you want to rebuild the model from the datasets:

```bash
python train_model.py
python fix_skill_map.py
```

### 4. Start the API server

```bash
python -m uvicorn app.main:app --reload
```

API will be available at: `http://127.0.0.1:8000`  
Swagger UI: `http://127.0.0.1:8000/docs`

---

## Deployment

### Railway

This project is deployed to [Railway](https://railway.app). The Uvicorn server must be started with `0.0.0.0` as the host and the `PORT` environment variable provided by Railway:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Set the start command in Railway's service settings accordingly.

### Docker (if Dockerfile is added)

```bash
docker build -t skillmap-ai .
docker run -p 8000:8000 skillmap-ai
```

---

## Limitations

- **Model accuracy is modest** — the deep learning component alone has limited classification accuracy due to class imbalance and overlapping skill patterns. The hybrid approach compensates for this.
- **Niche majors may not map well** — fields such as oceanography, geophysics, actuarial science, archaeology, and other rare disciplines are not well represented in the training dataset. Users with these backgrounds may receive less accurate recommendations.
- **Language mixing** — the system handles both Indonesian and English keywords, but mixed-language CVs may cause incomplete skill extraction.
- **Bias toward common fields** — careers in IT, data, and business are better represented in the dataset than highly specialized non-IT careers.
- **Ongoing improvement** — the system can be improved incrementally by adding new keyword mappings, curating new job_skill_map entries, and expanding training data for underrepresented career categories.

---

## Latest Update — v3.2

### What's New

- **Triple-dataset training pipeline** — CSV1 + CSV2 + CSV3 with per-class sample capping for CSV3
- **Expanded career taxonomy** — 109 career categories (up from 100), 41 model classes (up from 35)
- **9 new career categories added:**
  - From CSV3 (model-trained): `animator`, `diplomat`, `business consultant`
  - Curated (rule-based only): `forestry officer`, `environmental officer`, `conservation officer`, `sustainability officer`, `gis analyst`, `field officer`
- **Reduced IT bias** — better coverage for non-IT domains including law, forestry, environment, logistics, and social fields
- **`career_recommendations` field** — the `/predict` response now includes a dynamic list of relevant career options, not just a single top pick
- **Improved skill validation** — single-character tokens, noise, and invalid skills are filtered from all skill fields

### Taxonomy Coverage

| Area | Supported Careers |
|---|---|
| IT & Software | software engineer, web developer, mobile developer, data analyst, data scientist, data engineer, machine learning engineer, devops engineer, cloud engineer, cyber security analyst, ai engineer, solutions architect, embedded engineer, network engineer, erp consultant, and more |
| Business & Management | business analyst, business consultant, product manager, project manager, operations manager, management trainee, entrepreneur, general manager |
| Finance & Accounting | accountant, financial analyst, finance staff, tax staff, auditor, banking staff, insurance agent, investment analyst |
| Legal | legal staff, paralegal, company secretary |
| HR & Admin | human resources, recruiter, admin officer, secretary, receptionist, office staff |
| Sales & Marketing | sales executive, sales manager, marketing staff, digital marketing, social media specialist, customer service, public relations, copywriter, content creator |
| Engineering (Non-IT) | civil engineer, mechanical engineer, electrical engineer, industrial engineer, architect, drafter, surveyor, safety officer |
| Forestry & Environment | forestry officer, environmental officer, conservation officer, sustainability officer, gis analyst, field officer |
| Healthcare | doctor, nurse, pharmacist, healthcare staff, laboratory analyst |
| Education | teacher, lecturer, tutor, training staff, researcher |
| Creative & Media | graphic designer, video editor, photographer, journalist, writer, animator, translator |
| Hospitality & F&B | chef / cook, barista, hospitality staff, tourism staff |
| Logistics & Supply Chain | logistics staff, warehouse staff, supply chain staff, procurement staff, purchasing staff, production staff |
| Diplomatic & Advisory | diplomat, business consultant |

---

## Notes for Frontend Integration

1. **Use `recommended_career`** as the primary display value — it is always the top result.
2. **Use `career_recommendations`** to display multiple career options (e.g., a dropdown or card list) so users can explore alternatives.
3. **Do not display `raw_model_prediction`** to end users — it is a raw model output intended for debugging and internal use only.
4. `career_match_score` and `gap_score` always sum to 100.
5. `skill_dimiliki` and `skill_gap` are computed relative to `recommended_career`, even when multiple alternatives are shown.
6. If you send an empty string for `target_job`, the system will auto-recommend. If you send a specific value (e.g., `"data analyst"`), the system will use it directly and return `source: "user_input"`.
7. All scores are floats in the range `0.0–100.0`.
