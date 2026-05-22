"""
SkillMap AI - Training Script
=============================
Preprocessing dataset, training deep learning model (TF Functional API),
dan menyimpan semua artifacts untuk inference.

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer


# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "combined_career_recommender_clean.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. Skill alias map & invalid value list
# ============================================================
SKILL_ALIASES = {
    "problem solving": "problem solving skills",
    "analytic thinking": "analytical skills",
    "data visualization skills( power bi/ tableau )": "data visualization",
    "programming language skills": "programming",
}

INVALID_SKILLS = {
    "tidak", "ya", "na", "nan", "not applicable",
    "belum bekerja", "student (unemployed)", "",
}

INVALID_JOBS = {
    "tidak", "ya", "na", "nan", "not applicable",
    "belum bekerja", "student (unemployed)", "",
    "pc", "se",
}


# ============================================================
# 2. Text cleaning helpers
# ============================================================
def clean_text(text):
    """Bersihkan karakter aneh, newline, spasi berlebih."""
    if not isinstance(text, str):
        return ""
    # Hapus karakter unicode aneh
    text = text.replace("Â", "").replace("â€™", "'").replace("â€˜", "'")
    text = text.replace("Ã‰", "").replace("ÃŠ", "").replace("\r", " ")
    # Ganti newline dengan spasi
    text = text.replace("\n", " ")
    # Hapus spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_skill(skill):
    """Lowercase, strip, dan terapkan alias."""
    skill = clean_text(skill).lower().strip()
    # Hapus trailing semicolon/comma
    skill = skill.strip(";,. ")
    return SKILL_ALIASES.get(skill, skill)


def split_skills(raw_skills):
    """Split kolom Skill berdasarkan koma, titik koma, dan newline."""
    if not isinstance(raw_skills, str) or raw_skills.strip() == "":
        return []
    # Split by ; , \n
    parts = re.split(r"[;,\n]+", raw_skills)
    skills = []
    for part in parts:
        s = normalize_skill(part)
        if s and s not in INVALID_SKILLS and len(s) > 1:
            # Jangan masukkan value yang cuma angka
            if not s.replace(".", "", 1).isdigit():
                skills.append(s)
    return list(set(skills))


def clean_job(job):
    """Clean dan lowercase job title."""
    job = clean_text(job).lower().strip()
    job = job.strip(";,. \"'")
    return job


# ============================================================
# 3. Load & preprocess dataset
# ============================================================
print("=" * 60)
print("SkillMap AI - Training Pipeline")
print("=" * 60)

print("\n[1/7] Loading dataset...")
df = pd.read_csv(DATASET_PATH)
print(f"  Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
print(f"  Columns: {list(df.columns)}")

# Rename kolom job yang panjang
job_col_candidates = [c for c in df.columns if "first Job" in c or "first job" in c.lower()]
if job_col_candidates:
    old_name = job_col_candidates[0]
    df.rename(columns={old_name: "Pekerjaan_Pertama"}, inplace=True)
    print(f"  Renamed '{old_name[:50]}...' -> 'Pekerjaan_Pertama'")
else:
    # Fallback: cari kolom ke-7 (index 6)
    if len(df.columns) > 6:
        old_name = df.columns[6]
        df.rename(columns={old_name: "Pekerjaan_Pertama"}, inplace=True)
        print(f"  Renamed '{old_name[:50]}...' -> 'Pekerjaan_Pertama'")

print(f"  Columns after rename: {list(df.columns)}")


# ============================================================
# 4. Process skills & jobs
# ============================================================
print("\n[2/7] Preprocessing skills & jobs...")

df["Skill_clean"] = df["Skill"].apply(split_skills)
df["Job_clean"] = df["Pekerjaan_Pertama"].apply(clean_job)

# Filter baris dengan job invalid
df = df[~df["Job_clean"].isin(INVALID_JOBS)]
df = df[df["Job_clean"].str.len() > 1]

# Filter baris dengan skill kosong
df = df[df["Skill_clean"].apply(lambda x: len(x) > 0)]

print(f"  After filtering: {len(df)} rows")
print(f"  Unique jobs: {df['Job_clean'].nunique()}")

# Collect semua unique skills
all_skills = set()
for skills in df["Skill_clean"]:
    all_skills.update(skills)
print(f"  Unique skills: {len(all_skills)}")


# ============================================================
# 5. Build job_skill_map
# ============================================================
print("\n[3/7] Building job_skill_map...")

job_skill_counter = {}
for _, row in df.iterrows():
    job = row["Job_clean"]
    skills = row["Skill_clean"]
    if job not in job_skill_counter:
        job_skill_counter[job] = Counter()
    job_skill_counter[job].update(skills)

# Untuk setiap job, ambil top skills (yang muncul minimal 1 kali)
job_skill_map = {}
for job, counter in job_skill_counter.items():
    # Ambil top 10 skills, atau semua jika kurang dari 10
    top_skills = [skill for skill, count in counter.most_common(10)]
    if top_skills:
        job_skill_map[job] = top_skills

print(f"  Jobs in map: {len(job_skill_map)}")


# ============================================================
# 6. Filter rare classes & encode data untuk training
# ============================================================
print("\n[4/7] Encoding data...")

# Filter jobs dengan < 2 sampel (agar stratified split bisa jalan)
job_counts = df["Job_clean"].value_counts()
valid_jobs = job_counts[job_counts >= 2].index
df_train = df[df["Job_clean"].isin(valid_jobs)].copy()

removed_jobs = len(job_counts) - len(valid_jobs)
print(f"  Removed {removed_jobs} rare job classes (< 2 samples)")
print(f"  Training data: {len(df_train)} rows, {df_train['Job_clean'].nunique()} jobs")

# MultiLabelBinarizer untuk skills (fit on ALL data untuk coverage)
mlb = MultiLabelBinarizer()
mlb.fit(df["Skill_clean"])  # fit on semua data
X = mlb.transform(df_train["Skill_clean"])  # transform training subset

# LabelEncoder untuk jobs
le = LabelEncoder()
y_encoded = le.fit_transform(df_train["Job_clean"])

num_skills = X.shape[1]
num_jobs = len(le.classes_)

print(f"  Input features (skills): {num_skills}")
print(f"  Output classes (jobs): {num_jobs}")
print(f"  Training samples: {X.shape[0]}")

# One-hot encode target
y_onehot = tf.keras.utils.to_categorical(y_encoded, num_classes=num_jobs)

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")


# ============================================================
# 7. Custom Callback - SkillMapTrainingMonitor
# ============================================================
class SkillMapTrainingMonitor(tf.keras.callbacks.Callback):
    """
    Custom Callback untuk monitoring training SkillMap model.

    Fitur:
    - Track best accuracy dan best epoch
    - Log ringkasan per epoch
    - Print training summary di akhir
    """

    def __init__(self):
        super().__init__()
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        self.history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get("accuracy", 0)
        val_acc = logs.get("val_accuracy", 0)
        loss = logs.get("loss", 0)
        val_loss = logs.get("val_loss", 0)

        self.history["accuracy"].append(acc)
        self.history["val_accuracy"].append(val_acc)
        self.history["loss"].append(loss)
        self.history["val_loss"].append(val_loss)

        if val_acc > self.best_val_accuracy:
            self.best_val_accuracy = val_acc
            self.best_epoch = epoch + 1
            print(f"  >> New best val_accuracy: {val_acc:.4f} at epoch {epoch + 1}")

    def on_train_end(self, logs=None):
        print("\n" + "=" * 60)
        print("Training Summary (SkillMapTrainingMonitor)")
        print("=" * 60)
        print(f"  Total epochs trained   : {len(self.history['accuracy'])}")
        print(f"  Best val_accuracy      : {self.best_val_accuracy:.4f}")
        print(f"  Best epoch             : {self.best_epoch}")
        if self.history["accuracy"]:
            print(f"  Final train_accuracy   : {self.history['accuracy'][-1]:.4f}")
            print(f"  Final val_accuracy     : {self.history['val_accuracy'][-1]:.4f}")
            print(f"  Final train_loss       : {self.history['loss'][-1]:.4f}")
            print(f"  Final val_loss         : {self.history['val_loss'][-1]:.4f}")
        print("=" * 60)


# ============================================================
# 8. Build & Train model (TF Functional API)
# ============================================================
print("\n[5/7] Building model (TensorFlow Functional API)...")

inputs = tf.keras.Input(shape=(num_skills,), name="skill_input")
x = tf.keras.layers.Dense(256, activation="relu", name="hidden_1")(inputs)
x = tf.keras.layers.BatchNormalization(name="bn_1")(x)
x = tf.keras.layers.Dropout(0.3, name="dropout_1")(x)
x = tf.keras.layers.Dense(128, activation="relu", name="hidden_2")(x)
x = tf.keras.layers.Dropout(0.2, name="dropout_2")(x)
x = tf.keras.layers.Dense(64, activation="relu", name="hidden_3")(x)
outputs = tf.keras.layers.Dense(num_jobs, activation="softmax", name="career_output")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="SkillMap_Career_Model")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

print("\n[6/7] Training model...")

# Callbacks
monitor_callback = SkillMapTrainingMonitor()
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[monitor_callback, early_stop],
    verbose=1,
)

# Evaluate
print("\n[Evaluation]")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"  Test Loss    : {test_loss:.4f}")
print(f"  Test Accuracy: {test_acc:.4f}")


# ============================================================
# 9. Save all artifacts
# ============================================================
print("\n[7/7] Saving artifacts...")

# Model
model.save(ARTIFACT_DIR / "skillmap_model.keras")
print(f"  [OK] Model saved -> artifacts/skillmap_model.keras")

# MultiLabelBinarizer
with open(ARTIFACT_DIR / "mlb.pkl", "wb") as f:
    pickle.dump(mlb, f)
print(f"  [OK] MLB saved -> artifacts/mlb.pkl")

# LabelEncoder
with open(ARTIFACT_DIR / "label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print(f"  [OK] LabelEncoder saved -> artifacts/label_encoder.pkl")

# job_skill_map
with open(ARTIFACT_DIR / "job_skill_map.json", "w", encoding="utf-8") as f:
    json.dump(job_skill_map, f, indent=2, ensure_ascii=False)
print(f"  [OK] job_skill_map saved -> artifacts/job_skill_map.json ({len(job_skill_map)} jobs)")

# known_skills
known_skills_list = sorted(list(all_skills))
with open(ARTIFACT_DIR / "known_skills.json", "w", encoding="utf-8") as f:
    json.dump(known_skills_list, f, indent=2, ensure_ascii=False)
print(f"  [OK] known_skills saved -> artifacts/known_skills.json ({len(known_skills_list)} skills)")

# course_links - generate dari known_skills (tetap pakai mapping manual + fallback)
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
}

with open(ARTIFACT_DIR / "course_links.json", "w", encoding="utf-8") as f:
    json.dump(MANUAL_COURSE_LINKS, f, indent=2, ensure_ascii=False)
print(f"  [OK] course_links saved -> artifacts/course_links.json")


# ============================================================
# Done
# ============================================================
print("\n" + "=" * 60)
print("Training pipeline complete!")
print(f"  Dataset rows used : {len(df)}")
print(f"  Unique jobs       : {num_jobs}")
print(f"  Unique skills     : {num_skills}")
print(f"  Model accuracy    : {test_acc:.4f}")
print(f"  Artifacts dir     : {ARTIFACT_DIR}")
print("=" * 60)
