import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis

# --------------------------------------
# KONFIGURASI
# --------------------------------------
DATASET_DIR = "captures"
OUTPUT_FILE = "face_embeddings8.pkl"   # File hasil training

# --------------------------------------
# INISIALISASI MODEL
# --------------------------------------
print("🔍 Memuat model InsightFace (CPU mode)...")
app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

# --------------------------------------
# FUNGSI MEMBUAT EMBEDDING
# --------------------------------------
def get_face_embedding(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Gagal baca gambar: {image_path}")
        return None
    faces = app.get(image)
    if len(faces) == 0:
        print(f"🚫 Tidak ada wajah terdeteksi: {image_path}")
        return None
    return faces[0].normed_embedding

# --------------------------------------
# LOOP SEMUA DATASET
# --------------------------------------
database = {}

for person_name in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue

    print(f"\n👤 Memproses {person_name}...")
    embeddings = []
    for filename in os.listdir(person_dir):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        image_path = os.path.join(person_dir, filename)
        embedding = get_face_embedding(image_path)
        if embedding is not None:
            embeddings.append(embedding)

    if embeddings:
        mean_embedding = np.mean(embeddings, axis=0)
        database[person_name] = mean_embedding
        print(f"✅ {len(embeddings)} wajah diproses untuk {person_name}")
    else:
        print(f"❌ Tidak ada embedding valid untuk {person_name}")

# --------------------------------------
# SIMPAN DATABASE EMBEDDING
# --------------------------------------
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(database, f)

print("\n🎯 Training selesai! Embedding disimpan ke:", OUTPUT_FILE)
