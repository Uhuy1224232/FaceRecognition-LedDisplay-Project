import time
import re
from pymongo import MongoClient

# --- Konfigurasi MongoDB (Server Docker) ---
MONGO_USER = "admin"
MONGO_PASS = "Admin123"
MONGO_HOST = "192.168.196.22"   # Ganti dengan IP server MongoDB kamu
MONGO_PORT = 27020
MONGO_AUTH_DB = "admin"         # Karena hanya admin yang bisa login

# Format URL koneksi
uri = f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_HOST}:{MONGO_PORT}/{MONGO_AUTH_DB}?authSource={MONGO_AUTH_DB}"

# Koneksi ke MongoDB
try:
    client = MongoClient(uri)
    db = client["face_recog_db"]         # Database tempat menyimpan log
    collection = db["detected_names"]    # Koleksi data deteksi wajah
    print("[✓] Koneksi ke MongoDB berhasil.")
except Exception as e:
    print(f"[!] Gagal konek ke MongoDB: {e}")
    exit(1)

# --- File log yang dipantau ---
LOG_FILE = "/projectface/PPM-Project/face_log.txt"

def tail_face_log(file_path):
    """Pantau file log secara real-time dan simpan data ke MongoDB."""
    with open(file_path, "r") as f:
        f.seek(0, 2)  # mulai dari akhir file
        print("[*] Monitoring face_log.txt untuk data baru...")

        while True:
            line = f.readline()
            if not line:
                time.sleep(1)
                continue

            # Deteksi pola log seperti: [12:30:01] MQTT terkirim: naufal
            match = re.search(r"\[(.*?)\]\s*MQTT terkirim:\s*(.*)", line)
            if match:
                log_time = match.group(1)
                detected_name = match.group(2).strip()

                data = {
                    "timestamp": time.strftime("%Y-%m-%d ") + log_time,
                    "detected_name": detected_name,
                    "raw_message": line.strip()
                }

                try:
                    collection.insert_one(data)
                    print(f"[+] Inserted ke MongoDB: {data}")
                except Exception as e:
                    print(f"[!] Gagal insert ke MongoDB: {e}")

if __name__ == "__main__":
    tail_face_log(LOG_FILE)
