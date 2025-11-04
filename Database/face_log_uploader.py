import time
import re
from pymongo import MongoClient

# --- Konfigurasi MongoDB ---
MONGO_USER = "face_user"
MONGO_PASS = "Face123!"
MONGO_HOST = "mongodb"        # pakai hostname service (bukan IP)
MONGO_PORT = 27017
MONGO_AUTH_DB = "admin"

uri = f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_HOST}:{MONGO_PORT}/?authSource={MONGO_AUTH_DB}"

client = MongoClient(uri)
db = client["face_recog_db"]
collection = db["detected_names"]

# --- File log yang dipantau ---
LOG_FILE = "/projectface/PPM-Project/face_log.txt"

def tail_face_log(file_path):
    with open(file_path, "r") as f:
        f.seek(0, 2)
        while True:
            line = f.readline()
            if not line:
                time.sleep(1)
                continue

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
                    print(f"[+] Inserted to MongoDB: {data}")
                except Exception as e:
                    print(f"[!] Gagal insert ke MongoDB: {e}")

if __name__ == "__main__":
    print("[*] Monitoring face_log.txt for detected names...")
    tail_face_log(LOG_FILE)
