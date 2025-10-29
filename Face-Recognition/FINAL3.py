import cv2
import pickle
import time
import numpy as np
from datetime import datetime
from insightface.app import FaceAnalysis
import threading
import paho.mqtt.client as mqtt
import sys
import os
import socket

# =========================
# KONFIGURASI
# =========================
RTSP_URL = "rtsp://admin:BABKQU@192.168.196.93:554/h264/ch1/main/av_stream"
LOG_FILE = "face_log.txt"
SIMILARITY_THRESHOLD = 0.35
LINE_TOP = 200
LINE_BOTTOM = 310
DETECTION_COOLDOWN = 8
HOLD_TIME = 5
FPS_INTERVAL = 2.0
RECONNECT_DELAY = 3
RECONNECT_BACKOFF_MAX = 60
TAMU_STABILITY_FRAMES = 10  # jika ada known face, tamu harus muncul N frame berturut2 untuk valid

MQTT_BROKER = "192.168.196.22"
MQTT_PORT = 1883
MQTT_TOPIC = "led/display"
MQTT_RETRY_INTERVAL = 5  # detik interval reconnect MQTT

# =========================
# LOGGING
# =========================
def log_event(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

# =========================
# CEK HOST
# =========================
def host_reachable(host, timeout=1):
    try:
        sock = socket.create_connection((host, 554), timeout=timeout)
        sock.close()
        return True
    except Exception:
        return False

# =========================
# LOAD WAJAH
# =========================
with open("face_embeddings8.pkl", "rb") as f:
    data = pickle.load(f)

known_names = list(data.keys())
known_encodings = np.array(list(data.values()))
print(f"Dataset wajah terload: {known_names}")

# =========================
# INSIGHTFACE GPU/CPU
# =========================
print("Memuat model InsightFace...")

try:
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("Model dijalankan di GPU.")
except Exception as e:
    print(f"Gagal menggunakan GPU ({e}), beralih ke CPU...")
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

# =========================
# MQTT SETUP + AUTO RECONNECT
# =========================
client = mqtt.Client()
mqtt_connected = False
last_mqtt_attempt = 0

def on_connect(client, userdata, flags, rc):
    global mqtt_connected
    if rc == 0:
        mqtt_connected = True
        log_event("MQTT: Terhubung ke broker.")
    else:
        mqtt_connected = False
        log_event(f"MQTT: Gagal koneksi (kode {rc})")

def on_disconnect(client, userdata, rc):
    global mqtt_connected
    mqtt_connected = False
    log_event("MQTT: Terputus dari broker.")

client.on_connect = on_connect
client.on_disconnect = on_disconnect

def mqtt_loop():
    global mqtt_connected, last_mqtt_attempt
    while True:
        if not mqtt_connected and time.time() - last_mqtt_attempt > MQTT_RETRY_INTERVAL:
            last_mqtt_attempt = time.time()
            try:
                client.connect(MQTT_BROKER, MQTT_PORT, 60)
                client.loop_start()
            except Exception as e:
                log_event(f"MQTT reconnect gagal: {e}")
        time.sleep(1)

mqtt_thread = threading.Thread(target=mqtt_loop, daemon=True)
mqtt_thread.start()

def send_mqtt(message):
    global mqtt_connected
    if mqtt_connected:
        try:
            client.publish(MQTT_TOPIC, message)
        except Exception as e:
            log_event(f"MQTT publish error: {e}")
    else:
        log_event("MQTT belum terhubung — pesan tidak dikirim.")

# =========================
# RTSP READER
# =========================
class RTSPReader:
    def __init__(self, url):
        self.url = url
        self.cap = None
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.error_count = 0
        self.max_errors = 10
        self.backoff = RECONNECT_DELAY
        self._connect()
        t = threading.Thread(target=self.update, daemon=True)
        t.start()

    def _connect(self):
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        except Exception:
            pass

        self.backoff = RECONNECT_DELAY
        while not self.stopped:
            log_event("Menghubungkan ulang ke kamera RTSP...")
            try:
                cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                time.sleep(1)
                if cap.isOpened():
                    self.cap = cap
                    self.error_count = 0
                    log_event("Terhubung ke kamera RTSP.")
                    return
                cap.release()
            except Exception as e:
                log_event(f"Gagal membuka VideoCapture: {e}")

            try:
                host = self.url.split("//")[-1].split("/")[0].split("@")[-1].split(":")[0]
                reachable = host_reachable(host)
            except Exception:
                reachable = False

            if not reachable:
                log_event("Host tidak dapat dijangkau, menunggu sebelum retry...")
                sleep_time = min(self.backoff * 2, RECONNECT_BACKOFF_MAX)
            else:
                sleep_time = min(self.backoff, RECONNECT_BACKOFF_MAX)

            time.sleep(sleep_time)
            self.backoff = min(self.backoff * 2, RECONNECT_BACKOFF_MAX)

    def update(self):
        while not self.stopped:
            try:
                if self.cap is None:
                    self._connect()
                    time.sleep(0.1)
                    continue

                ret, frame = self.cap.read()
                if not ret or frame is None or frame.size == 0:
                    self.error_count += 1
                    if self.error_count >= self.max_errors:
                        log_event("Deteksi error stream - reconnect otomatis...")
                        self._connect()
                        self.error_count = 0
                    time.sleep(0.05)
                    continue

                if np.mean(frame) < 1:
                    self.error_count += 1
                    if self.error_count >= self.max_errors:
                        log_event("Frame hitam - reconnect...")
                        self._connect()
                        self.error_count = 0
                    time.sleep(0.05)
                    continue

                with self.lock:
                    self.frame = frame
                self.error_count = 0

            except Exception as e:
                log_event(f"Exception di RTSPReader.update: {e}")
                self._connect()
                time.sleep(1)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        if self.cap:
            self.cap.release()

# =========================
# MAIN LOOP
# =========================
reader = RTSPReader(RTSP_URL)
time.sleep(1)

frame_count = 0
fps_start = time.time()
last_sent = {}
known_hold = {}     # nama -> waktu terakhir dikenali
fps_display = 0.0

# tracking tamu stabilitas
tamu_stable_count = 0

log_event("Streaming dimulai tanpa GUI, dengan Line Detector, Hold Logic, dan Auto-Reconnect aktif.")

try:
    while True:
        frame = reader.read()
        if frame is None:
            time.sleep(0.05)
            continue

        faces = app.get(frame)
        detected_names_raw = []  # raw names detected this frame (may include multiple entries)
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            center_y = int((y1 + y2) / 2)

            embedding = face.normed_embedding
            similarities = np.dot(known_encodings, embedding)
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]

            name = known_names[best_idx] if best_similarity > SIMILARITY_THRESHOLD else "Tamu"

            if LINE_TOP < center_y < LINE_BOTTOM:
                detected_names_raw.append(name)
                if name != "Tamu":
                    known_hold[name] = time.time()

        # build current known set from raw detections (excluding Tamu)
        current_known = {n for n in detected_names_raw if n != "Tamu"}

        # include names still within hold
        current_time = time.time()
        for name, t in list(known_hold.items()):
            if current_time - t <= HOLD_TIME:
                current_known.add(name)
            else:
                del known_hold[name]

        # tamu presence this frame
        tamu_present = any(n == "Tamu" for n in detected_names_raw)

        # Decide whether to include "Tamu"
        include_tamu = False
        if current_known:
            # there are known faces active -> require stability for tamu
            if tamu_present:
                tamu_stable_count += 1
            else:
                tamu_stable_count = 0
            if tamu_stable_count >= TAMU_STABILITY_FRAMES:
                include_tamu = True
        else:
            # no known faces -> tamu allowed immediately if present
            if tamu_present:
                include_tamu = True
                tamu_stable_count = 0  # reset stable counter since not needed now
            else:
                tamu_stable_count = 0

        # prepare final detected_names set (include known + tamu if allowed)
        final_detected = set(sorted(current_known))
        if include_tamu:
            final_detected.add("Tamu")

        # Kirim MQTT jika ada yang terdeteksi (gabungan)
        if final_detected:
            msg = ",".join(sorted(final_detected))
            now = time.time()
            if msg not in last_sent or now - last_sent[msg] > DETECTION_COOLDOWN:
                send_mqtt(msg)
                log_event(f"MQTT terkirim: {msg}")
                last_sent[msg] = now

        # Hitung & tampilkan FPS
        frame_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= FPS_INTERVAL:
            fps_display = frame_count / elapsed
            frame_count = 0
            fps_start = time.time()
        sys.stdout.write(f"\rFPS: {fps_display:.2f}")
        sys.stdout.flush()

except KeyboardInterrupt:
    print("\nProgram dihentikan oleh pengguna.")
finally:
    reader.stop()
    client.loop_stop()
    print("\nProgram selesai.")

