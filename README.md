# 📘 Face Recognition System dengan LED Display

> **Sistem pengenalan wajah real-time menggunakan Jetson Orin, Raspberry Pi, dan LED Matrix**

**Versi:** 1.1 | **Update Terakhir:** 28 Oktober 2025

---

## 📑 Daftar Isi

- [Overview Sistem](#overview-sistem)
- [Arsitektur](#arsitektur)
- [Requirements](#requirements)
- [Setup Guide](#setup-guide)
  - [1. Jetson Orin (Face Recognition)](#1-jetson-orin-face-recognition)
  - [2. Raspberry Pi (LED Controller)](#2-raspberry-pi-led-controller)
  - [3. MongoDB (Database)](#3-mongodb-database)
- [Troubleshooting](#troubleshooting)
- [Maintenance](#maintenance)

---

## Overview Sistem

Sistem ini terdiri dari 3 komponen utama:

1. **Jetson Orin** - Melakukan face recognition dari RTSP camera dengan auto-reconnect
2. **Raspberry Pi** - Mengontrol LED Matrix 256x64 untuk menampilkan nama dan jam
3. **MongoDB** - Menyimpan log deteksi

**Flow Kerja:**
```
RTSP Camera → Jetson (Face Recognition) → MQTT → Raspberry Pi → LED Display
                      ↓
                  face_log.txt → MongoDB (Logging)
```

---

## Arsitektur

```
┌─────────────────┐      MQTT       ┌──────────────────┐      GPIO      ┌─────────────────┐
│  Jetson Orin    │ ──────────────> │  Raspberry Pi    │ ────────────> │  LED Matrix     │
│  (Face Recog)   │   192.168       │  (LED Control)   │   (Parallel)   │     256x64      │
│                 │   .196.195      │  + MQTT Broker   │                │  (4x 64x64)     │
└─────────────────┘                 └──────────────────┘                └─────────────────┘
        │                                    │
        │                                    │
        v                                    v
  ┌──────────┐                        ┌──────────┐
  │ face_log │                        │ MongoDB  │
  │   .txt   │ ──────────────────────>│ Database │
  └──────────┘   log_monitor.py       └──────────┘
```

---

## Requirements

### Hardware

| Komponen | Spesifikasi | Keterangan |
|----------|-------------|------------|
| **Jetson Orin** | JetPack 5.1.2, CUDA 11.4+ | Face recognition processing |
| **Raspberry Pi** | Model yang digunakan | LED controller + MQTT Broker |
| **LED Matrix** | 256x64 (4 panel 64x64 HUB75) | Display output |
| **RTSP Camera** | Hikvision DS-2CD2xxx | IP: 192.168.196.93 |
| **Power Supply** | 5V 10A (terpisah) | Untuk LED Matrix |

### Software

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.8+ | Runtime |
| ONNX Runtime | 1.14.1 (GPU) | Inference engine |
| InsightFace | buffalo_l model | Face detection/recognition |
| OpenCV | 4.x | Video processing |
| Paho MQTT | 1.x | Message broker |
| rpi-rgb-led-matrix | Latest | LED control library |
| MongoDB | 4.x+ | Database |
| PyMongo | Latest | MongoDB Python driver |

### Network Configuration

| Device | IP Address | Port | Service |
|--------|-----------|------|---------|
| Jetson Orin | - | - | Face Recognition Client |
| Raspberry Pi | 192.168.196.195 | 1883 | MQTT Broker + LED Controller |
| RTSP Camera | 192.168.196.93 | 554 | Video Stream |
| MongoDB Server | 192.168.196.182 | 27017 | Database |

---

## Setup Guide

## 1. Jetson Orin (Face Recognition)

### 1.1 System Update

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git python3-dev python3-pip cmake build-essential
sudo apt install -y libopencv-dev python3-opencv
```

### 1.2 Install ONNX Runtime GPU

**Clone dan Build:**

```bash
cd /projectface/PPM-Project
git clone --recursive https://github.com/microsoft/onnxruntime.git
cd onnxruntime
git checkout v1.14.1

# Build dengan CUDA + TensorRT
./build.sh --config Release \
  --build_wheel \
  --use_tensorrt \
  --use_cuda \
  --cuda_home /usr/local/cuda \
  --cudnn_home /usr/lib/aarch64-linux-gnu \
  --tensorrt_home /usr/lib/aarch64-linux-gnu \
  --skip_tests \
```

**Install:**

```bash
pip3 install build/Linux/Release/dist/onnxruntime_gpu-1.14.1-cp38-cp38-linux_aarch64.whl
```

**Verifikasi GPU:**

```bash
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

✅ **Expected Output:**
```
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

### 1.3 Install Dependencies

```bash
pip3 install insightface opencv-python paho-mqtt numpy pillow
```

### 1.4 Persiapan Face Database

#### 1.4.1 Struktur Dataset

Buat folder `captures` dengan struktur berikut:

```
/projectface/PPM-Project/
├── captures/
│   ├── John_Doe/
│   │   ├── photo1.jpg
│   │   ├── photo2.jpg
│   │   └── photo3.jpg
│   ├── Jane_Smith/
│   │   ├── img1.png
│   │   ├── img2.png
│   │   └── img3.png
│   └── Ahmad_Rizki/
│       ├── face1.jpg
│       └── face2.jpg
└── train2.py
```

**Cara Menyiapkan Dataset:**

1. Buat folder untuk setiap karyawan di dalam `captures/`
2. Nama folder = nama yang akan ditampilkan di LED
3. Masukkan 3-5 foto wajah per orang
4. Foto harus jelas, pencahayaan baik, berbagai sudut

**Tips Foto Dataset:**
- ✅ Resolusi minimal 640x640 pixels
- ✅ Wajah menghadap kamera (frontal)
- ✅ Pencahayaan cukup (tidak gelap/silau)
- ✅ Variasi ekspresi & sudut ringan
- ✅ Format: JPG, PNG, JPEG
- ❌ Hindari foto blur/goyang
- ❌ Hindari wajah tertutup (masker, kacamata hitam)
- ❌ Hindari foto dari jauh (wajah terlalu kecil)

#### 1.4.2 Training Script

**File: `train2.py`**

```python
import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis

# --------------------------------------
# KONFIGURASI
# --------------------------------------
DATASET_DIR = "captures"
OUTPUT_FILE = "face_embeddings.pkl"   # File hasil training

# --------------------------------------
# INISIALISASI MODEL
# --------------------------------------
print("🔍 Memuat model InsightFace (GPU mode)...")
app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

# --------------------------------------
# FUNGSI MEMBUAT EMBEDDING
# --------------------------------------
def get_face_embedding(image_path):
    """
    Extract face embedding from image
    Returns: embedding vector or None
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Gagal baca gambar: {image_path}")
        return None
    
    faces = app.get(image)
    if len(faces) == 0:
        print(f"🚫 Tidak ada wajah terdeteksi: {image_path}")
        return None
    
    # Jika ada multiple faces, ambil yang terbesar
    if len(faces) > 1:
        faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
        print(f"   ⚠️ Terdeteksi {len(faces)} wajah, mengambil yang terbesar")
    
    return faces[0].normed_embedding

# --------------------------------------
# LOOP SEMUA DATASET
# --------------------------------------
database = {}
total_images = 0
total_success = 0

for person_name in sorted(os.listdir(DATASET_DIR)):
    person_dir = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue

    print(f"\n👤 Memproses {person_name}...")
    embeddings = []
    
    for filename in sorted(os.listdir(person_dir)):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        total_images += 1
        image_path = os.path.join(person_dir, filename)
        print(f"   📷 {filename}...", end=" ")
        
        embedding = get_face_embedding(image_path)
        if embedding is not None:
            embeddings.append(embedding)
            total_success += 1
            print("✅")
        else:
            print("❌")

    if embeddings:
        # Rata-rata embedding dari semua foto
        mean_embedding = np.mean(embeddings, axis=0)
        database[person_name] = mean_embedding
        print(f"✅ {len(embeddings)} wajah berhasil diproses untuk {person_name}")
    else:
        print(f"❌ Tidak ada embedding valid untuk {person_name}")

# --------------------------------------
# SIMPAN DATABASE EMBEDDING
# --------------------------------------
if database:
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(database, f)
    
    print("\n" + "="*60)
    print("🎯 TRAINING SELESAI!")
    print("="*60)
    print(f"📊 Total gambar diproses: {total_images}")
    print(f"✅ Berhasil: {total_success}")
    print(f"❌ Gagal: {total_images - total_success}")
    print(f"👥 Total orang: {len(database)}")
    print(f"💾 Database disimpan ke: {OUTPUT_FILE}")
    print("\nDaftar orang dalam database:")
    for i, name in enumerate(sorted(database.keys()), 1):
        print(f"  {i}. {name}")
else:
    print("\n❌ ERROR: Tidak ada data untuk disimpan!")
    print("Pastikan folder captures/ berisi foto-foto wajah yang valid.")
```

#### 1.4.3 Menjalankan Training

**Step-by-step:**

```bash
cd /projectface/PPM-Project

# 1. Buat folder captures jika belum ada
mkdir -p captures

# 2. Buat folder untuk setiap karyawan dengan Code python capturewajah2.py
**File: `capturewajah2.py`**
import cv2
import os

# =============================
# KONFIGURASI
# =============================
RTSP_URL = "rtsp://admin:BABKQU@192.168.196.93:554/h264/ch1/main/av_stream"  # Ganti sesuai RTSP kamu
SAVE_DIR = "captures"

# =============================
# PROGRAM UTAMA
# =============================

# Input nama folder dari user
folder_name = input("Masukkan nama folder untuk menyimpan foto: ").strip()
target_folder = os.path.join(SAVE_DIR, folder_name)

# Buat folder jika belum ada
os.makedirs(target_folder, exist_ok=True)
print(f"[INFO] Folder '{target_folder}' siap digunakan.")

# Hitung berapa file "image (x).jpg" yang sudah ada
existing_files = [f for f in os.listdir(target_folder) if f.startswith("image (") and f.endswith(").jpg")]

# Cari nomor terakhir
next_num = 1
if existing_files:
    numbers = []
    for fname in existing_files:
        try:
            num = int(fname.split("(")[1].split(")")[0])
            numbers.append(num)
        except:
            pass
    if numbers:
        next_num = max(numbers) + 1

# Buka RTSP camera
print("[INFO] Menghubungkan ke kamera RTSP...")
cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("[ERROR] Gagal membuka stream RTSP. Cek koneksi atau URL RTSP kamu.")
    exit()

print("[INFO] Tekan ENTER untuk ambil foto, Q untuk keluar.")

# =============================
# POSISI GARIS DETEKTOR
# =============================
line_y1 = 200
line_y2 = 400
line_color = (0, 255, 255)  # kuning

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Gagal membaca frame dari stream.")
        break

    # Gambar garis detektor
    height, width = frame.shape[:2]
    cv2.line(frame, (0, line_y1), (width, line_y1), line_color, 2)
    cv2.line(frame, (0, line_y2), (width, line_y2), line_color, 2)

    # Tampilkan hasil
    cv2.imshow("RTSP Stream (Tekan ENTER untuk ambil foto, Q untuk keluar)", frame)

    key = cv2.waitKey(1) & 0xFF

    # ENTER = ambil gambar
    if key == 13:  # ENTER
        filename = os.path.join(target_folder, f"image ({next_num}).jpg")
        cv2.imwrite(filename, frame)
        print(f"[INFO] Gambar tersimpan: {filename}")
        next_num += 1

    # Q = keluar
    elif key == ord('q'):
        print("[INFO] Program selesai. Menutup kamera...")
        break

# Tutup semua
cap.release()
cv2.destroyAllWindows()
print("[INFO] Semua jendela ditutup. Selesai.")


```
# 3. Buat folder untuk setiap karyawan
mkdir -p captures/John_Doe
mkdir -p captures/Jane_Smith
mkdir -p captures/Ahmad_Rizki


# 4. Copy foto-foto ke folder masing-masing
# (transfer via scp, USB, atau method lain)

# 5. Verifikasi struktur folder
tree captures/
# atau
ls -R captures/

# 6. Jalankan training script
python3 train2.py
```

**Expected Output:**

```
🔍 Memuat model InsightFace (GPU mode)...

👤 Memproses Ahmad_Rizki...
   📷 face1.jpg... ✅
   📷 face2.jpg... ✅
✅ 2 wajah berhasil diproses untuk Ahmad_Rizki

👤 Memproses Jane_Smith...
   📷 img1.png... ✅
   📷 img2.png... ✅
   📷 img3.png... ✅
✅ 3 wajah berhasil diproses untuk Jane_Smith

👤 Memproses John_Doe...
   📷 photo1.jpg... ✅
   📷 photo2.jpg... ✅
   📷 photo3.jpg... ❌
✅ 2 wajah berhasil diproses untuk John_Doe

============================================================
🎯 TRAINING SELESAI!
============================================================
📊 Total gambar diproses: 7
✅ Berhasil: 6
❌ Gagal: 1
👥 Total orang: 3
💾 Database disimpan ke: face_embeddings.pkl

Daftar orang dalam database:
  1. Ahmad_Rizki
  2. Jane_Smith
  3. John_Doe
```

#### 1.4.4 Verifikasi Database

**Script: `verify_embeddings.py`**

```python
import pickle
import numpy as np

# Load database
with open("face_embeddings.pkl", "rb") as f:
    database = pickle.load(f)

print("📊 Database Face Embeddings")
print("="*60)
print(f"Total orang dalam database: {len(database)}")
print()

for name, embedding in database.items():
    print(f"👤 {name}")
    print(f"   • Embedding shape: {embedding.shape}")
    print(f"   • Embedding type: {type(embedding)}")
    print(f"   • Mean value: {np.mean(embedding):.4f}")
    print(f"   • Std deviation: {np.std(embedding):.4f}")
    print()
```
### 1.5 Konfigurasi Face Recognition Script

**File: `FINAL3.py`**

**Parameter Utama yang Perlu Disesuaikan:**

```python
# ===== RTSP CAMERA =====
RTSP_URL = "rtsp://admin:BABKQU@192.168.196.93:554/h264/ch1/main/av_stream"

# ===== MQTT BROKER =====
MQTT_BROKER = "192.168.196.195"
MQTT_PORT = 1883
MQTT_TOPIC = "led/display"
MQTT_RETRY_INTERVAL = 5  # detik

# ===== DETECTION SETTINGS =====
SIMILARITY_THRESHOLD = 0.6      # Threshold pengenalan wajah (0.0 - 1.0)
LINE_TOP = 200                  # Batas atas detection zone (pixels)
LINE_BOTTOM = 400               # Batas bawah detection zone (pixels)

# ===== TIMING SETTINGS =====
DETECTION_COOLDOWN = 8          # Cooldown antar-pesan MQTT (detik)
HOLD_TIME = 5                   # Durasi hold wajah terdeteksi (detik)
TAMU_STABILITY_FRAMES = 3       # Frame validasi "Tamu" saat ada known face

# ===== FPS & RECONNECT =====
FPS_INTERVAL = 2.0              # Interval update FPS
RECONNECT_DELAY = 3             # Initial reconnect delay (detik)
RECONNECT_BACKOFF_MAX = 60      # Max backoff delay (detik)

# ===== LOG FILE =====
LOG_FILE = "face_log.txt"
.........
```

**Fitur-fitur Penting:**

1. **Auto-Reconnect RTSP**
   - Reconnect otomatis jika stream terputus
   - Exponential backoff untuk retry
   - Deteksi frame hitam/kosong

2. **Hold Logic**
   - Wajah tetap terdeteksi selama `HOLD_TIME` detik setelah hilang dari frame
   - Mencegah flicker saat wajah terhalangi sesaat

3. **Tamu Stability**
   - "Tamu" harus muncul `TAMU_STABILITY_FRAMES` frame berturut-turut untuk valid
   - Hanya berlaku jika ada known face aktif
   - Mengurangi false positive

4. **MQTT Auto-Reconnect**
   - Thread terpisah untuk menjaga koneksi MQTT
   - Retry otomatis setiap `MQTT_RETRY_INTERVAL` detik

5. **Multi-Face Detection**
   - Bisa mendeteksi beberapa wajah sekaligus
   - Menggabungkan nama dengan koma: "John, Jane, Tamu"

### 1.6 Running the Service

**Manual Testing:**

```bash
cd /projectface/PPM-Project
python3 face_recognition_jetson.py
```

**Output yang Diharapkan:**

```
Dataset wajah terload: ['John Doe', 'Jane Smith', ...]
Memuat model InsightFace...
Model dijalankan di GPU.
[10:30:15] MQTT: Terhubung ke broker.
[10:30:20] Terhubung ke kamera RTSP.
Streaming dimulai tanpa GUI, dengan Line Detector, Hold Logic, dan Auto-Reconnect aktif.
FPS: 28.50
```

**Auto-start dengan systemd:**

```bash
sudo nano /etc/systemd/system/face-recognition.service
```

**Service File:**

```ini
[Unit]
Description=Face Recognition Service
After=network.target

[Service]
Type=simple
User=visuil
WorkingDirectory=/projectface/PPM-Project
ExecStart=/usr/bin/python3 /projectface/PPM-Project/FINAL3.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
Environment="CUDA_HOME=/usr/local/cuda"
Environment="LD_LIBRARY_PATH=/usr/local/cuda/lib64"

[Install]
WantedBy=multi-user.target
```

**Enable & Start:**

```bash
sudo systemctl daemon-reload
sudo systemctl enable face-recognition.service
sudo systemctl start face-recognition.service
sudo systemctl status face-recognition.service
```

### 1.7 Monitoring

**View Logs:**

```bash
# Real-time log file
tail -f /projectface/PPM-Project/face_log.txt

# Systemd logs
journalctl -u face-recognition.service -f

# Filter FPS info
tail -f face_log.txt | grep FPS
```

**Log Format:**

```
[10:30:25] MQTT terkirim: Selamat datang John Doe
[10:30:33] MQTT terkirim: Selamat datang Jane Smith, Tamu
```

### 1.8 Performance Optimization

**Max Performance Mode:**

```bash
# Enable maximum performance
sudo nvpmodel -m 0
sudo jetson_clocks

# Monitor GPU usage
sudo tegrastats
```

**Environment Variables:**

Tambahkan ke `~/.bashrc`:

```bash
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
```

---

## 2. Raspberry Pi (LED Controller)

### 2.1 Install rpi-rgb-led-matrix

```bash
sudo apt update
sudo apt install -y git build-essential python3-dev python3-pillow

# Clone repository
git clone https://github.com/hzeller/rpi-rgb-led-matrix.git
cd rpi-rgb-led-matrix

# Build Python bindings
make build-python
sudo make install-python

# Test installation
python3 -c "from rgbmatrix import RGBMatrix, RGBMatrixOptions; print('✅ Installation OK')"
```

### 2.2 Matrix Configuration

**Hardware Setup: 4 Panel 64x64 (Total 256x64)**

```python
options = RGBMatrixOptions()
options.rows = 64              # Tinggi 1 panel
options.cols = 128             # Lebar base (2 panel)
options.chain_length = 2       # 2 panel horizontal
options.parallel = 1           # 1 chain
options.hardware_mapping = 'regular'
options.gpio_slowdown = 4      # Sesuaikan dengan Pi model
options.multiplexing = 0
options.row_address_type = 3
options.scan_mode = 0
options.brightness = 70        # 0-100
options.drop_privileges = False
```

**Hasil:** Display 256x64 pixels (4 panel 64x64)

### 2.3 GPIO Wiring Reference

#### Shared Signals (All Panels)

| GPIO Pin | Signal | Function |
|----------|--------|----------|
| GPIO 11 | CLK | Clock signal |
| GPIO 7 | LAT/STB | Latch/Strobe |
| GPIO 12 | OE | Output Enable |
| GPIO 15 | A | Row select A |
| GPIO 16 | B | Row select B |
| GPIO 18 | C | Row select C |
| GPIO 22 | D | Row select D |
| GND | GND | Ground |

#### Data Lines (Chain 1)

| GPIO Pin | Signal | Function |
|----------|--------|----------|
| GPIO 23 | R1 | Red (upper) |
| GPIO 13 | G1 | Green (upper) |
| GPIO 26 | B1 | Blue (upper) |
| GPIO 24 | R2 | Red (lower) |
| GPIO 21 | G2 | Green (lower) |
| GPIO 19 | B2 | Blue (lower) |

⚠️ **CRITICAL - Power Supply:**
- LED Matrix menggunakan power supply terpisah 5V 10A minimum
- **WAJIB** hubungkan GND power supply ke GND Raspberry Pi (common ground)
- Jangan ambil 5V dari Pi

### 2.4 Install MQTT Broker

```bash
# Install Mosquitto MQTT Broker
sudo apt install -y mosquitto mosquitto-clients

# Enable & Start
sudo systemctl enable mosquitto
sudo systemctl start mosquitto

# Install Python MQTT client
pip3 install paho-mqtt
```

**Test MQTT:**

```bash
# Terminal 1 - Subscribe
mosquitto_sub -h localhost -t "led/display"

# Terminal 2 - Publish
mosquitto_pub -h localhost -t "led/display" -m "Test Message"
```

### 2.5 Font Setup

**Font Locations:**

```bash
/home/PINDAD/rpi-rgb-led-matrix/fonts/7x13B.bdf   # Small
/home/PINDAD/rpi-rgb-led-matrix/fonts/7x14B.bdf   # Medium
/home/PINDAD/rpi-rgb-led-matrix/fonts/10x20.bdf   # Large
```

### 2.6 Emoticon Image

**File: `/home/PINDAD/rpi-rgb-led-matrix/examples-api-use/emot.png`**

- Format: PNG
- Ukuran: 20x20 pixels (akan di-resize)
- RGB mode

### 2.7 LED Controller Configuration

**File: `led_display_raspberry.py`**

**Fitur-fitur:**

1. **Mode Normal (Jam & Tanggal)**
   - Tampilan jam realtime (HH:MM:SS)
   - Tanggal dalam bahasa Indonesia
   - Scrolling text "DEPARTEMEN LTI TUREN-BANDUNG"

2. **Mode MQTT (Nama Tamu)**
   - Header: "SELAMAT DATANG" (3 detik)
   - Nama tamu (3 detik)
   - "SILAHKAN MASUK" (3 detik)
   - Total durasi: 6 detik
   - Emoticon di samping nama

3. **Mode Merah Putih (10:00 - 10:05)**
   - Background merah (atas) dan putih (bawah)
   - Tidak ada clear canvas untuk menghindari flicker
   - Pre-generated background image

**MQTT Message Format:**

```python
# Single person
"Selamat datang John Doe"

# Multiple people
"Selamat datang John Doe, Jane Smith, Tamu"
```

**Colors Defined:**

```python
white = graphics.Color(255, 255, 255)
blue = graphics.Color(0, 150, 255)
yellow = graphics.Color(255, 255, 0)
orange = graphics.Color(255, 165, 0)
```

### 2.8 Running LED Controller

**Manual Testing:**

```bash
cd /home/PINDAD/rpi-rgb-led-matrix/examples-api-use
sudo python3 led_display_raspberry.py
```

**Auto-start dengan systemd:**

```bash
sudo nano /etc/systemd/system/led-display.service
```

**Service File:**

```ini
[Unit]
Description=LED Display Controller
After=network.target mosquitto.service

[Service]
Type=simple
User=root
WorkingDirectory=/home/PINDAD/rpi-rgb-led-matrix/examples-api-use
ExecStart=/usr/bin/python3 led_display_raspberry.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

**Enable & Start:**

```bash
sudo systemctl daemon-reload
sudo systemctl enable led-display.service
sudo systemctl start led-display.service
sudo systemctl status led-display.service
```

### 2.9 Testing LED Display

**Test dengan runtext:**

```bash
cd /home/PINDAD/rpi-rgb-led-matrix/examples-api-use
sudo ./scrolling-text-example   -f ../fonts/7x13.bdf   -C 255,255,255   -B 0,0,0   --led-rows=64   --led-cols=128   --led-chain=1   --led-parallel=1   --led-gpio-mapping=regular   --led-slowdown-gpio=4   --led-multiplexing=0   --led-row-addr-type=3   --led-scan-mode=0   "HELLO WORLD! WELCOME TO LED MATRIX"--led-chain=2
```
**Test dengan jam:**
```bash
sudo ./clock     -f ../fonts/7x13.bdf     -d "%A"     -d "%H:%M:%S"     -x 120     -y 20     --led-rows=64     --led-cols=128     --led-chain=2     --led-parallel=1     --led-gpio-mapping=regular     --led-slowdown-gpio=4     --led-multiplexing=0     --led-row-addr-type=3     --led-scan-mode=0
```
**Test dengan Demo:**
```bash
sudo ./demo -D 1 gambar.ppm --led-rows=64 --led-cols=128 --led-chain=1 --led-parallel=1   --led-gpio-mapping=regular --led-slowdown-gpio=4   --led-multiplexing=0 --led-row-addr-type=3 --led-scan-mode=0
```

**Test MQTT Message:**

```bash
# Dari terminal lain
mosquitto_pub -h 192.168.196.195 -t "led/display" -m "Selamat datang Test User"
```

---

## 3. MongoDB (Database)

### 3.1 Install MongoDB Server

**On MongoDB Server (192.168.196.182):**

```bash
# Install MongoDB
sudo apt update
sudo apt install -y mongodb

# Enable & Start
sudo systemctl enable mongodb
sudo systemctl start mongodb
sudo systemctl status mongodb
```

**Configure Network Access:**

```bash
sudo nano /etc/mongodb.conf
```

**Edit configuration:**

```yaml
# Allow connections from network
bind_ip = 0.0.0.0
port = 27017
```

**Restart:**

```bash
sudo systemctl restart mongodb
```

**Test Connection dari Jetson:**

```bash
pip3 install pymongo

python3 << EOF
from pymongo import MongoClient
client = MongoClient('mongodb://192.168.196.182:27017')
print('✅ MongoDB OK:', client.server_info()['version'])
EOF
```

### 3.2 Database Structure

**Database:** `face_recog_db`

**Collection:** `logs`

**Document Structure:**

```json
{
  "_id": ObjectId("..."),
  "file": "face_log.txt",
  "message": "[10:30:25] MQTT terkirim: Selamat datang John Doe",
  "timestamp": "2025-10-28 10:30:25"
}
```

### 3.3 Log Monitor Script

**File: `log_monitor_mongodb.py`**

**Konfigurasi:**

```python
MONGO_URI = "mongodb://192.168.196.182:27017"
DB_NAME = "face_recog_db"
COLLECTION_NAME = "logs"

# File log yang dimonitor
LOG_FILES = [
    "/projectface/PPM-Project/face_log.txt"
]
```

**Cara Kerja:**

1. Monitor `face_log.txt` setiap 10 detik
2. Baca baris baru sejak posisi terakhir
3. Upload ke MongoDB dengan timestamp
4. Track posisi file untuk iterasi berikutnya

**Running:**

```bash
cd /projectface/PPM-Project
python3 log_monitor_mongodb.py
```

**Output:**

```
🔄 Monitoring log dan upload ke MongoDB...
[+] Inserted log: [10:30:25] MQTT terkirim: Selamat datang John Doe
[+] Inserted log: [10:30:33] MQTT terkirim: Selamat datang Jane Smith
```

### 3.4 Setup as Service

```bash
sudo nano /etc/systemd/system/log-monitor.service
```

**Service File:**

```ini
[Unit]
Description=Log Monitor to MongoDB
After=network.target

[Service]
Type=simple
User=visuil
WorkingDirectory=/projectface/PPM-Project
ExecStart=/usr/bin/python3 log_monitor_mongodb.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

**Enable & Start:**

```bash
sudo systemctl daemon-reload
sudo systemctl enable log-monitor.service
sudo systemctl start log-monitor.service
sudo systemctl status log-monitor.service
```

### 3.5 Query MongoDB

**MongoDB Shell:**

```bash
mongo 192.168.196.182:27017
```

**Query Examples:**

```javascript
// Gunakan database
use face_recog_db

// Lihat semua logs (terbatas 10)
db.logs.find().limit(10).pretty()

// Logs terbaru
db.logs.find().sort({_id: -1}).limit(10)

// Filter by message content
db.logs.find({message: /MQTT terkirim/})

// Count total logs
db.logs.count()

// Logs hari ini
db.logs.find({
    timestamp: {
        $regex: "^2025-10-28"
    }
})

// Delete logs lebih dari 30 hari
var cutoff = new Date();
cutoff.setDate(cutoff.getDate() - 30);
db.logs.deleteMany({
    timestamp: {
        $lt: cutoff.toISOString().split('T')[0]
    }
})
```

**Python Query:**

```python
from pymongo import MongoClient
from datetime import datetime, timedelta

client = MongoClient("mongodb://192.168.196.182:27017")
db = client["face_recog_db"]
collection = db["logs"]

# Get logs dari 1 jam terakhir
one_hour_ago = (datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
recent_logs = collection.find({"timestamp": {"$gte": one_hour_ago}})

for log in recent_logs:
    print(f"[{log['timestamp']}] {log['message']}")

# Count logs per file
pipeline = [
    {"$group": {"_id": "$file", "count": {"$sum": 1}}}
]
result = collection.aggregate(pipeline)
for doc in result:
    print(f"{doc['_id']}: {doc['count']} logs")
```

**📅 Last Updated:** 28 Oktober 2025  
**Version:** 1.1  
**Maintained by:** DEPARTEMEN LTI TUREN-BANDUNG