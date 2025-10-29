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

