#!/usr/bin/env python3
import time
import paho.mqtt.client as mqtt
from rgbmatrix import RGBMatrix, RGBMatrixOptions, graphics
import datetime
from PIL import Image

# --- Konfigurasi Matrix ---
options = RGBMatrixOptions()
options.rows = 64
options.cols = 128
options.chain_length = 2
options.parallel = 1
options.hardware_mapping = 'regular'
options.gpio_slowdown = 4
options.multiplexing = 0
options.row_address_type = 3
options.scan_mode = 0
options.brightness = 70
options.drop_privileges = False
matrix = RGBMatrix(options=options)

# --- Fonts ---
font_small = graphics.Font()
font_small.LoadFont("/home/PINDAD/rpi-rgb-led-matrix/fonts/7x13B.bdf")
font_medium = graphics.Font()
font_medium.LoadFont("/home/PINDAD/rpi-rgb-led-matrix/fonts/7x14B.bdf")
font_large = graphics.Font()
font_large.LoadFont("/home/PINDAD/rpi-rgb-led-matrix/fonts/10x20.bdf")

# --- Colors ---
white = graphics.Color(255, 255, 255)
blue = graphics.Color(0, 150, 255)
yellow = graphics.Color(255, 255, 0)
orange = graphics.Color(255, 165, 0)

# --- Variabel global MQTT ---
mqtt_message = None
message_start = 0
message_expire = 0
mqtt_connected = False

# --- Load Emoticon ---
try:
    emot = Image.open("/home/PINDAD/rpi-rgb-led-matrix/examples-api-use/emot.png")
    emot = emot.convert("RGB")
except Exception as e:
    print(f"[WARNING] Emoticon gagal dimuat: {e}")
    emot = None

# --- Callback MQTT ---
def on_connect(client, userdata, flags, rc):
    global mqtt_connected
    if rc == 0:
        mqtt_connected = True
        print("[INFO] Connected to MQTT broker")
        client.subscribe("led/display")
    else:
        mqtt_connected = False
        print(f"[WARNING] Failed to connect, return code {rc}")

def on_disconnect(client, userdata, rc):
    global mqtt_connected
    mqtt_connected = False
    print("[WARNING] MQTT broker tidak aktif, mencoba reconnect...")

def on_message(client, userdata, msg):
    global mqtt_message, message_start, message_expire
    mqtt_message = msg.payload.decode("utf-8")
    message_start = time.time()
    message_expire = message_start + 6

    print(f"[MESSAGE] Pesan diterima: {mqtt_message}")

# --- MQTT Client ---
client = mqtt.Client()
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_message = on_message
try:
    client.connect_async("192.168.196.22", 1883, 60)
except Exception as e:
    print(f"[WARNING] Gagal connect MQTT broker: {e}")
client.loop_start()

# --- Scroll teks ---
scroll_x = matrix.width
canvas = matrix.CreateFrameCanvas()

# --- Hari & Bulan Bahasa Indonesia (huruf besar) ---
hari_list = ["SENIN", "SELASA", "RABU", "KAMIS", "JUMAT", "SABTU", "MINGGU"]
bulan_list = ["JANUARI", "FEBRUARI", "MARET", "APRIL", "MEI", "JUNI",
              "JULI", "AGUSTUS", "SEPTEMBER", "OKTOBER", "NOVEMBER", "DESEMBER"]

# --- Pre-buat background merah-putih
red_white_bg = Image.new("RGB", (canvas.width, canvas.height))
for y in range(canvas.height // 2):
    for x in range(canvas.width):
        red_white_bg.putpixel((x, y), (255, 0, 0))
for y in range(canvas.height // 2, canvas.height):
    for x in range(canvas.width):
        red_white_bg.putpixel((x, y), (255, 255, 255))

# --- Loop utama ---
while True:
    now = datetime.datetime.now()
    jam = now.strftime("%H:%M:%S")
    hour = now.hour
    minute = now.minute
    hari_ind = hari_list[now.weekday()]
    bulan_ind = bulan_list[now.month - 1]
    hari_tgl = f"{hari_ind}, {now.day} {bulan_ind} {now.year}"

    # --- Mode merah putih ------
    if hour == 10 and minute < 3:
        # Jangan canvas.Clear(), langsung swap background merah-putih
        canvas.SetImage(red_white_bg)

    else:
        # Clear canvas hanya untuk mode normal (jam, tanggal, mqtt)
        canvas.Clear()

        # --- Jika ada pesan MQTT aktif ---
        if mqtt_message and time.time() < message_expire:
            elapsed = time.time() - message_start

            # Baris atas: SELAMAT DATANG
            text_w_header = graphics.DrawText(canvas, font_large, 0, -100, yellow, "SELAMAT DATANG")
            pos_x_header = (canvas.width - text_w_header) // 2
            graphics.DrawText(canvas, font_large, pos_x_header, 20, yellow, "SELAMAT DATANG")

            # Baris bawah: Nama (3 detik pertama) lalu SILAHKAN MASUK (3 detik berikutnya)
            if elapsed < 3:
                nama_text = mqtt_message
            else:
                nama_text = "SILAHKAN MASUK"

            text_w_nama = graphics.DrawText(canvas, font_large, 0, -100, white, nama_text)
            pos_x_nama = (canvas.width - text_w_nama) // 2
            graphics.DrawText(canvas, font_large, pos_x_nama, 50, white, nama_text)

            # Tambahkan emot sejajar dengan nama
            if emot:
                emot_resized = emot.resize((20, 20))
                emot_x = pos_x_nama + text_w_nama + 5
                emot_y = 30
                canvas.SetImage(emot_resized, emot_x, emot_y)

        else:
            mqtt_message = None
            # Tampilkan jam dan tanggal
            text_w_jam = graphics.DrawText(canvas, font_medium, 0, -100, white, jam)
            pos_x_jam = (canvas.width - text_w_jam) // 2
            graphics.DrawText(canvas, font_medium, pos_x_jam, 15, white, jam)

            text_w_date = graphics.DrawText(canvas, font_small, 0, -100, white, hari_tgl)
            pos_x_date = (canvas.width - text_w_date) // 2
            graphics.DrawText(canvas, font_small, pos_x_date, 30, white, hari_tgl)

            # Teks berjalan bawah
            scroll_text = "DEPARTEMEN LTI BANDUNG-TUREN"
            graphics.DrawText(canvas, font_medium, scroll_x, 55, blue, scroll_text)
            scroll_x -= 1
            if scroll_x + graphics.DrawText(canvas, font_medium, 0, -100, blue, scroll_text) < 0:
                scroll_x = canvas.width

    matrix.SwapOnVSync(canvas)
    time.sleep(0.05)
