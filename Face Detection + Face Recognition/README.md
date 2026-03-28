# Face Detection + Face Recognition

ระบบ Face Detection และ Face Recognition ที่รันบน **Qualcomm Snapdragon 6490** ผ่าน SNPE (Snapdragon Neural Processing Engine)

---

## โมเดลที่ใช้

| โมเดล | ไฟล์ | หน้าที่ |
|-------|------|--------|
| **SCRFD 2.5G** | `scrfd_quantized_6490.dlc` | ตรวจจับใบหน้า + 5 facial landmarks |
| **ArcFace ResNet100** | `arcface_quantized_6490.dlc` | แปลงใบหน้าเป็น embedding vector 512 มิติ |

ทั้งสองโมเดลถูก quantize มาเพื่อรันบน **DSP (Hexagon)** ของ Snapdragon 6490 ใน BURST mode

---

## โครงสร้าง Project

```
Face Detection + Face Recognition/
├── web.py          # Web server (Flask) + core logic ทั้งหมด
├── camera.py       # โหมดแสดงผลผ่านหน้าต่าง OpenCV
├── README.md
│
datasets/           # รูปภาพสำหรับ pre-enroll (แยก folder ตามชื่อคน)
│   ├── rooney/
│   │   ├── rooney1.jpg
│   │   └── rooney2.jpg
│   └── <ชื่อคน>/
│       └── *.jpg
│
face_database/      # สร้างอัตโนมัติตอนรัน — เก็บ embeddings และ metadata
    ├── embeddings.pkl
    └── metadata.json
```

---

## การเตรียม Dataset

สร้าง sub-folder ชื่อเดียวกับชื่อคนใน `datasets/` แล้วใส่รูปภาพ:

```
datasets/
├── rooney/
│   ├── rooney1.jpg
│   └── rooney2.jpg
└── titi/
    ├── 001.jpg
    └── 002.jpg
```

- ใส่รูปได้หลายใบต่อคน — ระบบจะเฉลี่ย embedding ทุกรูป เพื่อให้จำได้แม่นขึ้น
- รองรับนามสกุล: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`
- แนะนำอย่างน้อย 2–5 รูปต่อคน

---

## วิธีรัน

### 1. Web Mode (`web.py`) — แนะนำ

แสดงผลผ่านเบราว์เซอร์ เข้าถึงได้จากอุปกรณ์อื่นในเครือข่ายได้

```bash
python web.py \
  --datasets ../datasets \
  --scrfd-dlc "../SCRFD (Face Detection)/Model/scrfd_quantized_6490.dlc" \
  --arcface-dlc "../ArcFace (Face Recognition)/Model/arcface_quantized_6490.dlc"
```

แล้วเปิดเบราว์เซอร์ที่ `http://localhost:5000`

**Arguments ทั้งหมด:**

| Argument | Default | คำอธิบาย |
|----------|---------|----------|
| `--datasets` | `""` | Path ไปยัง datasets folder (ถ้าไม่ระบุจะข้ามขั้นตอน pre-enroll) |
| `--db-path` | `face_database` | Directory สำหรับเก็บ face database |
| `--camera` | `0` | Camera ID |
| `--scrfd-dlc` | *(ดูค่า default)* | Path ไปยัง SCRFD model |
| `--arcface-dlc` | *(ดูค่า default)* | Path ไปยัง ArcFace model |
| `--runtime` | `DSP` | Runtime: `DSP` หรือ `CPU` |
| `--threshold` | `0.4` | Similarity threshold สำหรับการจำแนกใบหน้า |
| `--skip-frames` | `1` | Process ทุกกี่เฟรม (ลดภาระ CPU/DSP) |
| `--host` | `0.0.0.0` | Web server host |
| `--port` | `5000` | Web server port |

---

### 2. Camera Mode (`camera.py`) — local window

แสดงผลผ่านหน้าต่าง OpenCV บนเครื่องโดยตรง

```bash
python camera.py \
  --db-path face_database \
  --scrfd-dlc "../SCRFD (Face Detection)/Model/scrfd_quantized_6490.dlc" \
  --arcface-dlc "../ArcFace (Face Recognition)/Model/arcface_quantized_6490.dlc"
```

> **หมายเหตุ:** `camera.py` ไม่มี `--datasets` argument — ต้องรัน `web.py` ก่อนเพื่อ build database แล้ว `camera.py` จะอ่าน database จาก `--db-path` เดียวกัน

**Controls:**
- `q` — ออกจากโปรแกรม

---

## การทำงานของระบบ

```
ตอน Startup
  └─ โหลด datasets/ → SCRFD detect ใบหน้า → ArcFace สร้าง embedding
     → เฉลี่ย embedding ต่อคน → บันทึกลง face_database/

ตอนกล้องทำงาน (ทุกเฟรม)
  └─ จับภาพจากกล้อง
     └─ SCRFD → ตรวจจับใบหน้า + landmark
        └─ ArcFace → แปลงใบหน้าเป็น vector 512 มิติ
           └─ FaceDatabase.search() → เทียบ cosine similarity
              ├─ similarity ≥ threshold → แสดงชื่อ + % (กรอบเขียว)
              └─ ไม่ตรงใคร → "Unknown / Not in DB" (กรอบแดง)
```

---

## สิ่งที่แก้ไขจาก Version เดิม

### เปลี่ยน Logic การ Enroll ใบหน้า

**เดิม:** ระบบให้ user กรอกชื่อ real-time ผ่าน modal บนหน้าเว็บทุกครั้งที่เจอหน้าใหม่

**ใหม่:** ระบบโหลด dataset ที่เตรียมไว้ล่วงหน้าตอน startup — กล้องรู้จักทุกคนได้ทันทีโดยไม่ต้อง input ใดๆ

### สิ่งที่ถูกลบออก
- Modal dialog "Enroll New Person" (HTML + CSS + JavaScript)
- ปุ่ม "Enroll" ในการ์ดแสดงใบหน้า
- `/enroll` Flask API route
- ฟังก์ชัน `openEnrollModal()`, `closeModal()`, `enrollForm.onsubmit`

### สิ่งที่เพิ่มเข้ามา
- ฟังก์ชัน `build_database_from_folder()` — อ่านรูปจาก datasets folder, detect ใบหน้าด้วย SCRFD, สร้าง embedding ด้วย ArcFace, เฉลี่ยต่อคน แล้วบันทึกลง database
- `--datasets` argument ใน `web.py`
- ใบหน้าที่ไม่รู้จักแสดงว่า **"Not in DB"** แทนปุ่ม Enroll
