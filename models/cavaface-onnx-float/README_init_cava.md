# CavaFace — Face Recognition Setup Guide

ระบบจำหน้าบุคคลโดยใช้ **MediaPipe BlazeFace** (ตรวจจับหน้า) + **CavaFace IR_SE_100** (จำหน้า)
ทำงานด้วย ONNX Runtime ไม่ต้องมี PyTorch หรือ GPU

---

## สิ่งที่ต้องมี

### ไฟล์ในโฟลเดอร์นี้

| ไฟล์ | หน้าที่ |
|------|---------|
| `FaceDetector.onnx` + `FaceDetector.data` | ตรวจจับหน้าและ landmark |
| `cavaface.onnx` + `cavaface.data` | สร้าง face embedding 512 มิติ |
| `build_database.py` | สร้าง database จากรูปภาพ |
| `recognize.py` | จำหน้าจากรูปภาพ หรือ live camera |

### Dependencies

```bash
pip install onnxruntime opencv-python numpy
```

> **หมายเหตุ:** บน Qualcomm QCS6490 (QC Linux / aarch64) ให้ใช้ `pip3` แทน `pip`

---

## Pipeline Overview

```
รูปภาพ input
     │
     ▼
FaceDetector.onnx          ← ตรวจจับหน้า + 5 landmark (ตา, จมูก, ปาก)
     │
     │  affine alignment   ← หมุน/ปรับให้ตาอยู่ตำแหน่งมาตรฐาน
     ▼
หน้า 112×112 (aligned)
     │
     ▼
cavaface.onnx              ← แปลงเป็น embedding 512 มิติ
     │
     ▼
cosine similarity          ← เปรียบกับ database
     │
     ▼
ชื่อบุคคล + ค่าความเหมือน
```

---

## Step 1 — เตรียมรูปภาพ

สร้างโฟลเดอร์ `photos/` แล้วแบ่งย่อยตามชื่อบุคคล:

```
photos/
├── Alice/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── 003.jpg
├── Bob/
│   ├── 001.jpg
│   └── 002.jpg
└── ...
```

**คำแนะนำสำหรับรูปภาพ:**
- ใช้อย่างน้อย **3–5 รูปต่อคน** เพื่อความแม่นยำสูงขึ้น
- รูปควรเห็นหน้าชัดเจน แสงสว่างพอ
- ใส่รูปหลายมุม (ตรง, เอียงเล็กน้อย) เพื่อ coverage ที่ดีขึ้น
- รองรับไฟล์: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

---

## Step 2 — สร้าง Face Database

```bash
python3 build_database.py --cavaface cavaface.onnx
```

**ผลลัพธ์ที่ควรได้:**
```
Loading models …
  [Alice] 001.jpg ✓
  [Alice] 002.jpg ✓
  [Alice] 003.jpg ✓
  → Alice: 3 photo(s) enrolled

  [Bob] 001.jpg ✓
  [Bob] 002.jpg ✓
  → Bob: 2 photo(s) enrolled

Database saved → face_db.npz
Enrolled 2 person(s): ['Alice', 'Bob']
```

ถ้าเห็น `no face in xxx.jpg — skipped` แสดงว่า model ตรวจไม่เจอหน้าในรูปนั้น ให้เปลี่ยนรูป

**Options เพิ่มเติม:**
```bash
python3 build_database.py --photos my_photos/ --db my_db.npz --cavaface cavaface.onnx
```

| Option | Default | คำอธิบาย |
|--------|---------|-----------|
| `--photos` | `photos/` | โฟลเดอร์รูปภาพ |
| `--db` | `face_db.npz` | ชื่อไฟล์ database output |
| `--cavaface` | `CavaFace.onnx` | path ของ CavaFace model |
| `--detector` | `FaceDetector.onnx` | path ของ FaceDetector model |

---

## Step 3 — จำหน้า

`recognize.py` รองรับ 2 mode โดยใช้ `--image` หรือ `--camera` (เลือกได้อย่างเดียว)

### Mode A: รูปภาพ (Static Image)

```bash
python3 recognize.py --image test.jpg --cavaface cavaface.onnx
```

**ผลลัพธ์ที่ควรได้:**
```
Database loaded: ['Alice', 'Bob']
  Alice: 0.821
  Bob:   0.134

Result : Alice  (similarity=0.821)
```

### Mode B: Live Camera

```bash
# Mac (built-in camera)
python3 recognize.py --camera 0 --cavaface cavaface.onnx

# Qualcomm QCS6490 (USB camera)
python3 recognize.py --camera 1 --cavaface cavaface.onnx
```

**สิ่งที่จะเห็นบนหน้าจอ:**
```
┌──────────────────────────────┐
│   ┌──────────────────┐       │
│   │ Alice (0.87)     │  ← กล่องเขียว = จำได้
│   └──────────────────┘       │
│                              │
│   ┌──────────────────┐       │
│   │ Unknown          │  ← กล่องแดง  = ไม่รู้จัก
│   └──────────────────┘       │
└──────────────────────────────┘
กด ESC หรือ Q เพื่อออก
```

ถ้าได้ `Unknown` ให้ลองลด threshold:
```bash
python3 recognize.py --image test.jpg --cavaface cavaface.onnx --threshold 0.35
python3 recognize.py --camera 0   --cavaface cavaface.onnx --threshold 0.35
```

**Options ทั้งหมด:**

| Option | Default | คำอธิบาย |
|--------|---------|-----------|
| `--image <path>` | — | รูปที่ต้องการทดสอบ (static mode) |
| `--camera <id>` | — | Camera device ID (live mode) |
| `--db` | `face_db.npz` | ไฟล์ database จาก Step 2 |
| `--threshold` | `0.45` | ค่า cosine similarity ขั้นต่ำ (0–1) |
| `--cavaface` | `CavaFace.onnx` | path ของ CavaFace model |
| `--detector` | `FaceDetector.onnx` | path ของ FaceDetector model |

> `--image` และ `--camera` เลือกได้แค่อย่างเดียว

---

## ค่า Similarity แปลความหมายอย่างไร

| ค่า | ความหมาย |
|-----|-----------|
| > 0.7 | จำได้ดีมาก (รูปชัด, แสงดี) |
| 0.5 – 0.7 | จำได้ (รูปทั่วไป) |
| 0.45 – 0.5 | ขอบเขต (อาจถูกหรือผิด) |
| < 0.45 | ไม่จำได้ → แสดงผลเป็น Unknown |

---

## โครงสร้างไฟล์หลังรัน

```
cavaface-onnx-float/
├── FaceDetector.onnx       ← model ตรวจจับหน้า
├── FaceDetector.data       ← weights ของ FaceDetector
├── cavaface.onnx           ← model จำหน้า
├── cavaface.data           ← weights ของ CavaFace (250MB)
├── build_database.py       ← script สร้าง database
├── recognize.py            ← script จำหน้า
├── photos/                 ← รูปสำหรับ enroll
│   ├── Alice/
│   └── Bob/
└── face_db.npz             ← database (สร้างโดย build_database.py)
```

---

## Troubleshooting

**`No face detected` / รูปถูก skip ทั้งหมด**
- ตรวจสอบว่ารูปเห็นหน้าชัดเจน
- แสงสว่างพอ ไม่ถูกบัง
- ลอง threshold ต่ำลง: แก้ `SCORE_THRESHOLD = 0.6` ใน build_database.py

**`Result: Unknown` ทั้งที่เป็นคนเดิม**
- ลอง `--threshold 0.35`
- เพิ่มรูปใน database (หลายมุม/แสง)
- ลบ `face_db.npz` แล้ว build ใหม่

**`File doesn't exist: FaceDetector.onnx`**
- ตรวจสอบว่า run script จากโฟลเดอร์ `cavaface-onnx-float/`
- หรือใช้ `--detector /path/to/FaceDetector.onnx`

**`Cannot open camera X`**
- ลอง ID อื่น: `--camera 0`, `--camera 1`, `--camera 2`
- บน Qualcomm QCS6490 camera มักอยู่ที่ index 1 หรือ 2

**บน Qualcomm QCS6490 — `git lfs` ไม่มี**
```bash
# ติดตั้ง git-lfs ด้วย pip
pip3 install git-lfs
git lfs install
git pull
git lfs pull   # ดึง cavaface.data (250MB)
```
