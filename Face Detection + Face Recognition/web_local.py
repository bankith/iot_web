"""  
web_local.py — Face Detection + Recognition (Mac / Local version)  
  
ไฟล์นี้เป็น version สำหรับทดสอบบน macOS หรือ Linux ทั่วไป  
ใช้ ONNX Runtime แทน SNPE — ไม่ต้องการ Qualcomm hardware  
  
Models:  
    FaceDetector.onnx   — BlazeFace face detection  
    cavaface.onnx       — CavaFace face recognition (512-dim embedding)  
  
Usage:  
    pip install flask onnxruntime opencv-python numpy  
  
    python web_local.py \\  
        --datasets   /path/to/datasets \\  
        --detector   /path/to/FaceDetector.onnx \\  
        --cavaface   /path/to/cavaface.onnx  
  
  
    ****** Command usage ******  
     python web_local.py --datasets ../datasets --detector ../models/cavaface-onnx-float/FaceDetector.onnx --cavaface ../models/cavaface-onnx-float/cavaface.onnx  
  
    Then open: http://localhost:5000  
"""  
  
import argparse  
import os  
import threading  
import time  
from pathlib import Path  
  
import cv2  
import numpy as np  
import onnxruntime as ort  
from flask import Flask, Response, jsonify, render_template_string  
  
# =============================================================================  
# Constants  
# =============================================================================  
  
DETECT_INPUT_HW   = (256, 256)  
CAVAFACE_INPUT_HW = (112, 112)  
SCORE_THRESHOLD   = 0.75  
NMS_IOU_THRESHOLD = 0.3  
IMG_EXTENSIONS    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}  
  
ARCFACE_DST = np.array([  
    [38.2946, 51.6963],  
    [73.5318, 51.5014],  
    [56.0252, 71.7366],  
    [41.5493, 92.3655],  
    [70.7299, 92.2041],  
], dtype=np.float32)  
  
  
# =============================================================================  
# Anchor / Decode helpers  
# =============================================================================  
  
def generate_anchors(input_size: int = 256) -> np.ndarray:  
    strides, anchors_per_cell = [16, 32], [2, 6]  
    rows = []  
    for stride, n in zip(strides, anchors_per_cell):  
        grid = input_size // stride  
        for y in range(grid):  
            for x in range(grid):  
                cx, cy = (x + 0.5) / grid, (y + 0.5) / grid  
                for _ in range(n):  
                    rows.append([cx, cy, 1.0, 1.0])  
    return np.array(rows, dtype=np.float32).reshape(-1, 2, 2)  
  
  
def resize_pad(img_rgb: np.ndarray, target_hw: tuple):  
    h, w = img_rgb.shape[:2]  
    th, tw = target_hw  
    scale = min(th / h, tw / w)  
    nh, nw = int(h * scale), int(w * scale)  
    resized = cv2.resize(img_rgb, (nw, nh))  
    pt, pl = (th - nh) // 2, (tw - nw) // 2  
    padded = np.zeros((th, tw, 3), dtype=np.uint8)  
    padded[pt:pt + nh, pl:pl + nw] = resized  
    tensor = (padded.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]  
    return tensor, scale, pt, pl  
  
  
def decode_boxes(raw: np.ndarray, anchors: np.ndarray, img_hw: tuple) -> np.ndarray:  
    H, W = img_hw  
    center = anchors[:, 0:1, :] * np.array([[W, H]], dtype=np.float32)  
    scale  = anchors[:, 1:2, :]  
    K = raw.shape[1]  
    mask = np.ones((K, 1), dtype=np.float32)  
    mask[1] = 0.0  
    return raw * scale + center * mask  
  
  
def box_iou(a: np.ndarray, b: np.ndarray) -> float:  
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])  
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])  
    inter  = max(0.0, x2 - x1) * max(0.0, y2 - y1)  
    union  = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter  
    return inter / union if union > 0 else 0.0  
  
  
def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list:  
    order = scores.argsort()[::-1].tolist()  
    keep  = []  
    while order:  
        i = order.pop(0)  
        keep.append(i)  
        order = [j for j in order if box_iou(boxes[i], boxes[j]) < iou_thr]  
    return keep  
  
  
# =============================================================================  
# Face Detection (BlazeFace ONNX)  
# =============================================================================  
  
def detect_faces(img_rgb: np.ndarray,  
                 det_sess: ort.InferenceSession,  
                 anchors: np.ndarray,  
                 score_threshold: float = SCORE_THRESHOLD) -> list:  
    inp, scale, pt, pl = resize_pad(img_rgb, DETECT_INPUT_HW)  
    name = det_sess.get_inputs()[0].name  
    c1, c2, s1, s2 = det_sess.run(None, {name: inp})  
  
    coords = np.concatenate([c1[0], c2[0]], axis=0).reshape(-1, 8, 2)  
    scores = np.concatenate([s1[0], s2[0]], axis=0).reshape(-1)  
    scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -100.0, 100.0)))  
  
    decoded = decode_boxes(coords, anchors, DETECT_INPUT_HW)  
  
    cx, cy = decoded[:, 0, 0], decoded[:, 0, 1]  
    bw, bh = decoded[:, 1, 0], decoded[:, 1, 1]  
    boxes  = np.stack([cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2], axis=1)  
  
    mask = scores >= score_threshold  
    if not mask.any():  
        return []  
  
    boxes_f  = boxes[mask]  
    scores_f = scores[mask]  
    decoded_f = decoded[mask]  
  
    keep = nms(boxes_f, scores_f, NMS_IOU_THRESHOLD)  
  
    results = []  
    for idx in keep:  
        b   = boxes_f[idx]  
        kps = decoded_f[idx, 2:7, :].copy()  
  
        box_orig = np.array([  
            (b[0] - pl) / scale,  
            (b[1] - pt) / scale,  
            (b[2] - pl) / scale,  
            (b[3] - pt) / scale,  
        ])  
        kps[:, 0] = (kps[:, 0] - pl) / scale  
        kps[:, 1] = (kps[:, 1] - pt) / scale  
  
        M, _ = cv2.estimateAffinePartial2D(kps, ARCFACE_DST, method=cv2.LMEDS)  
        aligned = None  
        if M is not None:  
            aligned = cv2.warpAffine(  
                img_rgb, M, (CAVAFACE_INPUT_HW[1], CAVAFACE_INPUT_HW[0])  
            )  
  
        results.append({  
            "bbox":    box_orig.tolist(),  
            "score":   float(scores_f[idx]),  
            "aligned": aligned,  
        })  
  
    return results  
  
  
# =============================================================================  
# Face Recognition (CavaFace ONNX)  
# =============================================================================  
  
def get_embedding(face_rgb: np.ndarray, cava_sess: ort.InferenceSession) -> np.ndarray:  
    inp  = (face_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]  
    name = cava_sess.get_inputs()[0].name  
    emb  = cava_sess.run(None, {name: inp})[0].reshape(-1)  
    norm = np.linalg.norm(emb)  
    return emb / (norm + 1e-8)  
  
  
# =============================================================================  
# Face Database  
# =============================================================================  
  
class FaceDB:  
    def __init__(self, db_path: str = "face_db_local.npz"):  
        self.db_path = db_path  
        self.embeddings: dict[str, np.ndarray] = {}  
        self._load()  
  
    def _load(self) -> None:  
        if Path(self.db_path).exists():  
            data = np.load(self.db_path)  
            self.embeddings = {name: data[name] for name in data.files}  
            print(f"  ✓ Loaded existing database: {list(self.embeddings.keys())}")  
  
    def save(self) -> None:  
        np.savez(self.db_path, **self.embeddings)  
  
    def add(self, name: str, embedding: np.ndarray) -> None:  
        self.embeddings[name] = embedding  
        self.save()  
  
    def search(self, query: np.ndarray, threshold: float = 0.45) -> tuple:  
        best_name, best_score = None, -1.0  
        for name, emb in self.embeddings.items():  
            score = float(np.dot(query, emb))  
            if score > best_score:  
                best_score, best_name = score, name  
        if best_score < threshold:  
            return None, best_score  
        return best_name, best_score  
  
    def __len__(self) -> int:  
        return len(self.embeddings)  
  
  
# =============================================================================  
# Pre-enrollment from datasets folder  
# =============================================================================  
  
def build_database_from_folder(datasets_dir, det_sess, anchors, cava_sess, db):  
    datasets_path = Path(datasets_dir)  
    if not datasets_path.exists():  
        print(f"  ⚠ Datasets folder not found: {datasets_dir}")  
        return  
  
    person_dirs = sorted([d for d in datasets_path.iterdir() if d.is_dir()])  
    if not person_dirs:  
        print(f"  ⚠ No sub-folders found in {datasets_dir}")  
        return  
  
    print(f"\nBuilding database from: {datasets_dir}")  
    print("-" * 40)  
  
    enrolled = 0  
    for person_dir in person_dirs:  
        name  = person_dir.name  
        files = [f for f in sorted(person_dir.iterdir())  
                 if f.suffix.lower() in IMG_EXTENSIONS]  
        if not files:  
            print(f"  [{name}] no images — skipped")  
            continue  
  
        embeddings = []  
        for img_path in files:  
            img_bgr = cv2.imread(str(img_path))  
            if img_bgr is None:  
                continue  
            img_rgb   = img_bgr[:, :, ::-1]  
            detections = detect_faces(img_rgb, det_sess, anchors)  
  
            if not detections:  
                print(f"  [{name}] no face in {img_path.name} — skipped")  
                continue  
  
            best  = max(detections, key=lambda d: d["score"])  
            aligned = best["aligned"]  
            if aligned is None:  
                continue  
  
            emb = get_embedding(aligned, cava_sess)  
            embeddings.append(emb)  
            print(f"  [{name}] {img_path.name} ✓")  
  
        if embeddings:  
            avg = np.mean(embeddings, axis=0)  
            avg /= np.linalg.norm(avg) + 1e-8  
            db.add(name, avg)  
            print(f"  → Enrolled: {name} ({len(embeddings)} photo(s))\n")  
            enrolled += 1  
        else:  
            print(f"  [{name}] no valid faces — not enrolled\n")  
  
    print(f"Pre-enrollment done: {enrolled} person(s) added")  
    print(f"Total in database  : {len(db)} person(s)")  
    print("-" * 40)  
  
  
# =============================================================================  
# HTML Template — with Start/Stop Scanning + Edge Light  
# =============================================================================  
  
HTML_TEMPLATE = """  
<!DOCTYPE html>  
<html lang="en">  
<head>  
<meta charset="UTF-8">  
<title>Face Recognition (Local)</title>  
<meta name="viewport" content="width=device-width, initial-scale=1.0">  
<style>  
:root {  
    --bg-1: #0f172a;  
    --bg-2: #1e293b;  
    --glass: rgba(255,255,255,0.08);  
    --border: rgba(255,255,255,0.15);  
    --primary: #6366f1;  
    --success: #22c55e;  
    --danger: #ef4444;  
    --text-main: #f8fafc;  
    --text-sub: #cbd5e1;  
}  
* { margin: 0; padding: 0; box-sizing: border-box; }  
body {  
    font-family: Inter, system-ui, sans-serif;  
    background:  
        radial-gradient(1200px 600px at 10% 10%, #1e1b4b, transparent),  
        linear-gradient(135deg, var(--bg-1), var(--bg-2));  
    color: var(--text-main);  
    min-height: 100vh;  
}  
.container { max-width: 1440px; margin: auto; padding: 32px; }  
h1 { text-align: center; font-size: 2.4rem; font-weight: 700; }  
.subtitle { text-align: center; color: var(--text-sub); margin: 8px 0 32px; }  
.main-content {  
    display: grid;  
    grid-template-columns: 3fr 1.2fr;  
    gap: 24px;  
}  
@media (max-width: 1024px) { .main-content { grid-template-columns: 1fr; } }  
.video-panel {  
    position: relative;  
    background: var(--glass);  
    padding: 16px;  
    border: 1px solid var(--border);  
    backdrop-filter: blur(16px);  
    box-shadow: 0 20px 60px rgba(0,0,0,.35);  
    transition: box-shadow .4s ease, border-color .4s ease;  
}  
.video-panel.edge-green {  
    box-shadow: 0 0 30px 8px rgba(34,197,94,.6), inset 0 0 30px 4px rgba(34,197,94,.15);  
    border-color: rgba(34,197,94,.5);  
}  
.video-panel.edge-red {  
    box-shadow: 0 0 30px 8px rgba(239,68,68,.6), inset 0 0 30px 4px rgba(239,68,68,.15);  
    border-color: rgba(239,68,68,.5);  
}  
.video-panel.edge-blue {  
    box-shadow: 0 0 30px 8px rgba(99,102,241,.6), inset 0 0 30px 4px rgba(99,102,241,.15);  
    border-color: rgba(99,102,241,.5);  
}  
#videoStream { width: 100%; background: #000; display: block; }  
.faces-panel {  
    background: var(--glass);  
    padding: 20px;  
    border: 1px solid var(--border);  
    backdrop-filter: blur(16px);  
    max-height: 640px;  
    overflow-y: auto;  
}  
.faces-panel h2 { margin-bottom: 16px; }  
.face-card {  
    background: linear-gradient(180deg, rgba(255,255,255,.12), rgba(255,255,255,.05));  
    padding: 16px;  
    margin-bottom: 14px;  
    border: 1px solid var(--border);  
    transition: .25s;  
}  
.face-card:hover { transform: translateY(-4px); box-shadow: 0 10px 30px rgba(0,0,0,.35); }  
.face-card.identified { border-left: 4px solid var(--success); }  
.face-card.unknown    { border-left: 4px solid var(--danger); }  
.face-header {  
    display: flex;  
    justify-content: space-between;  
    align-items: center;  
    margin-bottom: 8px;  
}  
.face-name { font-weight: 600; }  
.face-similarity { font-size: .85rem; color: var(--text-sub); }  
.stats {  
    display: grid;  
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));  
    gap: 20px;  
    margin-top: 28px;  
}  
.stat-box {  
    background: var(--glass);  
    padding: 22px;  
    border: 1px solid var(--border);  
    backdrop-filter: blur(16px);  
    text-align: center;  
}  
.stat-value { font-size: 2rem; font-weight: 700; margin-top: 8px; }  
.stat-label { font-size: .75rem; letter-spacing: .15em; text-transform: uppercase; color: var(--text-sub); }  
.badge-local {  
    display: inline-block;  
    background: rgba(99,102,241,.25);  
    border: 1px solid var(--primary);  
    color: #a5b4fc;  
    font-size: .7rem;  
    padding: 2px 8px;  
    border-radius: 999px;  
    margin-left: 10px;  
    vertical-align: middle;  
    letter-spacing: .05em;  
}  
.scan-overlay {  
    position: absolute;  
    top: 16px; left: 16px; right: 16px; bottom: 16px;  
    display: flex;  
    align-items: center;  
    justify-content: center;  
    background: rgba(0,0,0,.55);  
    z-index: 10;  
}  
.btn-scan {  
    display: inline-block;  
    padding: 14px 36px;  
    font-size: 1.1rem;  
    font-weight: 600;  
    color: #fff;  
    background: var(--primary);  
    border: none;  
    border-radius: 8px;  
    cursor: pointer;  
    letter-spacing: .04em;  
    transition: background .2s, transform .15s;  
}  
.btn-scan:hover { background: #4f46e5; transform: translateY(-2px); }  
.btn-stop { background: var(--danger); padding: 10px 28px; font-size: .95rem; }  
.btn-stop:hover { background: #dc2626; }  
</style>  
</head>  
<body>  
<div class="container">  
    <h1>Face Recognition <span class="badge-local">LOCAL / ONNX</span></h1>  
    <p class="subtitle">BlazeFace + CavaFace — running on CPU via ONNX Runtime</p>  
  
    <div class="main-content">  
        <div class="video-panel" id="videoPanel">  
            <img id="videoStream" src="">  
            <div class="scan-overlay" id="scanOverlay">  
                <button class="btn-scan" onclick="startScanning()">&#9654; Start Scanning</button>  
            </div>  
            <div style="text-align:center; margin-top:12px;">  
                <button class="btn-scan btn-stop" id="btnStop" onclick="stopScanning()" style="display:none;">&#9632; Stop Scanning</button>  
            </div>  
        </div>  
        <div class="faces-panel">  
            <h2>Detected Faces</h2>  
            <div id="facesList"></div>  
        </div>  
    </div>  
  
    <div class="stats">  
        <div class="stat-box">  
            <div class="stat-label">People in Database</div>  
            <div class="stat-value" id="dbCount">--</div>  
        </div>  
        <div class="stat-box">  
            <div class="stat-label">Faces Detected</div>  
            <div class="stat-value" id="facesCount">--</div>  
        </div>  
    </div>  
</div>  
  
<script>  
const facesCount = document.getElementById('facesCount');  
const dbCount    = document.getElementById('dbCount');  
const facesList  = document.getElementById('facesList');  
const videoPanel = document.getElementById('videoPanel');  
let pollTimer = null;  
  
function updateFaces() {  
    fetch('/get_faces')  
        .then(r => r.json())  
        .then(data => {  
            facesCount.textContent = data.faces.length;  
            dbCount.textContent    = data.db_size;  
  
            facesList.innerHTML = data.faces.length === 0  
                ? '<div style="opacity:.6;text-align:center">No faces detected</div>'  
                : data.faces.map(face => `  
                    <div class="face-card ${face.identified ? 'identified' : 'unknown'}">  
                        <div class="face-header">  
                            <div class="face-name">  
                                ${face.identified ? '&#10003; ' + face.name : '&#10067; Unknown'}  
                            </div>  
                            ${face.identified  
                                ? '<div class="face-similarity">' + (face.similarity * 100).toFixed(1) + '%</div>'  
                                : '<div class="face-similarity" style="color:var(--danger)">Not in DB</div>'  
                            }  
                        </div>  
                        <div style="font-size:.85rem;opacity:.7">  
                            Score: ${face.detection_score.toFixed(3)}  
                        </div>  
                    </div>  
                `).join('');  
  
            // Edge Light  
            videoPanel.classList.remove('edge-green', 'edge-red', 'edge-blue');  
            if (data.faces.length > 0) {  
                const hasIdentified = data.faces.some(f => f.identified);  
                const hasUnknown    = data.faces.some(f => !f.identified);  
                if (hasIdentified && !hasUnknown)      videoPanel.classList.add('edge-green');  
                else if (!hasIdentified && hasUnknown)  videoPanel.classList.add('edge-red');  
                else                                    videoPanel.classList.add('edge-blue');  
            }  
        });  
}  
  
function startScanning() {  
    document.getElementById('videoStream').src = '/video_feed';  
    document.getElementById('scanOverlay').style.display = 'none';  
    document.getElementById('btnStop').style.display = 'inline-block';  
    pollTimer = setInterval(updateFaces, 1000);  
    updateFaces();  
}  
  
function stopScanning() {  
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }  
    document.getElementById('videoStream').src = '';  
    document.getElementById('scanOverlay').style.display = 'flex';  
    document.getElementById('btnStop').style.display = 'none';  
    videoPanel.classList.remove('edge-green', 'edge-red', 'edge-blue');  
    facesList.innerHTML = '<div style="opacity:.6;text-align:center">Scanning stopped</div>';  
    facesCount.textContent = '--';  
}  
</script>  
</body>  
</html>  
"""
# =============================================================================  
# Flask App + Detection Thread  
# =============================================================================  
  
app          = Flask(__name__)  
output_frame = None  
lock         = threading.Lock()  
face_results = []  
  
det_sess_g   = None  
cava_sess_g  = None  
anchors_g    = None  
db_g         = None  
  
  
def detection_thread(camera_id: int, threshold: float, skip_frames: int = 1) -> None:  
    global output_frame, lock, face_results  
  
    cap = cv2.VideoCapture(camera_id)  
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  
  
    if not cap.isOpened():  
        print(f"Error: Could not open camera {camera_id}")  
        return  
  
    print("Camera opened successfully")  
  
    frame_count = 0  
    last_faces  = []  
  
    while True:  
        ret, frame = cap.read()  
        if not ret:  
            time.sleep(0.05)  
            continue  
  
        frame_count += 1  
  
        if frame_count % (skip_frames + 1) == 1:  
            img_rgb    = frame[:, :, ::-1]  
            detections = detect_faces(img_rgb, det_sess_g, anchors_g)  
  
            faces = []  
            for det in detections:  
                aligned = det["aligned"]  
                if aligned is None:  
                    continue  
  
                emb  = get_embedding(aligned, cava_sess_g)  
                name, similarity = db_g.search(emb, threshold=threshold)  
  
                faces.append({  
                    "bbox":            det["bbox"],  
                    "detection_score": det["score"],  
                    "embedding":       emb,  
                    "name":            name,  
                    "similarity":      similarity,  
                    "identified":      name is not None,  
                })  
  
            if len(faces) > 1:  
                seen: dict[str, int] = {}  
                for i, face in enumerate(faces):  
                    if not face["identified"]:  
                        continue  
                    n = face["name"]  
                    if n not in seen or face["similarity"] > faces[seen[n]]["similarity"]:  
                        seen[n] = i  
  
                best_indices = set(seen.values())  
                for i, face in enumerate(faces):  
                    if face["identified"] and i not in best_indices:  
                        face["name"]       = None  
                        face["similarity"] = 0.0  
                        face["identified"] = False  
  
            last_faces = faces  
  
        annotated = frame.copy()  
        for face in last_faces:  
            x1, y1, x2, y2 = [int(v) for v in face["bbox"]]  
  
            if face["identified"]:  
                color = (0, 255, 0)  
                label = f"{face['name']} ({face['similarity']:.2f})"  
            else:  
                color = (0, 0, 255)  
                label = "Unknown"  
  
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)  
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)  
            cv2.putText(annotated, label, (x1, y1 - 5),  
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  
  
        with lock:  
            output_frame = annotated.copy()  
            face_results = last_faces.copy()  
  
    cap.release()  
  
  
def generate_frames():  
    global output_frame, lock  
    while True:  
        with lock:  
            if output_frame is None:  
                time.sleep(0.05)  
                continue  
            ok, encoded = cv2.imencode(".jpg", output_frame)  
            if not ok:  
                continue  
            frame_bytes = bytearray(encoded)  
  
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"  
               + frame_bytes + b"\r\n")  
        time.sleep(0.033)  
  
  
@app.route("/")  
def index():  
    return render_template_string(HTML_TEMPLATE)  
  
  
@app.route("/video_feed")  
def video_feed():  
    return Response(generate_frames(),  
                    mimetype="multipart/x-mixed-replace; boundary=frame")  
  
  
@app.route("/get_faces")  
def get_faces():  
    global face_results, db_g  
    with lock:  
        faces = [  
            {  
                "bbox":            f["bbox"],  
                "detection_score": f["detection_score"],  
                "name":            f["name"],  
                "similarity":      float(f["similarity"]),  
                "identified":      f["identified"],  
            }  
            for f in face_results  
        ]  
    return jsonify({"faces": faces, "db_size": len(db_g)})  
  
  
# =============================================================================  
# Main  
# =============================================================================  
  
def main():  
    global det_sess_g, cava_sess_g, anchors_g, db_g  
  
    parser = argparse.ArgumentParser(  
        description="Web-Based Face Recognition — Local (ONNX) version for Mac/Linux"  
    )  
    parser.add_argument("--camera",    type=int,   default=0,  
                        help="Camera ID (default: 0)")  
    parser.add_argument("--datasets",  default="",  
                        help="Path to datasets folder (sub-folder name = person name)")  
    parser.add_argument("--db",        default="face_db_local.npz",  
                        help="Face database file (default: face_db_local.npz)")  
    parser.add_argument("--detector",  default="FaceDetector.onnx",  
                        help="Path to FaceDetector.onnx")  
    parser.add_argument("--cavaface",  default="cavaface.onnx",  
                        help="Path to cavaface.onnx")  
    parser.add_argument("--threshold", type=float, default=0.45,  
                        help="Cosine similarity threshold (default: 0.45)")  
    parser.add_argument("--skip-frames", type=int, default=1,  
                        help="Process every N+1 frames (default: 1)")  
    parser.add_argument("--host",      default="0.0.0.0")  
    parser.add_argument("--port",      type=int, default=5001)  
    args = parser.parse_args()  
  
    print("=" * 60)  
    print("Face Recognition System  [LOCAL / ONNX]")  
    print("=" * 60)  
  
    print("\nLoading models ...")  
    if not Path(args.detector).exists():  
        print(f"  ✗ FaceDetector not found: {args.detector}")  
        print("    Specify with --detector /path/to/FaceDetector.onnx")  
        return 1  
    if not Path(args.cavaface).exists():  
        print(f"  ✗ CavaFace not found: {args.cavaface}")  
        print("    Specify with --cavaface /path/to/cavaface.onnx")  
        return 1  
  
    orig_dir = os.getcwd()  
  
    detector_path = Path(args.detector).resolve()  
    os.chdir(detector_path.parent)  
    det_sess_g = ort.InferenceSession(detector_path.name)  
    os.chdir(orig_dir)  
    print("  ✓ FaceDetector loaded")  
  
    cavaface_path = Path(args.cavaface).resolve()  
    os.chdir(cavaface_path.parent)  
    cava_sess_g = ort.InferenceSession(cavaface_path.name)  
    os.chdir(orig_dir)  
    print("  ✓ CavaFace loaded")  
  
    anchors_g = generate_anchors()  
  
    print("\nInitializing database ...")  
    db_g = FaceDB(args.db)  
  
    if args.datasets:  
        build_database_from_folder(  
            args.datasets, det_sess_g, anchors_g, cava_sess_g, db_g  
        )  
    else:  
        print("  (No --datasets specified — using existing database only)")  
  
    if len(db_g) == 0:  
        print("  ⚠ Database is empty — everyone will appear as Unknown")  
    else:  
        print(f"  ✓ Database ready: {list(db_g.embeddings.keys())}")  
  
    print("\nStarting camera ...")  
    t = threading.Thread(  
        target=detection_thread,  
        args=(args.camera, args.threshold, args.skip_frames),  
        daemon=True,  
    )  
    t.start()  
    time.sleep(1)  
  
    print(f"\n{'='*60}")  
    print("  Open your browser:")  
    print(f"  http://localhost:{args.port}")  
    print(f"\n  Press Ctrl+C to stop")  
    print(f"{'='*60}\n")  
  
    try:  
        app.run(host=args.host, port=args.port, threaded=True, debug=False)  
    except KeyboardInterrupt:  
        print("\nShutting down ...")  
    return 0  
  
  
if __name__ == "__main__":  
    exit(main())