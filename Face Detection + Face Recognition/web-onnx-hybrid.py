import argparse
import os
import threading
import time
import base64
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, Response, jsonify, render_template_string

# =============================================================================
# Constants & Model Parameters
# =============================================================================
DETECT_INPUT_HW   = (256, 256)
CAVAFACE_INPUT_HW = (112, 112)
SCORE_THRESHOLD   = 0.70
NMS_IOU_THRESHOLD = 0.3
IMG_EXTENSIONS    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

# Global State & Locks
shared_frame_lock = threading.Lock()
ai_lock = threading.Lock()

shared_frame = None
last_known_box = None

det_sess_g = None
cava_sess_g = None
anchors_g = None
db_g = None

app = Flask(__name__)

# =============================================================================
# ONNX Math & Helpers (BlazeFace & CavaFace)
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

def detect_faces(img_rgb: np.ndarray, det_sess: ort.InferenceSession, anchors: np.ndarray) -> list:
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

    mask = scores >= SCORE_THRESHOLD
    if not mask.any(): return []

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

        # Face Alignment for CavaFace
        M, _ = cv2.estimateAffinePartial2D(kps, ARCFACE_DST, method=cv2.LMEDS)
        aligned = None
        if M is not None:
            aligned = cv2.warpAffine(img_rgb, M, (CAVAFACE_INPUT_HW[1], CAVAFACE_INPUT_HW[0]))

        results.append({
            "bbox":    box_orig.tolist(),
            "score":   float(scores_f[idx]),
            "aligned": aligned,
        })
    return results

def get_embedding(face_rgb: np.ndarray, cava_sess: ort.InferenceSession) -> np.ndarray:
    inp  = (face_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    name = cava_sess.get_inputs()[0].name
    emb  = cava_sess.run(None, {name: inp})[0].reshape(-1)
    norm = np.linalg.norm(emb)
    return emb / (norm + 1e-8)

# =============================================================================
# Face Database Manager
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

def build_database_from_folder(datasets_dir: str):
    datasets_path = Path(datasets_dir)
    if not datasets_path.exists(): return
    
    person_dirs = sorted([d for d in datasets_path.iterdir() if d.is_dir()])
    print(f"\n--- Enrolling faces from: {datasets_dir} ---")
    
    for person_dir in person_dirs:
        name  = person_dir.name
        files = [f for f in sorted(person_dir.iterdir()) if f.suffix.lower() in IMG_EXTENSIONS]
        embeddings = []
        for img_path in files:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None: continue
            img_rgb   = img_bgr[:, :, ::-1]
            detections = detect_faces(img_rgb, det_sess_g, anchors_g)
            if not detections: continue
            best  = max(detections, key=lambda d: d["score"])
            if best["aligned"] is not None:
                emb = get_embedding(best["aligned"], cava_sess_g)
                embeddings.append(emb)
        if embeddings:
            avg = np.mean(embeddings, axis=0)
            avg /= np.linalg.norm(avg) + 1e-8
            db_g.add(name, avg)
            print(f"  ✓ Enrolled: {name}")

# =============================================================================
# Flask Server & Camera Logic (with Throttling)
# =============================================================================

def camera_thread(camera_index):
    global shared_frame, last_known_box
    cap = cv2.VideoCapture(camera_index)
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            
            # Draw box instantly from memory (No AI lag)
            if last_known_box:
                bx, by, bw, bh = last_known_box
                cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (255, 0, 0), 2)
            
            with shared_frame_lock:
                shared_frame = frame.copy()
        else:
            time.sleep(0.01)

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with shared_frame_lock:
                if shared_frame is None: continue
                ok, encoded = cv2.imencode(".jpg", shared_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok: yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + bytearray(encoded) + b"\r\n")
            time.sleep(0.033)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/check_proximity", methods=["POST"])
def check_proximity():
    global last_known_box

    with shared_frame_lock:
        if shared_frame is None: return jsonify({"status": "no_frame"})
        frame = shared_frame.copy()

    H, W = frame.shape[:2]

    with ai_lock:
        img_rgb = frame[:, :, ::-1]
        detections = detect_faces(img_rgb, det_sess_g, anchors_g)

    if not detections:
        last_known_box = None
        return jsonify({"status": "no_face", "instruction": "Searching for Face...", "ratio": 0})

    # Pick the largest/most confident face
    best_face = max(detections, key=lambda d: d["score"])
    x1, y1, x2, y2 = best_face["bbox"]
    
    bx, by = int(x1), int(y1)
    bw, bh = int(x2 - x1), int(y2 - y1)
    last_known_box = [max(0, bx), max(0, by), bw, bh]

    ratio = (bw * bh) / (W * H)
    
    # Proximity Limits for BlazeFace
    status = "wait"
    if ratio < 0.05:
        instruction = "Move Closer"
        color = (0, 165, 255) # Orange
    elif ratio > 0.22:
        instruction = "Too Close! Move Back"
        color = (0, 165, 255)
    else:
        status = "success"
        
        # WE ARE AT PERFECT DISTANCE! RUN RECOGNITION!
        with ai_lock:
            emb = get_embedding(best_face["aligned"], cava_sess_g)
            name, similarity = db_g.search(emb, threshold=0.45)
            
        if name:
            instruction = f"Verified: {name} ({(similarity*100):.1f}%)"
            color = (0, 255, 0) # Green
        else:
            instruction = "Intruder Alert! Face not recognized."
            color = (0, 0, 255) # Red

    # Draw the preview box for the snapshot
    annotated = frame.copy()
    cv2.rectangle(annotated, (max(0, bx), max(0, by)), (bx+bw, by+bh), color, 3)
    
    _, buffer = cv2.imencode('.jpg', annotated)
    img_b64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({"status": status, "instruction": instruction, "ratio": float(ratio), "image": img_b64})

# =============================================================================
# Dark HTML UI (With Continuous Scanning)
# =============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Face ID: Proximity & Verification</title>
    <style>
        body { background: #0e0e12; color: #fff; font-family: 'Segoe UI', sans-serif; text-align: center; }
        .wrap { display: flex; justify-content: center; gap: 30px; padding: 40px; flex-wrap: wrap; }
        .panel { background: #1c1c24; padding: 20px; border-radius: 15px; border: 1px solid #333; width: 500px; }
        img { width: 100%; border-radius: 10px; border: 2px solid #222; }
        #instruction { font-size: 1.6rem; font-weight: bold; margin: 20px 0; color: #00d4ff; min-height: 50px; }
        .bar-container { background: #222; height: 12px; border-radius: 6px; overflow: hidden; margin: 10px 0; }
        #bar { background: #00ff88; height: 100%; width: 0%; transition: 0.2s; }
        button { background: #00d4ff; color: #000; border: none; padding: 15px 40px; border-radius: 30px; font-size: 1.1rem; font-weight: bold; cursor: pointer; transition: 0.2s; }
        button:hover { background: #00ff88; transform: scale(1.05); }
        button:disabled { background: #444; color: #888; transform: none; }
    </style>
</head>
<body>
    <h1 style="margin-top:30px;">👤 Face ID: ONNX Engine</h1>
    <div class="wrap">
        <div class="panel">
            <h3>Live Guidance</h3>
            <img src="/video_feed">
            <div id="instruction">Ready</div>
            <div class="bar-container"><div id="bar"></div></div>
            <br>
            <button id="btn" onclick="start()">Start Verification</button>
        </div>
        <div class="panel">
            <h3>Verified Snapshot</h3>
            <div style="height:375px; background:#000; border-radius:10px; display:flex; align-items:center; justify-content:center;">
                <img id="snap" src="" style="display:none;">
                <span id="msg" style="color:#555;">Waiting for perfect distance...</span>
            </div>
        </div>
    </div>
    <script>
        let active = false;
        function start() {
            active = true;
            document.getElementById('btn').innerText = "Scanning Continuously...";
            document.getElementById('btn').disabled = true;
            loop();
        }
        function loop() {
            if(!active) return;
            fetch('/check_proximity', {method:'POST'})
            .then(r => r.json())
            .then(d => {
                if(d.status !== "no_frame") {
                    document.getElementById('instruction').innerText = d.instruction;
                    
                    // Cap progress bar visually at the perfect threshold (approx 0.22)
                    let p = Math.min(100, (d.ratio / 0.22) * 100);
                    document.getElementById('bar').style.width = p + "%";
                    
                    // Update UI colors based on recognized/intruder text
                    if(d.instruction.includes("Intruder")) {
                        document.getElementById('instruction').style.color = "#ff4444";
                        document.getElementById('bar').style.background = "#ff4444";
                    } else if (d.status === "success") {
                        document.getElementById('instruction').style.color = "#00ff88";
                        document.getElementById('bar').style.background = "#00ff88";
                    } else {
                        document.getElementById('instruction').style.color = "#00d4ff";
                        document.getElementById('bar').style.background = "#00d4ff";
                    }

                    if(d.status === "success") {
                        document.getElementById('snap').src = "data:image/jpeg;base64," + d.image;
                        document.getElementById('snap').style.display = "block";
                        document.getElementById('msg').style.display = "none";
                    } 
                    
                    // Continuous scanning (Throttled to 500ms for CPU performance)
                    setTimeout(loop, 500);
                }
            })
            .catch(() => { if(active) setTimeout(loop, 1000); });
        }
    </script>
</body>
</html>
"""

# =============================================================================
# Main Initialization
# =============================================================================

def main():
    global det_sess_g, cava_sess_g, anchors_g, db_g

    parser = argparse.ArgumentParser()
    parser.add_argument("--camera",   type=int, default=0, help="Camera ID (0 for Mac, 1/2 for IoT)")
    parser.add_argument("--datasets", default="", help="Path to datasets folder")
    parser.add_argument("--detector", required=True, help="Path to FaceDetector.onnx")
    parser.add_argument("--cavaface", required=True, help="Path to cavaface.onnx")
    args = parser.parse_args()

    print("--- Loading ONNX Models ---")
    det_sess_g = ort.InferenceSession(args.detector)
    cava_sess_g = ort.InferenceSession(args.cavaface)
    anchors_g = generate_anchors()

    db_g = FaceDB("face_db_local.npz")
    if args.datasets:
        build_database_from_folder(args.datasets)

    threading.Thread(target=camera_thread, args=(args.camera,), daemon=True).start()
    app.run(host="0.0.0.0", port=5001, threaded=True)

if __name__ == "__main__":
    main()