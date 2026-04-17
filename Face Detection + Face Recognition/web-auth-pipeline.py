import argparse
import os
import threading
import time
import base64
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, Response, jsonify, render_template_string, request

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter

# =============================================================================
# Global State & Locks
# =============================================================================
shared_frame_lock = threading.Lock()
ai_lock = threading.Lock()

shared_frame = None
last_known_face = None
last_known_hand = None
use_fallback = False

app = Flask(__name__)

# =============================================================================
# FACE MODELS (ONNX: BlazeFace + CavaFace)
# =============================================================================
DETECT_INPUT_HW   = (256, 256)
CAVAFACE_INPUT_HW = (112, 112)
FACE_SCORE_THR    = 0.70
NMS_IOU_THRESHOLD = 0.3
IMG_EXTENSIONS    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

ARCFACE_DST = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041],
], dtype=np.float32)

def generate_anchors(input_size: int = 256) -> np.ndarray:
    strides, anchors_per_cell = [16, 32], [2, 6]
    rows = []
    for stride, n in zip(strides, anchors_per_cell):
        grid = input_size // stride
        for y in range(grid):
            for x in range(grid):
                cx, cy = (x + 0.5) / grid, (y + 0.5) / grid
                for _ in range(n): rows.append([cx, cy, 1.0, 1.0])
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

def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter  = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union  = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0

def detect_faces(img_rgb: np.ndarray, det_sess: ort.InferenceSession, anchors: np.ndarray) -> list:
    inp, scale, pt, pl = resize_pad(img_rgb, DETECT_INPUT_HW)
    name = det_sess.get_inputs()[0].name
    c1, c2, s1, s2 = det_sess.run(None, {name: inp})

    coords = np.concatenate([c1[0], c2[0]], axis=0).reshape(-1, 8, 2)
    scores = np.concatenate([s1[0], s2[0]], axis=0).reshape(-1)
    scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -100.0, 100.0)))

    H, W = DETECT_INPUT_HW
    center = anchors[:, 0:1, :] * np.array([[W, H]], dtype=np.float32)
    scale_arr  = anchors[:, 1:2, :]
    mask_arr = np.ones((coords.shape[1], 1), dtype=np.float32)
    mask_arr[1] = 0.0
    decoded = coords * scale_arr + center * mask_arr

    cx, cy = decoded[:, 0, 0], decoded[:, 0, 1]
    bw, bh = decoded[:, 1, 0], decoded[:, 1, 1]
    boxes  = np.stack([cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2], axis=1)

    mask = scores >= FACE_SCORE_THR
    if not mask.any(): return []
    boxes_f, scores_f, decoded_f = boxes[mask], scores[mask], decoded[mask]

    order = scores_f.argsort()[::-1].tolist()
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if box_iou(boxes_f[i], boxes_f[j]) < NMS_IOU_THRESHOLD]

    results = []
    for idx in keep:
        b = boxes_f[idx]
        kps = decoded_f[idx, 2:7, :].copy()
        box_orig = np.array([(b[0]-pl)/scale, (b[1]-pt)/scale, (b[2]-pl)/scale, (b[3]-pt)/scale])
        kps[:, 0], kps[:, 1] = (kps[:, 0] - pl) / scale, (kps[:, 1] - pt) / scale

        M, _ = cv2.estimateAffinePartial2D(kps, ARCFACE_DST, method=cv2.LMEDS)
        aligned = cv2.warpAffine(img_rgb, M, (CAVAFACE_INPUT_HW[1], CAVAFACE_INPUT_HW[0])) if M is not None else None
        results.append({"bbox": box_orig.tolist(), "score": float(scores_f[idx]), "aligned": aligned})
    return results

def get_embedding(face_rgb: np.ndarray, cava_sess: ort.InferenceSession) -> np.ndarray:
    inp  = (face_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    name = cava_sess.get_inputs()[0].name
    emb  = cava_sess.run(None, {name: inp})[0].reshape(-1)
    return emb / (np.linalg.norm(emb) + 1e-8)

class FaceDB:
    def __init__(self):
        self.embeddings: dict[str, np.ndarray] = {}
        
    def add(self, name: str, embedding: np.ndarray):
        self.embeddings[name] = embedding

    def search(self, query: np.ndarray, threshold: float = 0.45) -> tuple:
        best_name, best_score = None, -1.0
        for name, emb in self.embeddings.items():
            score = float(np.dot(query, emb))
            if score > best_score: best_score, best_name = score, name
        return (best_name, best_score) if best_score >= threshold else (None, best_score)

def build_database_from_folder(datasets_dir: str):
    datasets_path = Path(datasets_dir)
    if not datasets_path.exists(): return
    for person_dir in [d for d in datasets_path.iterdir() if d.is_dir()]:
        name = person_dir.name
        files = [f for f in sorted(person_dir.iterdir()) if f.suffix.lower() in IMG_EXTENSIONS]
        embeddings = []
        for img_path in files:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None: continue
            detections = detect_faces(img_bgr[:, :, ::-1], det_sess_g, anchors_g)
            if not detections: continue
            best = max(detections, key=lambda d: d["score"])
            if best["aligned"] is not None:
                embeddings.append(get_embedding(best["aligned"], cava_sess_g))
        if embeddings:
            avg = np.mean(embeddings, axis=0)
            db_g.add(name, avg / (np.linalg.norm(avg) + 1e-8))
            print(f"  ✓ Enrolled Face: {name}")

# =============================================================================
# HAND MODELS (TFLite: Palm + Landmark + Gesture)
# =============================================================================
GESTURE_LABELS = ["None", "Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"]
HAND_CONNECTIONS = [(0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8), (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16), (0,17), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17)]
FINGER_COLORS = {"thumb": (0,255,255), "index": (0,165,255), "middle": (0,255,0), "ring": (255,0,0), "pinky": (255,0,255), "palm": (200,200,200)}
LANDMARK_FINGER_MAP = {0: "palm", 1: "thumb", 2: "thumb", 3: "thumb", 4: "thumb", 5: "index", 6: "index", 7: "index", 8: "index", 9: "middle", 10: "middle", 11: "middle", 12: "middle", 13: "ring", 14: "ring", 15: "ring", 16: "ring", 17: "pinky", 18: "pinky", 19: "pinky", 20: "pinky"}

def _dequantize(tensor, detail):
    if detail['dtype'] == np.uint8:
        scale, zp = detail['quantization']
        return (tensor.astype(np.float32) - zp) * scale
    return tensor.astype(np.float32)

def _quantize(data, detail):
    if detail['dtype'] == np.uint8:
        scale, zp = detail['quantization']
        return np.clip(np.round(data / scale + zp), 0, 255).astype(np.uint8)
    return data.astype(np.float32)

class PalmDetectorTFLite:
    def __init__(self, model_path):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.in_det = self.interpreter.get_input_details()
        self.out_det = self.interpreter.get_output_details()
        self.size = 256
        self.anchors = np.array([[ (x+0.5)*s/self.size, (y+0.5)*s/self.size ] for s in [8,16] for y in range(self.size//s) for x in range(self.size//s) for _ in range(2)], dtype=np.float32)
        self.s_idx = next(i for i, d in enumerate(self.out_det) if "score" in d["name"].lower())
        self.c_idx = next(i for i, d in enumerate(self.out_det) if "coord" in d["name"].lower() or "regress" in d["name"].lower())

    def detect(self, frame):
        h, w = frame.shape[:2]
        img = cv2.cvtColor(cv2.resize(frame, (self.size, self.size)), cv2.COLOR_BGR2RGB)
        img = img.astype(np.uint8) if self.in_det[0]['dtype'] == np.uint8 else img.astype(np.float32)/255.0
        self.interpreter.set_tensor(self.in_det[0]["index"], np.expand_dims(img, axis=0))
        self.interpreter.invoke()

        scores = _dequantize(self.interpreter.get_tensor(self.out_det[self.s_idx]["index"]), self.out_det[self.s_idx]).flatten()[:len(self.anchors)]
        coords = _dequantize(self.interpreter.get_tensor(self.out_det[self.c_idx]["index"]), self.out_det[self.c_idx]).reshape(-1, 18)[:len(self.anchors)]
        
        scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -100, 100)))
        mask = scores > 0.65
        if not mask.any(): return []

        scores, coords, anchors = scores[mask], coords[mask], self.anchors[mask]
        coords[:,0] += anchors[:,0] * self.size
        coords[:,1] += anchors[:,1] * self.size
        
        cx, cy, cw, ch = coords[:,0], coords[:,1], coords[:,2], coords[:,3]
        boxes = np.stack([cx - cw/2, cy - ch/2, cx + cw/2, cy + ch/2], axis=1)

        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1, yy1 = np.maximum(boxes[i,0], boxes[order[1:],0]), np.maximum(boxes[i,1], boxes[order[1:],1])
            xx2, yy2 = np.minimum(boxes[i,2], boxes[order[1:],2]), np.minimum(boxes[i,3], boxes[order[1:],3])
            inter = np.maximum(0.0, xx2-xx1) * np.maximum(0.0, yy2-yy1)
            area_i, area_o = (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1]), (boxes[order[1:],2]-boxes[order[1:],0])*(boxes[order[1:],3]-boxes[order[1:],1])
            ovr = inter / (area_i + area_o - inter + 1e-6)
            order = order[np.where(ovr <= 0.3)[0] + 1]

        sx, sy = w / self.size, h / self.size
        return [[boxes[i,0]*sx, boxes[i,1]*sy, boxes[i,2]*sx, boxes[i,3]*sy] for i in keep]

class HandLandmarkTFLite:
    def __init__(self, model_path):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.in_det = self.interpreter.get_input_details()
        self.out_det = self.interpreter.get_output_details()
        self.out_map = {k: i for i, d in enumerate(self.out_det) for k in ["landmarks", "scores", "lr"] if k[:4] in d["name"].lower()}

    def detect(self, frame, bbox):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        x1, y1 = max(0, x1 - bw*1.0), max(0, y1 - bh*1.5)
        x2, y2 = min(w, x2 + bw*1.0), min(h, y2 + bh*1.5)
        crop = frame[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0: return None

        img = cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2RGB)
        img = img.astype(np.uint8) if self.in_det[0]['dtype'] == np.uint8 else img.astype(np.float32)/255.0
        self.interpreter.set_tensor(self.in_det[0]["index"], np.expand_dims(img, axis=0))
        self.interpreter.invoke()

        lm = _dequantize(self.interpreter.get_tensor(self.out_det[self.out_map["landmarks"]]["index"]), self.out_det[self.out_map["landmarks"]]).flatten()[:63].reshape(21, 3)
        sc = _dequantize(self.interpreter.get_tensor(self.out_det[self.out_map["scores"]]["index"]), self.out_det[self.out_map["scores"]]).flatten()[0]
        lr = _dequantize(self.interpreter.get_tensor(self.out_det[self.out_map["lr"]]["index"]), self.out_det[self.out_map["lr"]]).flatten()[0]

        if 1.0 / (1.0 + np.exp(-np.clip(sc, -100, 100))) < 0.7: return None

        lm_frame = lm.copy()
        lm_frame[:, 0] = lm[:, 0] / 224 * (x2 - x1) + x1
        lm_frame[:, 1] = lm[:, 1] / 224 * (y2 - y1) + y1
        return {"local": lm, "frame": lm_frame, "handedness": float(1.0 / (1.0 + np.exp(-np.clip(lr, -100, 100))))}

class GestureClassifierTFLite:
    def __init__(self, model_path):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

    def classify(self, landmarks_local, handedness):
        pts = landmarks_local.copy().astype(np.float32)
        normed = pts - pts[[0, 1, 5, 9, 13, 17]].mean(axis=0)
        normed /= max(normed[:, 0].max() - normed[:, 0].min(), normed[:, 1].max() - normed[:, 1].min()) + 1e-5
        
        inp = np.expand_dims(_quantize(np.append(normed.flatten(), np.float32(handedness)), self.interpreter.get_input_details()[0]), axis=0)
        self.interpreter.set_tensor(self.interpreter.get_input_details()[0]["index"], inp)
        self.interpreter.invoke()
        
        logits = _dequantize(self.interpreter.get_tensor(self.interpreter.get_output_details()[0]["index"]), self.interpreter.get_output_details()[0]).flatten()[:8]
        probs = np.exp(logits - logits.max()) / (np.exp(logits - logits.max()).sum() + 1e-8)
        return GESTURE_LABELS[int(np.argmax(probs))], float(np.max(probs))

def draw_hand_landmarks(frame, landmarks, gesture=""):
    for i, j in HAND_CONNECTIONS:
        cv2.line(frame, (int(landmarks[i][0]), int(landmarks[i][1])), (int(landmarks[j][0]), int(landmarks[j][1])), FINGER_COLORS.get(LANDMARK_FINGER_MAP.get(i, "palm"), (200,200,200)), 2)
    for idx in range(21):
        cv2.circle(frame, (int(landmarks[idx][0]), int(landmarks[idx][1])), 5, FINGER_COLORS.get(LANDMARK_FINGER_MAP.get(idx, "palm"), (200,200,200)), -1)
    if gesture and gesture != "None":
        cv2.putText(frame, f"Gesture: {gesture}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

def is_victory_sign(landmarks):
    """Bypasses AI and uses pure geometry to check if exactly 2 fingers are up."""
    if landmarks is None or len(landmarks) < 21: return False
    
    fingers_up = 0
    # Map Tip vs PIP (middle knuckle) for Index, Middle, Ring, Pinky
    for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        # In OpenCV, Y=0 is the top of the screen. 
        # So tip[1] < pip[1] means the finger is pointing UP.
        if landmarks[tip][1] < landmarks[pip][1]: 
            fingers_up += 1
            
    # It's a victory sign if exactly 2 fingers (Index and Middle) are raised
    return fingers_up == 2

# =============================================================================
# Flask Server & API
# =============================================================================

def camera_thread(camera_index):
    global shared_frame, last_known_face, last_known_hand
    cap = cv2.VideoCapture(camera_index)
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            # Draw Face Box from memory
            if last_known_face:
                bx, by, bw, bh = last_known_face
                cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (255, 0, 0), 2)
            # Draw Hand from memory
            if last_known_hand:
                draw_hand_landmarks(frame, last_known_hand['pts'], last_known_hand['gesture'])
            
            with shared_frame_lock:
                shared_frame = frame.copy()
        else: time.sleep(0.01)

@app.route("/")
def index(): return render_template_string(HTML_TEMPLATE)

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

@app.route("/process", methods=["POST"])
def process_frame():
    global last_known_face, last_known_hand, use_fallback
    req = request.get_json()
    phase = req.get("phase", "face")

    with shared_frame_lock:
        if shared_frame is None: return jsonify({"status": "no_frame"})
        frame = shared_frame.copy()

    H, W = frame.shape[:2]
    annotated = frame.copy()

    # ---------------------------------------------------------
    # PHASE 1: FACE DETECTION & RECOGNITION
    # ---------------------------------------------------------
    if phase == "face":
        last_known_hand = None # Clear old hands
        with ai_lock:
            detections = detect_faces(frame[:, :, ::-1], det_sess_g, anchors_g)

        if not detections:
            last_known_face = None
            return jsonify({"status": "wait", "instruction": "Searching for Face...", "ratio": 0})

        best_face = max(detections, key=lambda d: d["score"])
        x1, y1, x2, y2 = best_face["bbox"]
        bx, by, bw, bh = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        last_known_face = [max(0, bx), max(0, by), bw, bh]

        ratio = (bw * bh) / (W * H)
                
        if ratio < 0.2:
            return jsonify({"status": "wait", "instruction": "Move Closer", "ratio": ratio})
        elif ratio > 0.4:
            return jsonify({"status": "wait", "instruction": "Too Close! Move Back", "ratio": ratio})
        
        # Perfect distance -> Verify Identity
        with ai_lock:
            emb = get_embedding(best_face["aligned"], cava_sess_g)
            name, similarity = db_g.search(emb, threshold=0.45)
            
        if name:
            cv2.rectangle(annotated, (bx, by), (bx+bw, by+bh), (0, 255, 0), 3)
            _, buffer = cv2.imencode('.jpg', annotated)
            return jsonify({
                "status": "identified", 
                "instruction": f"Verified: {name}", 
                "name": name, 
                "ratio": ratio,
                "image": base64.b64encode(buffer).decode('utf-8')
            })
        else:
            return jsonify({"status": "wait", "instruction": "Intruder Alert! Unknown Face.", "ratio": ratio})

    # ---------------------------------------------------------
    # PHASE 2: HAND GESTURE CONFIRMATION
    # ---------------------------------------------------------
    elif phase == "gesture":
        last_known_face = None # Stop drawing face box
        with ai_lock:
            boxes = palm_det.detect(frame)
            for box in boxes:
                lm_res = hand_lm.detect(frame, box)
                if lm_res:
                    # --- THE TOGGLE ---
                    if use_fallback:
                        # Brain 1: Pure Math Geometry
                        if is_victory_sign(lm_res["frame"]):
                            gesture = "Victory"
                        else:
                            gesture = "None"
                    else:
                        # Brain 2: TFLite Classifier Model
                        gesture, conf = gesture_cls.classify(lm_res["local"], lm_res["handedness"])
                    
                    last_known_hand = {"pts": lm_res["frame"], "gesture": gesture}
                    draw_hand_landmarks(annotated, lm_res["frame"], gesture)
                    
                    _, buffer = cv2.imencode('.jpg', annotated)
                    img_b64 = base64.b64encode(buffer).decode('utf-8')

                    if gesture == "Victory":
                        return jsonify({"status": "success", "instruction": "Access Granted! ✌️", "image": img_b64})
                    else:
                        return jsonify({"status": "wait_gesture", "instruction": "Almost done! Show 'Victory' sign.", "image": img_b64})
        
        # No hands found
        last_known_hand = None
        return jsonify({"status": "wait_gesture", "instruction": "Show 'Victory' sign to confirm."})


# =============================================================================
# HTML / JS UI
# =============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Factor Face & Gesture ID</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .glow-blue { box-shadow: 0 0 20px rgba(59, 130, 246, 0.3); }
        .glow-green { box-shadow: 0 0 20px rgba(16, 185, 129, 0.3); }
        .fade-in { animation: fadeIn 0.5s ease-in forwards; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    </style>
</head>
<body class="bg-gray-950 text-gray-100 font-sans min-h-screen flex flex-col items-center justify-center p-6 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-gray-900 via-gray-950 to-black relative">

    <div class="mb-8 text-center z-10">
        <h1 class="text-4xl md:text-5xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">
            Secure Terminal
        </h1>
        <p class="text-gray-400 mt-2 text-sm font-medium tracking-widest uppercase">Multi-Factor Biometrics</p>
    </div>

    <div class="w-full max-w-3xl bg-gray-900/60 backdrop-blur-xl border border-gray-800 rounded-3xl p-6 md:p-8 shadow-2xl flex flex-col items-center relative overflow-hidden z-10">
        
        <div class="absolute -top-32 -left-32 w-64 h-64 bg-blue-500/10 rounded-full blur-3xl"></div>
        <div class="absolute -bottom-32 -right-32 w-64 h-64 bg-emerald-500/10 rounded-full blur-3xl"></div>
        
        <div id="phaseBadge" class="relative z-10 mb-5 px-5 py-1.5 rounded-full text-xs font-bold tracking-widest uppercase transition-colors duration-300 ring-1" style="background-color: rgba(59,130,246,0.1); color: #60a5fa; ring-color: rgba(59,130,246,0.3);">
            Step 1: Face ID
        </div>

        <div class="relative w-full aspect-video rounded-2xl overflow-hidden bg-black border border-gray-700 shadow-inner flex items-center justify-center z-10 ring-4 ring-black/50">
            <img src="/video_feed" class="w-full h-full object-cover">
            <div class="absolute inset-0 pointer-events-none opacity-30">
                <div class="absolute top-4 left-4 w-8 h-8 border-t-2 border-l-2 border-white"></div>
                <div class="absolute top-4 right-4 w-8 h-8 border-t-2 border-r-2 border-white"></div>
                <div class="absolute bottom-4 left-4 w-8 h-8 border-b-2 border-l-2 border-white"></div>
                <div class="absolute bottom-4 right-4 w-8 h-8 border-b-2 border-r-2 border-white"></div>
            </div>
        </div>

        <div id="instruction" class="text-2xl font-bold h-10 flex items-center justify-center text-center transition-colors duration-300 mt-6 z-10" style="color: #60a5fa;">
            System Ready
        </div>

        <div id="barWrap" class="w-full mt-2 mb-6 z-10">
            <div class="h-2 w-full bg-gray-800 rounded-full overflow-hidden ring-1 ring-white/5">
                <div id="bar" class="h-full w-0 transition-all duration-300 ease-out" style="background-color: #3b82f6;"></div>
            </div>
        </div>

        <div class="w-full grid grid-cols-2 md:grid-cols-4 gap-3 mb-6 z-10">
            <div class="bg-gray-950/50 border border-gray-800 rounded-xl p-3 text-center">
                <div class="text-[10px] text-gray-500 uppercase tracking-widest mb-1">Identity</div>
                <div id="info-name" class="font-mono font-bold text-gray-300">--</div>
            </div>
            <div class="bg-gray-950/50 border border-gray-800 rounded-xl p-3 text-center">
                <div class="text-[10px] text-gray-500 uppercase tracking-widest mb-1">Proximity</div>
                <div id="info-dist" class="font-mono font-bold text-gray-300">--</div>
            </div>
            <div class="bg-gray-950/50 border border-gray-800 rounded-xl p-3 text-center">
                <div class="text-[10px] text-gray-500 uppercase tracking-widest mb-1">Auth Phase</div>
                <div id="info-phase" class="font-mono font-bold text-blue-400">Face</div>
            </div>
            <div class="bg-gray-950/50 border border-gray-800 rounded-xl p-3 text-center">
                <div class="text-[10px] text-gray-500 uppercase tracking-widest mb-1">Liveness</div>
                <div id="info-gesture" class="font-mono font-bold text-gray-300">Pending</div>
            </div>
        </div>

        <button id="btn" onclick="start()" class="z-10 w-full px-8 py-4 rounded-xl font-bold text-white text-lg bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 glow-blue transform transition-all duration-200 hover:-translate-y-1 active:translate-y-0 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none disabled:shadow-none">
            Initialize Login
        </button>
    </div>

    <div id="debugPanel" class="fixed bottom-6 right-6 w-[22rem] bg-gray-900/90 backdrop-blur-md border border-gray-700 rounded-2xl p-3 shadow-2xl z-50 hidden fade-in">
        <div class="flex items-center justify-between mb-2 px-1">
            <h3 class="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Capture Log</h3>
            <div class="w-2 h-2 rounded-full bg-red-500 animate-pulse"></div>
        </div>
        
        <div class="grid grid-cols-2 gap-2">
            
            <div class="flex flex-col">
                <div class="text-[9px] text-gray-500 text-center uppercase tracking-widest mb-1">1. Face</div>
                <div class="w-full aspect-[4/3] bg-black rounded-lg border border-gray-800 flex items-center justify-center overflow-hidden relative">
                    <span id="msgFace" class="text-[10px] text-gray-600 font-mono animate-pulse text-center">Waiting...</span>
                    <img id="snapFace" src="" class="hidden w-full h-full object-cover">
                </div>
            </div>

            <div class="flex flex-col">
                <div class="text-[9px] text-gray-500 text-center uppercase tracking-widest mb-1">2. Liveness</div>
                <div class="w-full aspect-[4/3] bg-black rounded-lg border border-gray-800 flex items-center justify-center overflow-hidden relative">
                    <span id="msgGesture" class="text-[10px] text-gray-600 font-mono animate-pulse text-center">Waiting...</span>
                    <img id="snapGesture" src="" class="hidden w-full h-full object-cover">
                </div>
            </div>
            
        </div>
    </div>

    <script>
        let active = false;
        let currentPhase = "face";
        
        const colors = {
            blue: "#3b82f6", blueLight: "#60a5fa",
            red: "#ef4444",
            orange: "#f97316", orangeLight: "#fb923c",
            green: "#10b981", greenLight: "#34d399",
            gray: "#9ca3af"
        };

        function setBadge(text, color, lightColor) {
            const badge = document.getElementById('phaseBadge');
            badge.innerText = text;
            badge.style.backgroundColor = `rgba(${hexToRgb(color)}, 0.1)`;
            badge.style.color = lightColor;
            badge.style.borderColor = `rgba(${hexToRgb(color)}, 0.3)`;
        }

        function hexToRgb(hex) {
            const bigint = parseInt(hex.replace('#', ''), 16);
            return `${(bigint >> 16) & 255}, ${(bigint >> 8) & 255}, ${bigint & 255}`;
        }

        function start() {
            active = true;
            currentPhase = "face";
            const btn = document.getElementById('btn');
            btn.innerText = "Scanning Environment...";
            btn.disabled = true;
            
            document.getElementById('debugPanel').style.display = 'block'; 
            document.getElementById('barWrap').style.display = "block";
            
            // Reset readouts
            document.getElementById('info-name').innerText = "--";
            document.getElementById('info-name').style.color = colors.gray;
            document.getElementById('info-gesture').innerText = "Pending";
            document.getElementById('info-gesture').style.color = colors.gray;
            document.getElementById('info-phase').innerText = "Face";
            document.getElementById('info-phase').style.color = colors.blueLight;

            // Reset images
            document.getElementById('snapFace').style.display = "none";
            document.getElementById('msgFace').style.display = "block";
            document.getElementById('snapGesture').style.display = "none";
            document.getElementById('msgGesture').style.display = "block";

            setBadge("Step 1: Face ID", colors.blue, colors.blueLight);
            loop();
        }

        function loop() {
            if(!active) return;
            
            fetch('/process', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ phase: currentPhase })
            })
            .then(r => r.json())
            .then(d => {
                if(d.status !== "no_frame") {
                    const inst = document.getElementById('instruction');
                    const bar = document.getElementById('bar');
                    inst.innerText = d.instruction;
                    
                    // --- FACE PHASE LOGIC ---
                    if (currentPhase === "face") {
                        // Change 0.60 back to whatever your max zoom limit is in your Python script
                        let p = Math.min(100, (d.ratio / 0.60) * 100); 
                        bar.style.width = p + "%";
                        
                        document.getElementById('info-dist').innerText = (d.ratio > 0 ? p.toFixed(0) + "%" : "--");
                        
                        if (d.instruction.includes("Intruder")) {
                            inst.style.color = colors.red;
                            bar.style.backgroundColor = colors.red;
                        } else {
                            inst.style.color = colors.blueLight;
                            bar.style.backgroundColor = colors.blue;
                        }

                        if(d.status === "identified") {
                            currentPhase = "gesture"; 
                            
                            // LOCK IN FACE SNAPSHOT
                            document.getElementById('snapFace').src = "data:image/jpeg;base64," + d.image;
                            document.getElementById('snapFace').style.display = "block";
                            document.getElementById('msgFace').style.display = "none";
                            
                            document.getElementById('info-name').innerText = d.name;
                            document.getElementById('info-name').style.color = colors.greenLight;
                            document.getElementById('info-phase').innerText = "Liveness";
                            document.getElementById('info-phase').style.color = colors.orangeLight;
                            
                            document.getElementById('barWrap').style.display = "none";
                            setBadge("Step 2: Liveness", colors.orange, colors.orangeLight);
                            inst.style.color = colors.orangeLight;
                        }
                    } 
                    
                    // --- GESTURE PHASE LOGIC ---
                    else if (currentPhase === "gesture") {
                        if (d.image) {
                            // STREAM GESTURE SNAPSHOT
                            document.getElementById('snapGesture').src = "data:image/jpeg;base64," + d.image;
                            document.getElementById('snapGesture').style.display = "block";
                            document.getElementById('msgGesture').style.display = "none";
                        }
                        
                        if(d.status === "success") {
                            active = false; 
                            inst.style.color = colors.greenLight;
                            setBadge("✅ Fully Unlocked", colors.green, colors.greenLight);
                            
                            document.getElementById('info-gesture').innerText = "Victory ✌️";
                            document.getElementById('info-gesture').style.color = colors.greenLight;
                            document.getElementById('info-phase').innerText = "Unlocked";
                            document.getElementById('info-phase').style.color = colors.greenLight;
                            
                            const btn = document.getElementById('btn');
                            btn.innerText = "Authentication Successful";
                            btn.className = "z-10 w-full px-8 py-4 rounded-xl font-bold text-white text-lg bg-gradient-to-r from-emerald-600 to-emerald-500 glow-green transform transition-all duration-200 disabled:opacity-100 disabled:cursor-default";
                        }
                    }
                    
                    if (active) setTimeout(loop, 500);
                }
            })
            .catch(() => { if(active) setTimeout(loop, 1000); });
        }
    </script>
</body>
</html>
"""

# =============================================================================
# Main
# =============================================================================
det_sess_g, cava_sess_g, anchors_g, db_g = None, None, None, None
palm_det, hand_lm, gesture_cls = None, None, None

def main():
    global det_sess_g, cava_sess_g, anchors_g, db_g
    global palm_det, hand_lm, gesture_cls, use_fallback

    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--datasets", default="", help="Path to face datasets")
    parser.add_argument("--detector", required=True, help="FaceDetector.onnx path")
    parser.add_argument("--cavaface", required=True, help="cavaface.onnx path")
    parser.add_argument("--palm", required=True, help="Palm TFLite path")
    parser.add_argument("--landmark", required=True, help="Landmark TFLite path")
    parser.add_argument("--gesture", required=True, help="Gesture TFLite path")
    parser.add_argument("--fallback", action="store_true", help="Use pure math geometry instead of Gesture AI")
    args = parser.parse_args()

    print("--- Loading AI Models ---")
    use_fallback = args.fallback

    # 1. Load ONNX Face Models
    det_sess_g = ort.InferenceSession(args.detector)
    cava_sess_g = ort.InferenceSession(args.cavaface)
    anchors_g = generate_anchors()
    db_g = FaceDB()
    if args.datasets: build_database_from_folder(args.datasets)

    # 2. Load TFLite Hand Models
    palm_det = PalmDetectorTFLite(args.palm)
    hand_lm = HandLandmarkTFLite(args.landmark)
    
    # Only load gesture AI if we aren't using the fallback math
    if not use_fallback:
        try: 
            gesture_cls = GestureClassifierTFLite(args.gesture)
        except Exception as e: 
            print(f"Warning: Gesture model failed to load. Forcing fallback mode. Error: {e}")
            use_fallback = True

    print(f"--- Starting Server (Fallback Math: {use_fallback}) ---")
    threading.Thread(target=camera_thread, args=(args.camera,), daemon=True).start()
    app.run(host="0.0.0.0", port=5001, threaded=True)

if __name__ == "__main__":
    main()