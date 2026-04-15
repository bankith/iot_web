"""
web-capture-gesture.py — "Photo Booth" Hand Gesture Recognition

Streams live webcam video to a Flask UI. 
When the user clicks "Capture", it grabs a single frame, 
runs the TFLite hand gesture pipeline, and returns the annotated image.
"""

import sys
import os
import cv2
import numpy as np
import time
import threading
import argparse
import base64
from flask import Flask, Response, render_template_string, jsonify

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# =============================================================================
# Constants & Helpers
# =============================================================================

GESTURE_LABELS = ["None", "Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"]
HAND_CONNECTIONS = [(0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8), (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16), (0,17), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17)]

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

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

def _generate_palm_anchors(input_size=256):
    strides = [8, 16]
    anchors = []
    for stride in strides:
        grid_h = grid_w = input_size // stride
        for y in range(grid_h):
            for x in range(grid_w):
                cx, cy = (x + 0.5) * stride / input_size, (y + 0.5) * stride / input_size
                anchors.extend([[cx, cy], [cx, cy]])
    return np.array(anchors, dtype=np.float32)

# =============================================================================
# ML Models
# =============================================================================

class PalmDetectorTFLite:
    def __init__(self, model_path, input_size=256, score_threshold=0.65, nms_threshold=0.3):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size = input_size
        self.input_dtype = self.input_details[0]['dtype']
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.anchors = _generate_palm_anchors(input_size)
        self._score_output_idx, self._coord_output_idx = 0, 1
        for i, detail in enumerate(self.output_details):
            name = detail["name"].lower()
            if "score" in name: self._score_output_idx = i
            elif "coord" in name or "box" in name or "regress" in name: self._coord_output_idx = i

    def preprocess(self, frame):
        self.orig_h, self.orig_w = frame.shape[:2]
        img = cv2.cvtColor(cv2.resize(frame, (self.input_size, self.input_size)), cv2.COLOR_BGR2RGB)
        img = img.astype(np.uint8) if self.input_dtype == np.uint8 else img.astype(np.float32) / 255.0
        self._input_data = np.expand_dims(img, axis=0)

    def execute(self):
        try:
            self.interpreter.set_tensor(self.input_details[0]["index"], self._input_data)
            self.interpreter.invoke()
            return True
        except: return False

    def postprocess(self):
        raw_scores = _dequantize(self.interpreter.get_tensor(self.output_details[self._score_output_idx]["index"]), self.output_details[self._score_output_idx])
        raw_coords = _dequantize(self.interpreter.get_tensor(self.output_details[self._coord_output_idx]["index"]), self.output_details[self._coord_output_idx])

        n_anchors = self.anchors.shape[0]
        scores = 1.0 / (1.0 + np.exp(-np.clip(raw_scores.flatten()[:n_anchors], -100, 100)))
        coords = raw_coords.reshape(-1, 18)[:n_anchors]
        
        decoded = coords.copy()
        sz = float(self.input_size)
        anchor_cx, anchor_cy = self.anchors[:n_anchors, 0] * sz, self.anchors[:n_anchors, 1] * sz
        decoded[:, 0] += anchor_cx
        decoded[:, 1] += anchor_cy
        for k in range(7):
            decoded[:, 4+k*2] += anchor_cx
            decoded[:, 4+k*2+1] += anchor_cy

        mask = scores > self.score_threshold
        scores, decoded = scores[mask], decoded[mask]
        if len(scores) == 0: return []

        cx, cy, w, h = decoded[:,0], decoded[:,1], decoded[:,2], decoded[:,3]
        boxes_xyxy = np.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], axis=1)

        x1, y1, x2, y2 = boxes_xyxy[:,0], boxes_xyxy[:,1], boxes_xyxy[:,2], boxes_xyxy[:,3]
        areas = (x2-x1)*(y2-y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
            xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, xx2-xx1) * np.maximum(0.0, yy2-yy1)
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[np.where(ovr <= self.nms_threshold)[0] + 1]

        sx, sy = self.orig_w / self.input_size, self.orig_h / self.input_size
        return [{"bbox": [boxes_xyxy[i,0]*sx, boxes_xyxy[i,1]*sy, boxes_xyxy[i,2]*sx, boxes_xyxy[i,3]*sy], "score": float(scores[i])} for i in keep]

class HandLandmarkTFLite:
    def __init__(self, model_path, input_size=224):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size = input_size
        self.input_dtype = self.input_details[0]['dtype']
        self._out_map = {k: i for i, d in enumerate(self.output_details) for k in ["landmarks", "scores", "lr", "world_landmarks"] if k[:4] in d["name"].lower()}
        if len(self._out_map) < 4: self._out_map = {"landmarks": 0, "scores": 1, "lr": 2, "world_landmarks": 3}

    def preprocess(self, frame, bbox):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        # FIXED PADDING: 1.0 width, 1.5 height to capture fingers!
        pad_x, pad_y = bw * 1.0, bh * 1.5 
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        
        self.crop_bbox = [int(x1), int(y1), int(x2), int(y2)]
        crop = frame[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0: return False

        img = cv2.cvtColor(cv2.resize(crop, (self.input_size, self.input_size)), cv2.COLOR_BGR2RGB)
        self._input_data = np.expand_dims(img.astype(np.uint8) if self.input_dtype == np.uint8 else img.astype(np.float32)/255.0, axis=0)
        return True

    def execute(self):
        try:
            self.interpreter.set_tensor(self.input_details[0]["index"], self._input_data)
            self.interpreter.invoke()
            return True
        except: return False

    def postprocess(self):
        lm_raw = _dequantize(self.interpreter.get_tensor(self.output_details[self._out_map["landmarks"]]["index"]), self.output_details[self._out_map["landmarks"]])
        sc_raw = _dequantize(self.interpreter.get_tensor(self.output_details[self._out_map["scores"]]["index"]), self.output_details[self._out_map["scores"]])
        lr_raw = _dequantize(self.interpreter.get_tensor(self.output_details[self._out_map["lr"]]["index"]), self.output_details[self._out_map["lr"]])

        landmarks_local = lm_raw.flatten()[:63].reshape(21, 3)
        score = float(1.0 / (1.0 + np.exp(-np.clip(float(sc_raw.flatten()[0]), -100, 100))))
        handedness = float(1.0 / (1.0 + np.exp(-np.clip(float(lr_raw.flatten()[0]), -100, 100))))

        cx1, cy1, cx2, cy2 = self.crop_bbox
        landmarks_frame = landmarks_local.copy()
        landmarks_frame[:, 0] = landmarks_local[:, 0] / self.input_size * (cx2 - cx1) + cx1
        landmarks_frame[:, 1] = landmarks_local[:, 1] / self.input_size * (cy2 - cy1) + cy1

        return {"landmarks_local": landmarks_local, "landmarks_frame": landmarks_frame, "score": score, "handedness": handedness}

class GestureClassifierTFLite:
    def __init__(self, model_path):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def preprocess(self, landmarks_local, handedness):
        pts = landmarks_local.copy().astype(np.float32)
        normed = pts - pts[[0, 1, 5, 9, 13, 17]].mean(axis=0)
        normed /= max(normed[:, 0].max() - normed[:, 0].min(), normed[:, 1].max() - normed[:, 1].min()) + 1e-5
        input_data = np.append(normed.flatten(), np.float32(handedness))
        self._input_data = np.expand_dims(_quantize(input_data, self.input_details[0]), axis=0)

    def execute(self):
        try:
            self.interpreter.set_tensor(self.input_details[0]["index"], self._input_data)
            self.interpreter.invoke()
            return True
        except: return False

    def postprocess(self):
        logits = _dequantize(self.interpreter.get_tensor(self.output_details[0]["index"]), self.output_details[0]).flatten()[:8]
        probs = np.exp(logits - logits.max()) / (np.exp(logits - logits.max()).sum() + 1e-8)
        idx = int(np.argmax(probs))
        return {"gesture": GESTURE_LABELS[idx], "confidence": float(probs[idx])}

def count_fingers_geometric(landmarks):
    if landmarks is None or len(landmarks) < 21: return "None", 0.0
    finger_count = 0
    is_right = landmarks[WRIST][0] < landmarks[MIDDLE_MCP][0]
    if (is_right and landmarks[THUMB_TIP][0] < landmarks[THUMB_IP][0]) or (not is_right and landmarks[THUMB_TIP][0] > landmarks[THUMB_IP][0]):
        finger_count += 1
    for tip_idx, pip_idx in zip([INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP], [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]):
        if landmarks[tip_idx][1] < landmarks[pip_idx][1]: finger_count += 1
    return {0: "Closed_Fist", 1: "Pointing_Up", 2: "Victory", 3: "Open_Palm", 4: "Open_Palm", 5: "Open_Palm"}.get(finger_count, "None"), 0.8

def draw_hand_landmarks(frame, landmarks, gesture_label="", confidence=0.0):
    if landmarks is None: return frame
    for i, j in HAND_CONNECTIONS:
        cv2.line(frame, (int(landmarks[i][0]), int(landmarks[i][1])), (int(landmarks[j][0]), int(landmarks[j][1])), FINGER_COLORS.get(LANDMARK_FINGER_MAP.get(i, "palm"), (200, 200, 200)), 2, cv2.LINE_AA)
    for idx in range(21):
        x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
        cv2.circle(frame, (x, y), 5, FINGER_COLORS.get(LANDMARK_FINGER_MAP.get(idx, "palm"), (200, 200, 200)), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 5, (0, 0, 0), 1, cv2.LINE_AA)
    if gesture_label and gesture_label != "None":
        label = f"{gesture_label} {confidence * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(frame, (10, 10), (20 + tw, 20 + th + 10), (0, 0, 0), -1)
        cv2.putText(frame, label, (15, 20 + th), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


# =============================================================================
# Flask Web App
# =============================================================================

app = Flask(__name__)
shared_frame = None
lock = threading.Lock()
palm_det, hand_lm, gesture_cls = None, None, None
use_fallback = False

def camera_thread(camera_id):
    global shared_frame
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {camera_id}")
        return
        
    print("Webcam opened successfully!")
    while True:
        ret, frame = cap.read()
        if ret:
            # MIRROR FIX: Flip the frame so it acts like a mirror
            frame = cv2.flip(frame, 1)
            # Quickly grab the lock, update the frame, and release it
            with lock:
                shared_frame = frame.copy()
        else:
            time.sleep(0.1)

def generate_video_feed():
    global shared_frame
    while True:
        # 1. Grab the frame quickly and release the lock immediately
        with lock:
            frame_copy = shared_frame.copy() if shared_frame is not None else None
            
        # 2. If no frame yet, sleep OUTSIDE the lock so the camera can work
        if frame_copy is None:
            time.sleep(0.1)
            continue
            
        # 3. Encode and send the image
        ok, encoded = cv2.imencode(".jpg", frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + bytearray(encoded) + b"\r\n")
        time.sleep(0.033)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Gesture Photo Booth</title>
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Segoe UI', sans-serif; }
    body { background: #0a0a0a; color: #e0e0e0; min-height: 100vh; display: flex; flex-direction: column; align-items: center; }
    .header { width: 100%; padding: 18px 24px; background: linear-gradient(135deg, #1a1a2e, #16213e); text-align: center; border-bottom: 1px solid #2a2a4a; }
    .main { display: flex; flex-wrap: wrap; gap: 20px; padding: 20px; max-width: 1200px; width: 100%; justify-content: center; }
    .panel { background: #111; border: 1px solid #2a2a4a; border-radius: 12px; overflow: hidden; display: flex; flex-direction: column; align-items: center; width: 48%; min-width: 320px; padding-bottom: 15px;}
    .panel h2 { background: #1a1a2e; width: 100%; padding: 10px; text-align: center; font-size: 1.1rem; border-bottom: 1px solid #2a2a4a; margin-bottom: 10px; }
    img { max-width: 95%; border-radius: 8px; display: block; }
    #result_img { min-height: 240px; background: #000; border: 1px dashed #333; display: flex; align-items: center; justify-content: center; color: #555; }
    
    /* Button Styles */
    .btn-container { display: flex; gap: 10px; margin-top: 15px; flex-wrap: wrap; justify-content: center;}
    button { color: white; border: none; padding: 12px 20px; font-size: 1rem; font-weight: bold; border-radius: 30px; cursor: pointer; transition: 0.2s; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    #snapBtn { background: linear-gradient(135deg, #4b6cb7, #182848); }
    #challengeBtn { background: linear-gradient(135deg, #ff9966, #ff5e62); box-shadow: 0 4px 15px rgba(255, 94, 98, 0.4); }
    
    .info-card { background: #1a1a2e; border: 1px solid #2a2a4a; border-radius: 10px; padding: 15px; text-align: center; width: 95%; margin-top: 15px;}
    .info-card .label { font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .info-card .value { font-size: 2rem; font-weight: bold; color: #00ff88; margin-top: 5px; }
    
    #instructionText { margin-top: 15px; font-size: 1.2rem; font-weight: bold; color: #ff9966; height: 30px; }
</style>
</head>
<body>
    <div class="header"><h2>📸 Hand Gesture Photo Booth</h2></div>
    <div class="main">
        <div class="panel">
            <h2>Live Camera</h2>
            <img src="/video_feed" alt="Video Stream">
            
            <div id="instructionText"></div>

            <div class="btn-container">
                <button onclick="takeSnapshot()" id="snapBtn">📸 Manual Capture</button>
                <button onclick="startChallenge()" id="challengeBtn">✌️ Auto-Detect "Victory"</button>
            </div>
        </div>
        <div class="panel">
            <h2>Snapshot Result</h2>
            <img id="result_img" alt="Captured Result">
            <div class="info-card">
                <div class="label">Detected Gesture</div>
                <div class="value" id="gesture_val">—</div>
            </div>
        </div>
    </div>
    <script>
        let isSearching = false;

        // Standard Manual Capture
        function takeSnapshot() {
            fetchAndDisplay();
        }

        // Start the automated challenge
        function startChallenge() {
            isSearching = true;
            document.getElementById('instructionText').innerText = "👀 Secretly scanning... Show a Peace Sign (Victory)!";
            document.getElementById('instructionText').style.color = "#ff9966";
            document.getElementById('snapBtn').disabled = true;
            document.getElementById('challengeBtn').disabled = true;
            
            autoCaptureLoop();
        }

        // Recursive loop that takes a photo in the background
        function autoCaptureLoop() {
            if (!isSearching) return;

            fetch('/capture', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    if (!data.error) {
                        // Update the screen with what the AI just saw
                        document.getElementById('result_img').src = "data:image/jpeg;base64," + data.image;
                        document.getElementById('gesture_val').innerText = data.gesture !== "None" ? `${data.gesture} (${(data.confidence*100).toFixed(0)}%)` : "None";
                        document.getElementById('gesture_val').style.color = data.gesture !== "None" ? "#00ff88" : "#666";

                        // DID WE WIN?
                        if (data.gesture === "Victory") {
                            isSearching = false; // Stop the loop
                            document.getElementById('instructionText').innerText = "✅ PASS! Victory Detected!";
                            document.getElementById('instructionText').style.color = "#00ff88";
                            document.getElementById('snapBtn').disabled = false;
                            document.getElementById('challengeBtn').disabled = false;
                            return; // Exit the loop entirely
                        }
                    }
                    
                    // If we didn't win, wait 500 milliseconds and take another secret picture
                    if (isSearching) {
                        setTimeout(autoCaptureLoop, 500);
                    }
                })
                .catch(err => {
                    // If there is a network glitch, just try again in a second
                    if (isSearching) setTimeout(autoCaptureLoop, 500);
                });
        }

        // Helper function for the manual button
        function fetchAndDisplay() {
            const btn = document.getElementById('snapBtn');
            btn.innerText = "Processing...";
            fetch('/capture', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    document.getElementById('result_img').src = "data:image/jpeg;base64," + data.image;
                    document.getElementById('gesture_val').innerText = data.gesture !== "None" ? `${data.gesture} (${(data.confidence*100).toFixed(0)}%)` : "None";
                    document.getElementById('gesture_val').style.color = data.gesture !== "None" ? "#00ff88" : "#666";
                })
                .finally(() => { btn.innerText = "📸 Manual Capture"; });
        }
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/video_feed")
def video_feed():
    return Response(generate_video_feed(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/capture", methods=["POST"])
def capture():
    global shared_frame, palm_det, hand_lm, gesture_cls, use_fallback
    with lock:
        if shared_frame is None:
            return jsonify({"error": "No camera frame"}), 500
        frame = shared_frame.copy()

    annotated = frame.copy()
    best_gesture, best_conf = "None", 0.0

    palm_det.preprocess(frame)
    if palm_det.execute():
        detections = palm_det.postprocess()
        for det in detections:
            if not hand_lm.preprocess(frame, det["bbox"]) or not hand_lm.execute():
                continue
            lm_result = hand_lm.postprocess()
            if lm_result["score"] < 0.7:
                continue

            g_label, g_conf = "None", 0.0
            if use_fallback:
                g_label, g_conf = count_fingers_geometric(lm_result["landmarks_frame"])
            else:
                gesture_cls.preprocess(lm_result["landmarks_local"], lm_result["handedness"])
                if gesture_cls.execute():
                    g_res = gesture_cls.postprocess()
                    g_label, g_conf = g_res["gesture"], g_res["confidence"]
                else:
                    g_label, g_conf = count_fingers_geometric(lm_result["landmarks_frame"])

            if g_conf > best_conf:
                best_gesture, best_conf = g_label, g_conf
            annotated = draw_hand_landmarks(annotated, lm_result["landmarks_frame"], g_label, g_conf)

    _, buffer = cv2.imencode('.jpg', annotated)
    b64_img = base64.b64encode(buffer).decode('utf-8')
    return jsonify({"image": b64_img, "gesture": best_gesture, "confidence": best_conf})


# =============================================================================
# Main
# =============================================================================

def main():
    global palm_det, hand_lm, gesture_cls, use_fallback
    parser = argparse.ArgumentParser()    
    parser.add_argument("--palm-model", default="../models/mediapipe_hand_gesture-palm_detector-w8a8.tflite", help="Path to palm model")
    parser.add_argument("--landmark-model", default="../models/mediapipe_hand_gesture-hand_landmark_detector-w8a8.tflite", help="Path to landmark model")
    parser.add_argument("--gesture-model", default="../models/mediapipe_hand_gesture-canned_gesture_classifier-w8a8.tflite", help="Path to gesture model")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--fallback", action="store_true", help="Use fallback math")
    args = parser.parse_args()

    print("Initialising models...")
    palm_det = PalmDetectorTFLite(model_path=args.palm_model)
    hand_lm = HandLandmarkTFLite(model_path=args.landmark_model)
    use_fallback = args.fallback
    if not use_fallback:
        try: gesture_cls = GestureClassifierTFLite(model_path=args.gesture_model)
        except: use_fallback = True
    
    print("Starting camera thread...")
    threading.Thread(target=camera_thread, args=(args.camera,), daemon=True).start()

    print("\nServer running at http://localhost:5002\nPress Ctrl+C to quit.")
    app.run(host="0.0.0.0", port=5002, threaded=True, debug=False)

if __name__ == "__main__":
    main()