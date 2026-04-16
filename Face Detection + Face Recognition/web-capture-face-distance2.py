import cv2
import numpy as np
import base64
import threading
import time
import argparse
from flask import Flask, Response, render_template_string, jsonify

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter

# =============================================================================
# Global Protection
# =============================================================================
shared_frame_lock = threading.Lock()
ai_lock = threading.Lock()
shared_frame = None

# =============================================================================
# Face Detector Class (Qualcomm Grid Version)
# =============================================================================

class FaceDetectorTFLite:
    def __init__(self, model_path, score_threshold=0.6):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Read exact dimensions from the model
        shape = self.input_details[0]['shape']
        self.input_height = shape[1] 
        self.input_width = shape[2]
        self.score_threshold = score_threshold
        print(f"--- Loaded Model: Expected {self.input_width}x{self.input_height} ---")

    def detect(self, frame):
        with ai_lock:
            h, w = frame.shape[:2]
            
            # 1. Grayscale Conversion & Resizing
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(img_gray, (self.input_width, self.input_height))
            img_resized = np.expand_dims(img_resized, axis=-1) 
            input_data = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)

            # 2. Run AI
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()

            # 3. Parse Grid Outputs
            boxes = None
            scores = None
            
            for detail in self.output_details:
                data = self.interpreter.get_tensor(detail['index']).copy()
                if len(data.shape) == 4:
                    flat = data.reshape(-1, data.shape[-1])
                    if flat.shape[-1] == 4:
                        boxes = flat
                    elif flat.shape[-1] == 1:
                        scores = flat.flatten()

            if boxes is None or scores is None: return None

            # 4. Find the best face
            best_idx = np.argmax(scores)
            best_score = scores[best_idx]
            
            # Convert raw model score to 0.0-1.0 probability
            if best_score > 1.0 or best_score < 0.0:
                best_score = 1.0 / (1.0 + np.exp(-np.clip(best_score, -100, 100)))

            if best_score > self.score_threshold:
                # Find the center of the face using the Grid Index (60x80 grid)
                grid_y = best_idx // 80
                grid_x = best_idx % 80
                
                cx_norm = grid_x / 80.0
                cy_norm = grid_y / 60.0

                # Use the raw box values for width/height in "grid tiles"
                # v2 and v3 act as height and width offsets
                raw_h, raw_w = boxes[best_idx][2], boxes[best_idx][3]
                
                # Approximate width and height normalized to 0.0-1.0
                w_norm = raw_w / 80.0
                h_norm = raw_h / 60.0

                # Calculate standard pixel coordinates
                px = int((cx_norm - w_norm/2) * w)
                py = int((cy_norm - h_norm/2) * h)
                pw = int(w_norm * w)
                ph = int(h_norm * h)

                return {
                    "box": [max(0, px), max(0, py), pw, ph],
                    "ratio": w_norm * h_norm, # Percentage of screen taken
                    "score": float(best_score)
                }
            return None

# =============================================================================
# Flask App Logic
# =============================================================================

app = Flask(__name__)
face_detector = None

def camera_thread():
    global shared_frame, face_detector
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            
            # Live Blue Box for debugging
            res = face_detector.detect(frame)
            if res:
                bx, by, bw, bh = res["box"]
                cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (255, 0, 0), 2)
            
            with shared_frame_lock:
                shared_frame = frame.copy()
        else:
            time.sleep(0.1)

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
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + bytearray(encoded) + b"\r\n")
            time.sleep(0.03)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/check_proximity", methods=["POST"])
def check_proximity():
    with shared_frame_lock:
        if shared_frame is None: return jsonify({"status": "no_frame"})
        frame = shared_frame.copy()

    res = face_detector.detect(frame)
    if not res:
        return jsonify({"status": "no_face", "instruction": "Searching for Face...", "ratio": 0})

    bx, by, bw, bh = res["box"]
    ratio = res["ratio"]

    # --- PROXIMITY LOGIC ---
    status = "wait"
    if ratio < 0.05: # Adjusted for Grid scaling
        instruction = "Move Closer"
    elif ratio > 0.18:
        instruction = "Too Close! Move Back"
    else:
        instruction = "Perfect! Scanning..."
        status = "success"

    color = (0, 255, 0) if status == "success" else (0, 165, 255)
    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), color, 3)
    
    _, buffer = cv2.imencode('.jpg', frame)
    img_b64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({"status": status, "instruction": instruction, "ratio": float(ratio), "image": img_b64})

# =============================================================================
# UI Design
# =============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Face ID Proximity Check</title>
    <style>
        body { background: #0e0e12; color: #fff; font-family: 'Segoe UI', sans-serif; text-align: center; }
        .wrap { display: flex; justify-content: center; gap: 30px; padding: 40px; }
        .panel { background: #1c1c24; padding: 20px; border-radius: 15px; border: 1px solid #333; width: 500px; }
        img { width: 100%; border-radius: 10px; border: 2px solid #222; }
        #instruction { font-size: 1.8rem; font-weight: bold; margin: 20px 0; color: #00d4ff; min-height: 50px; }
        .bar-container { background: #222; height: 12px; border-radius: 6px; overflow: hidden; margin: 10px 0; }
        #bar { background: #00ff88; height: 100%; width: 0%; transition: 0.2s; }
        button { background: #00d4ff; color: #000; border: none; padding: 15px 40px; border-radius: 30px; font-size: 1.1rem; font-weight: bold; cursor: pointer; transition: 0.2s; }
        button:hover { background: #00ff88; transform: scale(1.05); }
        button:disabled { background: #444; color: #888; transform: none; }
    </style>
</head>
<body>
    <h1 style="margin-top:30px;">👤 Face ID: Step 1 (Proximity)</h1>
    <div class="wrap">
        <div class="panel">
            <h3>Live Guidance</h3>
            <img src="/video_feed">
            <div id="instruction">Ready</div>
            <div class="bar-container"><div id="bar"></div></div>
            <br>
            <button id="btn" onclick="start()">Start Enrollment</button>
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
                    let p = Math.min(100, (d.ratio / 0.18) * 100);
                    document.getElementById('bar').style.width = p + "%";
                    if(d.status === "success") {
                        active = false;
                        document.getElementById('snap').src = "data:image/jpeg;base64," + d.image;
                        document.getElementById('snap').style.display = "block";
                        document.getElementById('msg').style.display = "none";
                        document.getElementById('btn').disabled = false;
                    } else {
                        setTimeout(loop, 250);
                    }
                }
            });
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="../models/face_det_lite-lightweight-face-detection-float.tflite")
    args = parser.parse_args()

    face_detector = FaceDetectorTFLite(args.model)
    threading.Thread(target=camera_thread, daemon=True).start()
    app.run(host="0.0.0.0", port=5002, threaded=True)