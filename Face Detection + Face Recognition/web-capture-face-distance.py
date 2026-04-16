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
# Face Detector Class
# =============================================================================

class FaceDetectorTFLite:
    def __init__(self, model_path, score_threshold=0.7):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size = self.input_details[0]['shape'][1]
        self.score_threshold = score_threshold
        
    def detect(self, frame):
        with ai_lock:
            h, w = frame.shape[:2]
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.input_size, self.input_size))
            
            # --- 1. Qualcomm UINT8 Handling ---
            input_dtype = self.input_details[0]['dtype']
            if input_dtype == np.uint8:
                input_data = np.expand_dims(img.astype(np.uint8), axis=0)
            else:
                input_data = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)

            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()

            # --- 2. Smart Output Mapping ---
            # Qualcomm models often have multiple outputs for boxes/scores/landmarks.
            # We look for the one with the most data (boxes) and the one with 1 channel (scores).
            box_data = None
            score_data = None
            
            for i in range(len(self.output_details)):
                detail = self.output_details[i]
                data = self.interpreter.get_tensor(detail['index']).copy()
                
                # Dequantize UINT8 output to Float
                if detail['dtype'] == np.uint8:
                    scale, zp = detail['quantization']
                    data = (data.astype(np.float32) - zp) * scale
                
                shape = data.shape
                if len(shape) == 3 and shape[2] > 4: # Likely the boxes + landmarks (usually 16)
                    box_data = data.reshape(-1, shape[2])
                elif len(shape) == 3 and shape[2] == 1: # Likely the scores
                    score_data = data.flatten()
                elif len(shape) == 2: # Alternative flat score output
                    score_data = data.flatten()

            if box_data is None or score_data is None:
                return None

            # --- 3. Parsing Results ---
            best_face = None
            max_s = -1

            for i in range(len(score_data)):
                # Convert logit to probability
                prob = 1.0 / (1.0 + np.exp(-np.clip(score_data[i], -100, 100)))
                
                if prob > self.score_threshold and prob > max_s:
                    max_s = prob
                    # Qualcomm MediaPipe typically uses [ymin, xmin, ymax, xmax]
                    ymin, xmin, ymax, xmax = box_data[i][:4]
                    
                    # Basic clamping
                    ymin, xmin = max(0, ymin), max(0, xmin)
                    ymax, xmax = min(1, ymax), min(1, xmax)

                    best_face = {
                        "box": [int(xmin * w), int(ymin * h), int((xmax - xmin) * w), int((ymax - ymin) * h)],
                        "score": float(prob)
                    }
            return best_face

# =============================================================================
# Flask App Logic
# =============================================================================

app = Flask(__name__)
face_detector = None

def camera_thread():
    global shared_frame, face_detector
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            res = face_detector.detect(frame)
            if res:
                bx, by, bw, bh = res["box"]
                # Blue box for live tracking
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
        return jsonify({"status": "no_face", "instruction": "Face Not Detected", "ratio": 0})

    bx, by, bw, bh = res["box"]
    h, w = frame.shape[:2]
    ratio = (bw * bh) / (w * h)

    status = "wait"
    if ratio < 0.12:
        instruction = "Move Closer"
    elif ratio > 0.35: # Tightened this to 35%
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
                    let p = Math.min(100, (d.ratio / 0.35) * 100);
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
    parser.add_argument("--model", default="../models/mediapipe_face-face_detector-w8a8.tflite")
    args = parser.parse_args()

    # Higher threshold to stop background hallucinations
    face_detector = FaceDetectorTFLite(args.model, score_threshold=0.75)
    
    threading.Thread(target=camera_thread, daemon=True).start()
    app.run(host="0.0.0.0", port=5002, threaded=True)
