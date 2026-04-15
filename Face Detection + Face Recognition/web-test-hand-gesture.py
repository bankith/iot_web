"""
web-test-hand-gesture.py — Hand Gesture Recognition (SNPE / DSP)

Uses three MediaPipe DLC models on Qualcomm QCS6490 via SNPE:
  1. Palm Detector        — detects palm bounding boxes
  2. Hand Landmark        — regresses 21 3-D hand landmarks
  3. Gesture Classifier   — classifies gesture from landmarks

Usage:
    cd "Face Detection + Face Recognition"
    source setup.sh
    python web-test-hand-gesture.py --runtime DSP

    Then open: http://localhost:5002
"""

import sys
import os
import cv2
import numpy as np
import time
import threading
import argparse
from flask import Flask, Response, render_template_string, jsonify

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from snpehelper_manager import PerfProfile, Runtime, SnpeContext


# =============================================================================
# Constants
# =============================================================================

GESTURE_LABELS = [
    "None",
    "Closed_Fist",
    "Open_Palm",
    "Pointing_Up",
    "Thumb_Down",
    "Thumb_Up",
    "Victory",
    "ILoveYou",
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),               # palm
]

# Finger landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# Finger group colours for drawing (BGR)
FINGER_COLORS = {
    "thumb":  (0, 255, 255),    # yellow
    "index":  (0, 165, 255),    # orange
    "middle": (0, 255, 0),      # green
    "ring":   (255, 0, 0),      # blue
    "pinky":  (255, 0, 255),    # magenta
    "palm":   (200, 200, 200),  # grey
}

LANDMARK_FINGER_MAP = {
    0: "palm",
    1: "thumb", 2: "thumb", 3: "thumb", 4: "thumb",
    5: "index", 6: "index", 7: "index", 8: "index",
    9: "middle", 10: "middle", 11: "middle", 12: "middle",
    13: "ring", 14: "ring", 15: "ring", 16: "ring",
    17: "pinky", 18: "pinky", 19: "pinky", 20: "pinky",
}


# =============================================================================
# Anchor generation for MediaPipe Palm Detector (256 × 256)
# =============================================================================

def _generate_palm_anchors(input_size=256):
    """
    Generate SSD-style anchors for the MediaPipe palm detector.
    The model uses strides [8, 16, 16, 16] with 2 anchors per location
    at the first two feature maps and 6 anchors at the last two.

    However, the Qualcomm AI Hub variant uses a simplified architecture
    with strides [8, 16] and 1 anchor per position. The DLC outputs
    box_coords of shape [N, 18] and box_scores of shape [N] where N
    is the total number of anchors.

    Based on the Qualcomm AI Hub implementation, the palm detector uses
    strides of [8, 16] with 2 anchors each at each location.
    """
    strides = [8, 16]
    anchors = []
    for stride in strides:
        grid_h = input_size // stride
        grid_w = input_size // stride
        for y in range(grid_h):
            for x in range(grid_w):
                # 2 anchors per location
                cx = (x + 0.5) * stride / input_size
                cy = (y + 0.5) * stride / input_size
                anchors.append([cx, cy])
                anchors.append([cx, cy])
    return np.array(anchors, dtype=np.float32)


# =============================================================================
# Palm Detector (SNPE)
# =============================================================================

class PalmDetectorSNPE(SnpeContext):
    """
    MediaPipe Palm Detector via SNPE.

    Input:  'image' — 1 × 3 × 256 × 256 (NHWC in SNPE) RGB float32 [0,1]
    Output: 'box_coords' — [N, 18]  (cx, cy, w, h, 7 keypoints × 2)
            'box_scores' — [N, 1]   (raw logit)
    """

    def __init__(
        self,
        dlc_path="None",
        input_layers=None,
        output_layers=None,
        output_tensors=None,
        runtime=Runtime.CPU,
        profile_level=PerfProfile.BALANCED,
        enable_cache=False,
        input_size=256,
        score_threshold=0.5,
        nms_threshold=0.3,
    ):
        if input_layers is None:
            input_layers = ["image"]
        if output_layers is None:
            output_layers = ["box_scores", "box_coords"]
        if output_tensors is None:
            output_tensors = ["box_scores", "box_coords"]

        super().__init__(
            dlc_path,
            input_layers,
            output_layers,
            output_tensors,
            runtime,
            profile_level,
            enable_cache,
        )
        self.input_size = input_size
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.anchors = _generate_palm_anchors(input_size)

    def preprocess(self, frame):
        """Resize, convert BGR→RGB, normalise to [0,1], flatten for SNPE."""
        self.orig_h, self.orig_w = frame.shape[:2]
        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        self.SetInputBuffer(img.flatten(), "image")

    def _decode_boxes(self, raw_coords):
        """
        Decode raw SSD-style predictions using anchors.
        raw_coords: [N, 18] — (cx_off, cy_off, w, h, kp0_x, kp0_y, ..., kp6_x, kp6_y)
        Returns decoded coords in pixel space [N, 18].
        """
        n = raw_coords.shape[0]
        decoded = raw_coords.copy()
        sz = float(self.input_size)

        anchor_cx = self.anchors[:n, 0] * sz
        anchor_cy = self.anchors[:n, 1] * sz

        # Centre offsets are relative to anchor
        decoded[:, 0] = raw_coords[:, 0] + anchor_cx  # cx
        decoded[:, 1] = raw_coords[:, 1] + anchor_cy  # cy
        # w, h stay as-is (already in pixel space relative to input_size)

        # Keypoints (indices 4..17, pairs of x,y)
        for k in range(7):
            decoded[:, 4 + k * 2] = raw_coords[:, 4 + k * 2] + anchor_cx
            decoded[:, 4 + k * 2 + 1] = raw_coords[:, 4 + k * 2 + 1] + anchor_cy

        return decoded

    def _nms(self, boxes_xyxy, scores):
        """Simple NMS on xyxy boxes."""
        x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(ovr <= self.nms_threshold)[0]
            order = order[inds + 1]
        return keep

    def postprocess(self):
        """
        Decode palm detector outputs into a list of bounding boxes.
        Returns list of dicts: {'bbox': [x1,y1,x2,y2], 'score': float,
                                 'keypoints': [[x,y], ...]}
        Coordinates are in original frame pixel space.
        """
        raw_scores = self.GetOutputBuffer("box_scores")
        raw_coords = self.GetOutputBuffer("box_coords")

        # Reshape — the model outputs flattened tensors
        n_anchors = self.anchors.shape[0]

        scores = raw_scores.flatten()
        if scores.shape[0] > n_anchors:
            scores = scores[:n_anchors]

        # Apply sigmoid to raw logits
        scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -100, 100)))

        coords = raw_coords.reshape(-1, 18)
        if coords.shape[0] > n_anchors:
            coords = coords[:n_anchors]

        # Decode using anchors
        decoded = self._decode_boxes(coords)

        # Filter by score
        mask = scores > self.score_threshold
        scores = scores[mask]
        decoded = decoded[mask]

        if len(scores) == 0:
            return []

        # Convert centre-wh to xyxy
        cx, cy, w, h = decoded[:, 0], decoded[:, 1], decoded[:, 2], decoded[:, 3]
        boxes_xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

        # NMS
        keep = self._nms(boxes_xyxy, scores)
        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]
        decoded = decoded[keep]

        # Scale back to original frame coordinates
        sx = self.orig_w / self.input_size
        sy = self.orig_h / self.input_size

        detections = []
        for i in range(len(scores)):
            x1 = boxes_xyxy[i, 0] * sx
            y1 = boxes_xyxy[i, 1] * sy
            x2 = boxes_xyxy[i, 2] * sx
            y2 = boxes_xyxy[i, 3] * sy

            kps = []
            for k in range(7):
                kx = decoded[i, 4 + k * 2] * sx
                ky = decoded[i, 4 + k * 2 + 1] * sy
                kps.append([kx, ky])

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "score": float(scores[i]),
                "keypoints": kps,
            })

        return detections


# =============================================================================
# Hand Landmark Detector (SNPE)
# =============================================================================

class HandLandmarkSNPE(SnpeContext):
    """
    MediaPipe Hand Landmark Detector via SNPE.

    Input:  'image' — 1 × 3 × 224 × 224 (NHWC in SNPE) RGB float32 [0,1]
    Output: 'landmarks'       — [1, 63]  (21 × 3 local coords)
            'world_landmarks' — [1, 63]  (21 × 3 world coords)
            'scores'          — [1, 1]   (confidence, after sigmoid)
            'lr'              — [1, 1]   (handedness, after sigmoid: >0.5 = right)
    """

    def __init__(
        self,
        dlc_path="None",
        input_layers=None,
        output_layers=None,
        output_tensors=None,
        runtime=Runtime.CPU,
        profile_level=PerfProfile.BALANCED,
        enable_cache=False,
        input_size=224,
    ):
        if input_layers is None:
            input_layers = ["image"]
        if output_layers is None:
            output_layers = ["landmarks", "scores", "lr", "world_landmarks"]
        if output_tensors is None:
            output_tensors = ["landmarks", "scores", "lr", "world_landmarks"]

        super().__init__(
            dlc_path,
            input_layers,
            output_layers,
            output_tensors,
            runtime,
            profile_level,
            enable_cache,
        )
        self.input_size = input_size

    def preprocess(self, frame, bbox):
        """
        Crop the palm region from the frame with padding, resize and normalise.

        Parameters
        ----------
        frame : np.ndarray  — full BGR frame
        bbox  : list[float] — [x1, y1, x2, y2] in frame pixel coords

        Returns
        -------
        True if crop is valid, False otherwise.
        Also stores self.crop_bbox for coordinate mapping.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        # Add ~30 % padding
        bw, bh = x2 - x1, y2 - y1
        pad_x = bw * 0.30
        pad_y = bh * 0.30
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        self.crop_bbox = [int(x1), int(y1), int(x2), int(y2)]
        crop = frame[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0:
            return False

        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size))
        img = img.astype(np.float32) / 255.0
        self.SetInputBuffer(img.flatten(), "image")
        return True

    def postprocess(self):
        """
        Decode landmarks, confidence and handedness.

        Returns
        -------
        dict with keys:
            'landmarks_local' — np.ndarray [21, 3]  in input-tensor space [0, input_size]
            'landmarks_frame' — np.ndarray [21, 3]  in original frame pixel space
            'world_landmarks' — np.ndarray [21, 3]  world-space
            'score'           — float  confidence
            'handedness'      — float  (>0.5 → right hand)
        """
        lm_raw = self.GetOutputBuffer("landmarks")
        wl_raw = self.GetOutputBuffer("world_landmarks")
        sc_raw = self.GetOutputBuffer("scores")
        lr_raw = self.GetOutputBuffer("lr")

        landmarks_local = lm_raw.flatten()[:63].reshape(21, 3)
        world_landmarks = wl_raw.flatten()[:63].reshape(21, 3)
        score = float(1.0 / (1.0 + np.exp(-np.clip(float(sc_raw.flatten()[0]), -100, 100))))
        handedness = float(1.0 / (1.0 + np.exp(-np.clip(float(lr_raw.flatten()[0]), -100, 100))))

        # Map local coords back to the original frame
        cx1, cy1, cx2, cy2 = self.crop_bbox
        crop_w = cx2 - cx1
        crop_h = cy2 - cy1

        landmarks_frame = landmarks_local.copy()
        landmarks_frame[:, 0] = landmarks_local[:, 0] / self.input_size * crop_w + cx1
        landmarks_frame[:, 1] = landmarks_local[:, 1] / self.input_size * crop_h + cy1
        # z stays as-is (relative depth)

        return {
            "landmarks_local": landmarks_local,
            "landmarks_frame": landmarks_frame,
            "world_landmarks": world_landmarks,
            "score": score,
            "handedness": handedness,
        }


# =============================================================================
# Gesture Classifier (SNPE)
# =============================================================================

class GestureClassifierSNPE(SnpeContext):
    """
    MediaPipe Canned Gesture Classifier via SNPE.

    Input:  'hand' — [1, 64]  (63 normalised landmark coords + 1 handedness)
    Output: 'Identity' — [1, 8]  (gesture class logits → softmax)
    """

    def __init__(
        self,
        dlc_path="None",
        input_layers=None,
        output_layers=None,
        output_tensors=None,
        runtime=Runtime.CPU,
        profile_level=PerfProfile.BALANCED,
        enable_cache=False,
    ):
        if input_layers is None:
            input_layers = ["hand"]
        if output_layers is None:
            output_layers = ["Identity"]
        if output_tensors is None:
            output_tensors = ["Identity"]

        super().__init__(
            dlc_path,
            input_layers,
            output_layers,
            output_tensors,
            runtime,
            profile_level,
            enable_cache,
        )

    def preprocess(self, landmarks_local, handedness):
        """
        Normalise hand landmarks and prepare input for gesture classifier.

        Following Qualcomm AI Hub's preprocess_hand_x64:
        1. Centre landmarks around anatomical anchors (indices 0,1,5,9,13,17)
        2. Scale so max x/y range = 1
        3. Flatten to 63 values, append handedness → 64 values

        Parameters
        ----------
        landmarks_local : np.ndarray [21, 3]  — landmark coordinates
        handedness      : float               — handedness score (>0.5 = right)
        """
        pts = landmarks_local.copy().astype(np.float32)

        # Anatomical anchor indices for centering
        center_idxs = [0, 1, 5, 9, 13, 17]
        center = pts[center_idxs].mean(axis=0)
        normed = pts - center

        # Scale by max range of x or y
        x_range = normed[:, 0].max() - normed[:, 0].min()
        y_range = normed[:, 1].max() - normed[:, 1].min()
        scale = max(x_range, y_range) + 1e-5

        normed = normed / scale
        flat = normed.flatten()  # 63 values

        # Append handedness
        input_data = np.append(flat, np.float32(handedness))  # 64 values

        self.SetInputBuffer(input_data.astype(np.float32), "hand")

    def postprocess(self):
        """
        Decode gesture classifier output.

        Returns
        -------
        dict with keys:
            'gesture'    — str   (gesture label)
            'confidence' — float (softmax probability)
            'scores'     — np.ndarray [8] (all softmax scores)
        """
        raw = self.GetOutputBuffer("Identity")
        logits = raw.flatten()[:8]

        # Softmax
        exp_logits = np.exp(logits - logits.max())
        probs = exp_logits / (exp_logits.sum() + 1e-8)

        idx = int(np.argmax(probs))
        return {
            "gesture": GESTURE_LABELS[idx],
            "confidence": float(probs[idx]),
            "scores": probs,
        }


# =============================================================================
# Geometric fallback: count fingers from landmarks
# =============================================================================

def count_fingers_geometric(landmarks):
    """
    Count raised fingers from 21 hand landmarks using simple geometry.
    Returns (gesture_label, confidence) as a fallback when gesture DLC fails.
    """
    if landmarks is None or len(landmarks) < 21:
        return "None", 0.0

    finger_count = 0

    # Thumb: compare tip.x vs IP.x depending on hand orientation
    wrist_x = landmarks[WRIST][0]
    middle_mcp_x = landmarks[MIDDLE_MCP][0]
    is_right = wrist_x < middle_mcp_x

    if is_right:
        if landmarks[THUMB_TIP][0] < landmarks[THUMB_IP][0]:
            finger_count += 1
    else:
        if landmarks[THUMB_TIP][0] > landmarks[THUMB_IP][0]:
            finger_count += 1

    # Other fingers: tip.y < PIP.y means raised (y decreases upward)
    tips = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    pips = [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
    for tip_idx, pip_idx in zip(tips, pips):
        if landmarks[tip_idx][1] < landmarks[pip_idx][1]:
            finger_count += 1

    # Map finger count to approximate gesture
    gesture_map = {
        0: "Closed_Fist",
        1: "Pointing_Up",
        2: "Victory",
        3: "Open_Palm",
        4: "Open_Palm",
        5: "Open_Palm",
    }
    gesture = gesture_map.get(finger_count, "None")
    return gesture, 0.8


# =============================================================================
# Visualisation helpers
# =============================================================================

def draw_hand_landmarks(frame, landmarks, gesture_label="", confidence=0.0):
    """Draw 21 landmarks with skeleton connections and gesture label."""
    if landmarks is None:
        return frame

    # Draw connections
    for i, j in HAND_CONNECTIONS:
        pt1 = (int(landmarks[i][0]), int(landmarks[i][1]))
        pt2 = (int(landmarks[j][0]), int(landmarks[j][1]))

        # Pick colour based on the first point's finger group
        group = LANDMARK_FINGER_MAP.get(i, "palm")
        colour = FINGER_COLORS.get(group, (200, 200, 200))
        cv2.line(frame, pt1, pt2, colour, 2, cv2.LINE_AA)

    # Draw landmark points
    for idx in range(21):
        x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
        group = LANDMARK_FINGER_MAP.get(idx, "palm")
        colour = FINGER_COLORS.get(group, (200, 200, 200))
        cv2.circle(frame, (x, y), 5, colour, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 5, (0, 0, 0), 1, cv2.LINE_AA)

    # Gesture label (top of frame)
    if gesture_label and gesture_label != "None":
        label = f"{gesture_label} {confidence * 100:.0f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(frame, (10, 10), (20 + tw, 20 + th + 10), (0, 0, 0), -1)
        cv2.putText(frame, label, (15, 20 + th), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    return frame


# =============================================================================
# Flask web application
# =============================================================================

app = Flask(__name__)
output_frame = None
lock = threading.Lock()
current_gesture = "None"
current_confidence = 0.0
current_fps = 0.0
system_ready = False


def detection_thread(camera_id, palm_det, hand_lm, gesture_cls, use_fallback=False):
    """
    Capture frames from camera and run the 3-stage hand gesture pipeline.
    """
    global output_frame, lock, current_gesture, current_confidence, current_fps, system_ready

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return

    print("Webcam opened successfully")
    system_ready = True

    fps_start = time.time()
    fps_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # --- Stage 1: Palm detection ---
            palm_det.preprocess(frame)
            detections = []
            if palm_det.Execute():
                detections = palm_det.postprocess()

            annotated = frame.copy()
            best_gesture = "None"
            best_conf = 0.0

            for det in detections:
                bbox = det["bbox"]

                # --- Stage 2: Hand landmark detection ---
                if not hand_lm.preprocess(frame, bbox):
                    continue

                if not hand_lm.Execute():
                    continue

                lm_result = hand_lm.postprocess()

                if lm_result["score"] < 0.2:
                    continue

                landmarks_frame = lm_result["landmarks_frame"]
                landmarks_local = lm_result["landmarks_local"]
                handedness = lm_result["handedness"]

                # --- Stage 3: Gesture classification ---
                gesture_label = "None"
                gesture_conf = 0.0

                if use_fallback:
                    gesture_label, gesture_conf = count_fingers_geometric(landmarks_frame)
                else:
                    gesture_cls.preprocess(landmarks_local, handedness)
                    if gesture_cls.Execute():
                        g_result = gesture_cls.postprocess()
                        gesture_label = g_result["gesture"]
                        gesture_conf = g_result["confidence"]
                    else:
                        # Fall back to geometric method
                        gesture_label, gesture_conf = count_fingers_geometric(landmarks_frame)

                if gesture_conf > best_conf:
                    best_gesture = gesture_label
                    best_conf = gesture_conf

                # Draw landmarks and gesture on annotated frame
                annotated = draw_hand_landmarks(
                    annotated, landmarks_frame, gesture_label, gesture_conf
                )

            # Update FPS
            fps_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                current_fps = fps_count / elapsed
                fps_count = 0
                fps_start = time.time()

            # Draw FPS
            fps_text = f"FPS: {current_fps:.1f}"
            cv2.putText(
                annotated, fps_text,
                (10, annotated.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
            )

            with lock:
                output_frame = annotated.copy()
                current_gesture = best_gesture
                current_confidence = best_conf

    except Exception as e:
        print(f"Error in detection thread: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()


def generate_frames():
    """MJPEG streaming generator."""
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                time.sleep(0.1)
                continue
            ok, encoded = cv2.imencode(".jpg", output_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encoded) + b"\r\n"
        )
        time.sleep(0.033)


# ---- HTML template ----
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Hand Gesture Recognition</title>
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
        background: #0a0a0a;
        color: #e0e0e0;
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .header {
        width: 100%;
        padding: 18px 24px;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 14px;
        border-bottom: 1px solid #2a2a4a;
    }
    .header h1 {
        font-size: 1.5rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .badge {
        background: linear-gradient(135deg, #00b4d8, #0077b6);
        color: #fff;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .main {
        flex: 1;
        width: 100%;
        max-width: 960px;
        padding: 20px;
        display: flex;
        flex-direction: column;
        gap: 16px;
    }
    .video-container {
        width: 100%;
        background: #111;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #2a2a4a;
        position: relative;
    }
    .video-container img {
        width: 100%;
        display: block;
    }
    .info-bar {
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
    }
    .info-card {
        flex: 1;
        min-width: 200px;
        background: #1a1a2e;
        border-radius: 10px;
        padding: 16px 20px;
        border: 1px solid #2a2a4a;
    }
    .info-card .label {
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .info-card .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #00ff88;
    }
    .info-card .value.none {
        color: #666;
    }
    .fps-value {
        color: #00b4d8 !important;
    }
</style>
</head>
<body>
    <div class="header">
        <h1>Hand Gesture Recognition</h1>
        <span class="badge">SNPE / DSP</span>
    </div>
    <div class="main">
        <div class="video-container">
            <img src="/video_feed" alt="Video Stream" />
        </div>
        <div class="info-bar">
            <div class="info-card">
                <div class="label">Current Gesture</div>
                <div class="value" id="gesture">—</div>
            </div>
            <div class="info-card">
                <div class="label">Confidence</div>
                <div class="value" id="confidence">—</div>
            </div>
            <div class="info-card">
                <div class="label">FPS</div>
                <div class="value fps-value" id="fps">—</div>
            </div>
        </div>
    </div>
    <script>
        function updateStatus() {
            fetch('/get_status')
                .then(r => r.json())
                .then(data => {
                    const gestureEl = document.getElementById('gesture');
                    gestureEl.textContent = data.gesture || '—';
                    gestureEl.className = (data.gesture && data.gesture !== 'None')
                        ? 'value' : 'value none';
                    const confEl = document.getElementById('confidence');
                    if (data.confidence > 0 && data.gesture !== 'None') {
                        confEl.textContent = (data.confidence * 100).toFixed(0) + '%';
                    } else {
                        confEl.textContent = '—';
                    }
                    document.getElementById('fps').textContent =
                        data.fps > 0 ? data.fps.toFixed(1) : '—';
                })
                .catch(() => {});
        }
        setInterval(updateStatus, 500);
        updateStatus();
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/get_status")
def get_status():
    with lock:
        return jsonify({
            "gesture": current_gesture,
            "confidence": current_confidence,
            "fps": current_fps,
            "ready": system_ready,
        })


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Hand Gesture Recognition (SNPE/DSP)")
    parser.add_argument(
        "--palm-model",
        default="../models/mediapipe_hand_gesture-palm_detector-w8a8.dlc",
        help="Path to palm detector DLC",
    )
    parser.add_argument(
        "--landmark-model",
        default="../models/mediapipe_hand_gesture-hand_landmark_detector-w8a8.dlc",
        help="Path to hand landmark detector DLC",
    )
    parser.add_argument(
        "--gesture-model",
        default="../models/mediapipe_hand_gesture-canned_gesture_classifier-w8a8.dlc",
        help="Path to gesture classifier DLC",
    )
    parser.add_argument("--runtime", default="DSP", choices=["CPU", "DSP"])
    parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    parser.add_argument("--port", type=int, default=5002, help="Web server port")
    parser.add_argument("--host", default="0.0.0.0", help="Web server host")
    parser.add_argument(
        "--fallback", action="store_true",
        help="Use geometric finger-counting instead of gesture classifier DLC",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Hand Gesture Recognition  [SNPE / DSP]")
    print("=" * 60)

    runtime = Runtime.DSP if args.runtime == "DSP" else Runtime.CPU
    print(f"\nRuntime: {args.runtime}")

    # ---- Initialise palm detector ----
    print("\nInitialising palm detector...")
    palm_det = PalmDetectorSNPE(
        dlc_path=args.palm_model,
        input_layers=["image"],
        output_layers=["box_scores", "box_coords"],
        output_tensors=["box_scores", "box_coords"],
        runtime=runtime,
        profile_level=PerfProfile.BURST,
    )
    if not palm_det.Initialize():
        print("ERROR: Failed to initialise palm detector!")
        return 1
    print("  Palm detector OK")

    # ---- Initialise hand landmark detector ----
    print("Initialising hand landmark detector...")
    hand_lm = HandLandmarkSNPE(
        dlc_path=args.landmark_model,
        input_layers=["image"],
        output_layers=["landmarks", "scores", "lr", "world_landmarks"],
        output_tensors=["landmarks", "scores", "lr", "world_landmarks"],
        runtime=runtime,
        profile_level=PerfProfile.BURST,
    )
    if not hand_lm.Initialize():
        print("ERROR: Failed to initialise hand landmark detector!")
        return 1
    print("  Hand landmark detector OK")

    # ---- Initialise gesture classifier ----
    use_fallback = args.fallback
    gesture_cls = None

    if not use_fallback:
        print("Initialising gesture classifier...")
        gesture_cls = GestureClassifierSNPE(
            dlc_path=args.gesture_model,
            input_layers=["hand"],
            output_layers=["Identity"],
            output_tensors=["Identity"],
            runtime=runtime,
            profile_level=PerfProfile.BURST,
        )
        if not gesture_cls.Initialize():
            print("WARNING: Failed to initialise gesture classifier — using geometric fallback")
            use_fallback = True
        else:
            print("  Gesture classifier OK")
    else:
        print("Using geometric finger-counting fallback (--fallback)")

    print("\nAll models initialised")

    # ---- Start detection thread ----
    print("\nStarting webcam detection...")
    det_thread = threading.Thread(
        target=detection_thread,
        args=(args.camera, palm_det, hand_lm, gesture_cls, use_fallback),
        daemon=True,
    )
    det_thread.start()

    time.sleep(2)

    print("\n" + "=" * 60)
    print("  Web Interface Starting...")
    print("=" * 60)
    print("\n  Open your browser:")
    print(f"   http://localhost:{args.port}")
    print("\n   Or from another device:")
    print(f"   http://<your-ip>:{args.port}")
    print("\nPress Ctrl+C to stop")
    print("=" * 60 + "\n")

    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False)
    except KeyboardInterrupt:
        print("\n\nShutting down...")

    return 0


if __name__ == "__main__":
    exit(main())
