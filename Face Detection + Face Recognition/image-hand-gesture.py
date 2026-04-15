"""
image-hand-gesture.py — Hand Gesture Recognition for Still Images (TFLite)
"""

import sys
import os
import cv2
import numpy as np
import argparse

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    # Fallback for environments with full TensorFlow
    from tensorflow.lite.python.interpreter import Interpreter


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
# Quantization helpers
# =============================================================================

def _dequantize(tensor, detail):
    """Dequantize a uint8 tensor using scale/zero_point from output detail."""
    if detail['dtype'] == np.uint8:
        scale, zp = detail['quantization']
        return (tensor.astype(np.float32) - zp) * scale
    return tensor.astype(np.float32)


def _quantize(data, detail):
    """Quantize float32 data to uint8 using scale/zero_point from input detail."""
    if detail['dtype'] == np.uint8:
        scale, zp = detail['quantization']
        return np.clip(np.round(data / scale + zp), 0, 255).astype(np.uint8)
    return data.astype(np.float32)


# =============================================================================
# Anchor generation for MediaPipe Palm Detector (256 × 256)
# =============================================================================

def _generate_palm_anchors(input_size=256):
    """Generate SSD-style anchors for the MediaPipe palm detector."""
    strides = [8, 16]
    anchors = []
    for stride in strides:
        grid_h = input_size // stride
        grid_w = input_size // stride
        for y in range(grid_h):
            for x in range(grid_w):
                cx = (x + 0.5) * stride / input_size
                cy = (y + 0.5) * stride / input_size
                anchors.append([cx, cy])
                anchors.append([cx, cy])
    return np.array(anchors, dtype=np.float32)


# =============================================================================
# Model Classes
# =============================================================================

class PalmDetectorTFLite:
    def __init__(self, model_path, input_size=256, score_threshold=0.5, nms_threshold=0.3):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size = input_size
        self.input_dtype = self.input_details[0]['dtype']
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.anchors = _generate_palm_anchors(input_size)

        self._score_output_idx = 0
        self._coord_output_idx = 1
        for i, detail in enumerate(self.output_details):
            name = detail["name"].lower()
            if "score" in name:
                self._score_output_idx = i
            elif "coord" in name or "box" in name or "regress" in name:
                self._coord_output_idx = i

    def preprocess(self, frame):
        self.orig_h, self.orig_w = frame.shape[:2]
        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.input_dtype == np.uint8:
            img = img.astype(np.uint8)
        else:
            img = img.astype(np.float32) / 255.0
        self._input_data = np.expand_dims(img, axis=0)

    def execute(self):
        try:
            self.interpreter.set_tensor(self.input_details[0]["index"], self._input_data)
            self.interpreter.invoke()
            return True
        except Exception as e:
            print(f"Palm detector inference error: {e}")
            return False

    def _decode_boxes(self, raw_coords):
        n = raw_coords.shape[0]
        decoded = raw_coords.copy()
        sz = float(self.input_size)
        anchor_cx = self.anchors[:n, 0] * sz
        anchor_cy = self.anchors[:n, 1] * sz
        decoded[:, 0] = raw_coords[:, 0] + anchor_cx
        decoded[:, 1] = raw_coords[:, 1] + anchor_cy
        for k in range(7):
            decoded[:, 4 + k * 2] = raw_coords[:, 4 + k * 2] + anchor_cx
            decoded[:, 4 + k * 2 + 1] = raw_coords[:, 4 + k * 2 + 1] + anchor_cy
        return decoded

    def _nms(self, boxes_xyxy, scores):
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
        raw_scores = self.interpreter.get_tensor(self.output_details[self._score_output_idx]["index"])
        raw_coords = self.interpreter.get_tensor(self.output_details[self._coord_output_idx]["index"])
        raw_scores = _dequantize(raw_scores, self.output_details[self._score_output_idx])
        raw_coords = _dequantize(raw_coords, self.output_details[self._coord_output_idx])

        n_anchors = self.anchors.shape[0]
        scores = raw_scores.flatten()
        if scores.shape[0] > n_anchors:
            scores = scores[:n_anchors]
        scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -100, 100)))

        coords = raw_coords.reshape(-1, 18)
        if coords.shape[0] > n_anchors:
            coords = coords[:n_anchors]
        decoded = self._decode_boxes(coords)

        mask = scores > self.score_threshold
        scores = scores[mask]
        decoded = decoded[mask]

        if len(scores) == 0:
            return []

        cx, cy, w, h = decoded[:, 0], decoded[:, 1], decoded[:, 2], decoded[:, 3]
        boxes_xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

        keep = self._nms(boxes_xyxy, scores)
        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]
        decoded = decoded[keep]

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


class HandLandmarkTFLite:
    def __init__(self, model_path, input_size=224):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size = input_size
        self.input_dtype = self.input_details[0]['dtype']

        self._out_map = {}
        for i, detail in enumerate(self.output_details):
            name = detail["name"].lower()
            if "world" in name:
                self._out_map["world_landmarks"] = i
            elif "landmark" in name:
                self._out_map["landmarks"] = i
            elif "score" in name or "confidence" in name:
                self._out_map["scores"] = i
            elif "lr" in name or "handed" in name:
                self._out_map["lr"] = i

        if len(self._out_map) < 4:
            self._out_map = {"landmarks": 0, "scores": 1, "lr": 2, "world_landmarks": 3}

    def _get_dequantized(self, output_key):
        idx = self._out_map[output_key]
        detail = self.output_details[idx]
        raw = self.interpreter.get_tensor(detail["index"])
        return _dequantize(raw, detail)

    def preprocess(self, frame, bbox):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        pad_x = bw * 1.0
        pad_y = bh * 1.5
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

        if self.input_dtype == np.uint8:
            img = img.astype(np.uint8)
        else:
            img = img.astype(np.float32) / 255.0
        self._input_data = np.expand_dims(img, axis=0)
        return True

    def execute(self):
        try:
            self.interpreter.set_tensor(self.input_details[0]["index"], self._input_data)
            self.interpreter.invoke()
            return True
        except Exception as e:
            print(f"Hand landmark inference error: {e}")
            return False

    def postprocess(self):
        lm_raw = self._get_dequantized("landmarks")
        wl_raw = self._get_dequantized("world_landmarks")
        sc_raw = self._get_dequantized("scores")
        lr_raw = self._get_dequantized("lr")

        landmarks_local = lm_raw.flatten()[:63].reshape(21, 3)
        world_landmarks = wl_raw.flatten()[:63].reshape(21, 3)
        score = float(1.0 / (1.0 + np.exp(-np.clip(float(sc_raw.flatten()[0]), -100, 100))))
        handedness = float(1.0 / (1.0 + np.exp(-np.clip(float(lr_raw.flatten()[0]), -100, 100))))

        cx1, cy1, cx2, cy2 = self.crop_bbox
        crop_w = cx2 - cx1
        crop_h = cy2 - cy1

        landmarks_frame = landmarks_local.copy()
        landmarks_frame[:, 0] = landmarks_local[:, 0] / self.input_size * crop_w + cx1
        landmarks_frame[:, 1] = landmarks_local[:, 1] / self.input_size * crop_h + cy1

        return {
            "landmarks_local": landmarks_local,
            "landmarks_frame": landmarks_frame,
            "world_landmarks": world_landmarks,
            "score": score,
            "handedness": handedness,
        }


class GestureClassifierTFLite:
    def __init__(self, model_path):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_dtype = self.input_details[0]['dtype']

    def preprocess(self, landmarks_local, handedness):
        pts = landmarks_local.copy().astype(np.float32)
        center_idxs = [0, 1, 5, 9, 13, 17]
        center = pts[center_idxs].mean(axis=0)
        normed = pts - center
        x_range = normed[:, 0].max() - normed[:, 0].min()
        y_range = normed[:, 1].max() - normed[:, 1].min()
        scale = max(x_range, y_range) + 1e-5
        normed = normed / scale
        flat = normed.flatten()
        input_data = np.append(flat, np.float32(handedness))
        self._input_data = np.expand_dims(_quantize(input_data, self.input_details[0]), axis=0)

    def execute(self):
        try:
            self.interpreter.set_tensor(self.input_details[0]["index"], self._input_data)
            self.interpreter.invoke()
            return True
        except Exception as e:
            print(f"Gesture classifier inference error: {e}")
            return False

    def postprocess(self):
        raw = self.interpreter.get_tensor(self.output_details[0]["index"])
        logits = _dequantize(raw, self.output_details[0]).flatten()[:8]
        exp_logits = np.exp(logits - logits.max())
        probs = exp_logits / (exp_logits.sum() + 1e-8)
        idx = int(np.argmax(probs))
        return {
            "gesture": GESTURE_LABELS[idx],
            "confidence": float(probs[idx]),
            "scores": probs,
        }


# =============================================================================
# Fallback & Visualisation Helpers
# =============================================================================

def count_fingers_geometric(landmarks):
    if landmarks is None or len(landmarks) < 21:
        return "None", 0.0

    finger_count = 0
    wrist_x = landmarks[WRIST][0]
    middle_mcp_x = landmarks[MIDDLE_MCP][0]
    is_right = wrist_x < middle_mcp_x

    if is_right:
        if landmarks[THUMB_TIP][0] < landmarks[THUMB_IP][0]:
            finger_count += 1
    else:
        if landmarks[THUMB_TIP][0] > landmarks[THUMB_IP][0]:
            finger_count += 1

    tips = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    pips = [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
    for tip_idx, pip_idx in zip(tips, pips):
        if landmarks[tip_idx][1] < landmarks[pip_idx][1]:
            finger_count += 1

    gesture_map = {0: "Closed_Fist", 1: "Pointing_Up", 2: "Victory", 3: "Open_Palm", 4: "Open_Palm", 5: "Open_Palm"}
    return gesture_map.get(finger_count, "None"), 0.8


def draw_hand_landmarks(frame, landmarks, gesture_label="", confidence=0.0):
    if landmarks is None:
        return frame

    for i, j in HAND_CONNECTIONS:
        pt1 = (int(landmarks[i][0]), int(landmarks[i][1]))
        pt2 = (int(landmarks[j][0]), int(landmarks[j][1]))
        group = LANDMARK_FINGER_MAP.get(i, "palm")
        colour = FINGER_COLORS.get(group, (200, 200, 200))
        cv2.line(frame, pt1, pt2, colour, 2, cv2.LINE_AA)

    for idx in range(21):
        x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
        group = LANDMARK_FINGER_MAP.get(idx, "palm")
        colour = FINGER_COLORS.get(group, (200, 200, 200))
        cv2.circle(frame, (x, y), 5, colour, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 5, (0, 0, 0), 1, cv2.LINE_AA)

    if gesture_label and gesture_label != "None":
        label = f"{gesture_label} {confidence * 100:.0f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(frame, (10, 10), (20 + tw, 20 + th + 10), (0, 0, 0), -1)
        cv2.putText(frame, label, (15, 20 + th), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    return frame


# =============================================================================
# Main Execution Logic
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Process a single image for hand gestures")
    parser.add_argument("--image", default="test.png", help="Path to input image")
    parser.add_argument("--palm-model", default="../models/mediapipe_hand_gesture-palm_detector-w8a8.tflite", help="Path to palm model")
    parser.add_argument("--landmark-model", default="../models/mediapipe_hand_gesture-hand_landmark_detector-w8a8.tflite", help="Path to landmark model")
    parser.add_argument("--gesture-model", default="../models/mediapipe_hand_gesture-canned_gesture_classifier-w8a8.tflite", help="Path to gesture model")
    parser.add_argument("--fallback", action="store_true", help="Use geometric finger-counting instead of TFLite classifier")
    args = parser.parse_args()

    # 1. Load the Image
    print(f"Reading image: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        print("Error: Could not load image. Please check the path.")
        return 1
    
    # image = cv2.flip(image, 1)

    # 2. Initialise the Models
    print("Initialising models...")
    try:        
        palm_det = PalmDetectorTFLite(model_path=args.palm_model, score_threshold=0.65)
        hand_lm = HandLandmarkTFLite(model_path=args.landmark_model)
        
        gesture_cls = None
        use_fallback = args.fallback
        if not use_fallback:
            try:
                gesture_cls = GestureClassifierTFLite(model_path=args.gesture_model)
            except Exception as e:
                print(f"WARNING: Failed to load gesture classifier ({e}). Using fallback.")
                use_fallback = True
    except Exception as e:
        print(f"ERROR Initialising models: {e}")
        return 1

    # 3. Process the Image Pipeline
    annotated_image = image.copy()
    
    palm_det.preprocess(image)
    if palm_det.execute():
        detections = palm_det.postprocess()
        print(f"Detected {len(detections)} hand(s).")

        for det in detections:
            bbox = det["bbox"]

            if not hand_lm.preprocess(image, bbox) or not hand_lm.execute():
                continue

            lm_result = hand_lm.postprocess()
            if lm_result["score"] < 0.1:
                continue

            # Classify Gesture
            gesture_label = "None"
            gesture_conf = 0.0

            if use_fallback:
                gesture_label, gesture_conf = count_fingers_geometric(lm_result["landmarks_frame"])
            else:
                gesture_cls.preprocess(lm_result["landmarks_local"], lm_result["handedness"])
                if gesture_cls.execute():
                    g_result = gesture_cls.postprocess()
                    gesture_label = g_result["gesture"]
                    gesture_conf = g_result["confidence"]
                else:
                    gesture_label, gesture_conf = count_fingers_geometric(lm_result["landmarks_frame"])

            # Draw on image
            annotated_image = draw_hand_landmarks(
                annotated_image, 
                lm_result["landmarks_frame"], 
                gesture_label, 
                gesture_conf
            )
            print(f"Result: {gesture_label} ({gesture_conf*100:.1f}%)")
    else:
        print("Palm detection execution failed.")

    # 4. Display and Save Result
    output_filename = "output_" + os.path.basename(args.image)
    cv2.imwrite(output_filename, annotated_image)
    print(f"\nSaved annotated image to: {output_filename}")

    # Show the image window (Press any key to close)
    print("Opening result window. Press any key in the window to exit...")
    cv2.imshow("Hand Gesture Recognition (Image)", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    sys.exit(main())