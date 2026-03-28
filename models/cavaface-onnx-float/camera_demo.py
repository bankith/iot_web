"""
MediaPipe Face Detection + Landmark Camera Demo
Runs on Qualcomm QCS6490 with ONNX Runtime — no PyTorch required.

Usage:
    pip install onnxruntime opencv-python numpy
    python3 camera_demo.py [--camera 0] [--detector FaceDetector.onnx] [--landmark FaceLandmarkDetector.onnx]

Press ESC or Q to quit.
"""
import argparse
import math

import cv2
import numpy as np
import onnxruntime as ort

# ── Config ─────────────────────────────────────────────────────────────────────
DETECT_INPUT_HW     = (256, 256)
LANDMARK_INPUT_HW   = (192, 192)
SCORE_THRESHOLD     = 0.75
NMS_IOU_THRESHOLD   = 0.3
MIN_LANDMARK_SCORE  = 0.5
DETECT_BOX_SCALE    = 1.1        # enlarge detected box before landmark step

# 468-point face mesh connections (lips, eyes, eyebrows, face oval)
FACE_CONNECTIONS = [
    (61,146),(146,91),(91,181),(181,84),(84,17),(17,314),(314,405),(405,321),(321,375),(375,291),
    (61,185),(185,40),(40,39),(39,37),(37,0),(0,267),(267,269),(269,270),(270,409),(409,291),
    (78,95),(95,88),(88,178),(178,87),(87,14),(14,317),(317,402),(402,318),(318,324),(324,308),
    (78,191),(191,80),(80,81),(81,82),(82,13),(13,312),(312,311),(311,310),(310,415),(415,308),
    (263,249),(249,390),(390,373),(373,374),(374,380),(380,381),(381,382),(382,362),
    (263,466),(466,388),(388,387),(387,386),(386,385),(385,384),(384,398),(398,362),
    (276,283),(283,282),(282,295),(295,285),(300,293),(293,334),(334,296),(296,336),
    (33,7),(7,163),(163,144),(144,145),(145,153),(153,154),(154,155),(155,133),
    (33,246),(246,161),(161,160),(160,159),(159,158),(158,157),(157,173),(173,133),
    (46,53),(53,52),(52,65),(65,55),(70,63),(63,105),(105,66),(66,107),
    (10,338),(338,297),(297,332),(332,284),(284,251),(251,389),(389,356),(356,454),
    (454,323),(323,361),(361,288),(288,397),(397,365),(365,379),(379,378),(378,400),
    (400,377),(377,152),(152,148),(148,176),(176,149),(149,150),(150,136),(136,172),
    (172,58),(58,132),(132,93),(93,234),(234,127),(127,162),(162,21),(21,54),
    (54,103),(103,67),(67,109),(109,10),
]


# ── Anchors ────────────────────────────────────────────────────────────────────
def generate_anchors(input_size: int = 256) -> np.ndarray:
    """
    Generate BlazeFace back-model anchors.
    stride 16 → 16x16 grid × 2 anchors = 512  (matches box_coords_1)
    stride 32 →  8x8 grid × 6 anchors = 384   (matches box_coords_2)
    shape: [896, 2, 2]  each row = [[cx_norm, cy_norm], [scale_x, scale_y]]
    """
    strides, anchors_per_cell = [16, 32], [2, 6]
    rows = []
    for stride, n in zip(strides, anchors_per_cell):
        grid = input_size // stride
        for y in range(grid):
            for x in range(grid):
                cx = (x + 0.5) / grid
                cy = (y + 0.5) / grid
                for _ in range(n):
                    rows.append([cx, cy, 1.0, 1.0])
    return np.array(rows, dtype=np.float32).reshape(-1, 2, 2)


# ── Pre-processing ─────────────────────────────────────────────────────────────
def resize_pad(frame_rgb: np.ndarray, target_hw: tuple):
    """Letterbox-resize to target, normalize [0,1], return NCHW tensor."""
    h, w = frame_rgb.shape[:2]
    th, tw = target_hw
    scale = min(th / h, tw / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(frame_rgb, (nw, nh))
    pt = (th - nh) // 2
    pl = (tw - nw) // 2
    padded = np.zeros((th, tw, 3), dtype=np.uint8)
    padded[pt:pt+nh, pl:pl+nw] = resized
    tensor = padded.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)[None]   # [1,3,H,W]
    return tensor, scale, pt, pl


# ── Anchor Decoding ────────────────────────────────────────────────────────────
def decode_boxes(raw: np.ndarray, anchors: np.ndarray, img_hw: tuple) -> np.ndarray:
    """
    raw     : [N, 8, 2]  — raw model output (box_center, box_size, 6 keypoints)
    anchors : [N, 2, 2]  — [[cx_norm, cy_norm], [sx, sy]]
    returns : [N, 8, 2]  — decoded coordinates in padded image pixel space
    """
    H, W = img_hw
    center = anchors[:, 0:1, :] * np.array([[W, H]], dtype=np.float32)  # [N,1,2]
    scale  = anchors[:, 1:2, :]                                           # [N,1,2]

    # add anchor center to everything EXCEPT index 1 (width/height)
    K = raw.shape[1]
    mask = np.ones((K, 1), dtype=np.float32)
    mask[1] = 0.0

    return raw * scale + center * mask


# ── NMS ────────────────────────────────────────────────────────────────────────
def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    order = scores.argsort()[::-1].tolist()
    keep  = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if box_iou(boxes[i], boxes[j]) < iou_thr]
    return keep


# ── Main Demo ──────────────────────────────────────────────────────────────────
class FaceDemo:
    def __init__(self, detector_path: str, landmark_path: str, camera_id: int = 0):
        print("Loading FaceDetector …")
        self.det = ort.InferenceSession(detector_path)
        print("Loading FaceLandmarkDetector …")
        self.ldm = ort.InferenceSession(landmark_path)
        self.anchors = generate_anchors(DETECT_INPUT_HW[0])   # [896, 2, 2]

        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        print(f"Camera {camera_id} opened. Press ESC or Q to quit.")

    # ── Stage 1: Face detection ────────────────────────────────────────────────
    def detect(self, frame_rgb: np.ndarray):
        """Return (box_xyxy [4], keypoints [6,2]) in original frame space, or (None,None)."""
        inp, scale, pt, pl = resize_pad(frame_rgb, DETECT_INPUT_HW)

        det_name = self.det.get_inputs()[0].name
        c1, c2, s1, s2 = self.det.run(None, {det_name: inp})
        # c1: [1,512,16]  c2: [1,384,16]  s1: [1,512,1]  s2: [1,384,1]

        coords = np.concatenate([c1[0], c2[0]], axis=0).reshape(-1, 8, 2)  # [896,8,2]
        scores = np.concatenate([s1[0], s2[0]], axis=0).reshape(-1)         # [896]

        # sigmoid + clip
        scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -100.0, 100.0)))

        # decode using anchors (pixel space of 256×256 padded image)
        decoded = decode_boxes(coords, self.anchors, DETECT_INPUT_HW)

        # build xyxy boxes from decoded center + size
        cx, cy = decoded[:, 0, 0], decoded[:, 0, 1]
        bw, bh = decoded[:, 1, 0], decoded[:, 1, 1]
        boxes = np.stack([cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2], axis=1)

        # filter by score threshold
        mask = scores >= SCORE_THRESHOLD
        if not mask.any():
            return None, None

        boxes_f, scores_f, decoded_f = boxes[mask], scores[mask], decoded[mask]

        keep = nms(boxes_f, scores_f, NMS_IOU_THRESHOLD)
        if not keep:
            return None, None

        b   = keep[0]
        box = boxes_f[b]          # [4] xyxy in 256-space
        kps = decoded_f[b, 2:, :] # [6, 2] keypoints in 256-space

        # map back to original frame pixel space
        def to_orig(pts):
            p = pts.copy()
            p[:, 0] = (p[:, 0] - pl) / scale
            p[:, 1] = (p[:, 1] - pt) / scale
            return p

        box_orig = to_orig(box.reshape(2, 2)).reshape(4)
        kps_orig = to_orig(kps)
        return box_orig, kps_orig

    # ── Stage 2: Landmark detection ───────────────────────────────────────────
    def landmarks(self, frame_rgb: np.ndarray, box: np.ndarray, kps: np.ndarray):
        """Crop face ROI, run landmark model. Returns [468, 2] in frame space or None."""
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        side   = max(x2 - x1, y2 - y1) * DETECT_BOX_SCALE

        # rotation: align by right-eye(kps[0]) → left-eye(kps[1]) axis
        re, le = kps[0], kps[1]
        angle  = math.degrees(math.atan2(le[1] - re[1], le[0] - re[0]))

        # affine crop → 192×192
        M    = cv2.getRotationMatrix2D((cx, cy), angle, LANDMARK_INPUT_HW[0] / side)
        crop = cv2.warpAffine(frame_rgb, M, (LANDMARK_INPUT_HW[1], LANDMARK_INPUT_HW[0]))

        inp = crop.astype(np.float32) / 255.0
        inp = inp.transpose(2, 0, 1)[None]  # [1,3,192,192]

        ldm_name = self.ldm.get_inputs()[0].name
        score_out, lm_out = self.ldm.run(None, {ldm_name: inp})
        # score_out: [1]   lm_out: [1, 468, 3]

        if float(score_out[0]) < MIN_LANDMARK_SCORE:
            return None

        lm = lm_out[0].copy()               # [468, 3]
        lm[:, 0] *= LANDMARK_INPUT_HW[1]    # x → 192 pixel space
        lm[:, 1] *= LANDMARK_INPUT_HW[0]    # y → 192 pixel space

        # invert affine → original frame space
        Mi  = cv2.invertAffineTransform(M)
        pts = lm[:, :2].reshape(-1, 1, 2).astype(np.float32)
        return cv2.transform(pts, Mi).reshape(-1, 2)

    # ── Drawing ───────────────────────────────────────────────────────────────
    @staticmethod
    def draw(frame_bgr: np.ndarray, box, kps, lm) -> np.ndarray:
        out = frame_bgr.copy()
        if box is not None:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 80, 255), 2)
        if kps is not None:
            for px, py in kps.astype(int):
                cv2.circle(out, (px, py), 4, (0, 255, 255), -1)
        if lm is not None:
            for px, py in lm.astype(int):
                cv2.circle(out, (px, py), 1, (0, 230, 0), -1)
            for a, b in FACE_CONNECTIONS:
                pa = tuple(lm[a].astype(int))
                pb = tuple(lm[b].astype(int))
                cv2.line(out, pa, pb, (255, 100, 0), 1)
        return out

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self) -> None:
        while True:
            ok, frame_bgr = self.cap.read()
            if not ok:
                print("Camera read failed.")
                break

            frame_rgb = frame_bgr[:, :, ::-1]  # BGR → RGB

            box, kps = self.detect(frame_rgb)
            lm = self.landmarks(frame_rgb, box, kps) if box is not None else None

            out = self.draw(frame_bgr, box, kps, lm)
            cv2.imshow("MediaPipe Face Demo  [ESC/Q to quit]", out)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

        self.cap.release()
        cv2.destroyAllWindows()


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediaPipe Face Demo (ONNX)")
    parser.add_argument("--camera",   type=int, default=0,
                        help="Camera device ID (default: 0)")
    parser.add_argument("--detector", default="FaceDetector.onnx",
                        help="Path to FaceDetector ONNX model")
    parser.add_argument("--landmark", default="FaceLandmarkDetector.onnx",
                        help="Path to FaceLandmarkDetector ONNX model")
    args = parser.parse_args()

    demo = FaceDemo(args.detector, args.landmark, args.camera)
    demo.run()
