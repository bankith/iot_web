"""
Face recognition — compare input image or live camera against enrolled database.

Usage:
    # Static image
    python3 recognize.py --image test.jpg --cavaface cavaface.onnx

    # Live camera
    python3 recognize.py --camera 0 --cavaface cavaface.onnx

Output: prints / overlays who is in the image (or "Unknown" if below threshold).
"""
import argparse

import cv2
import numpy as np
import onnxruntime as ort

# ── Constants ──────────────────────────────────────────────────────────────────
DETECT_INPUT_HW   = (256, 256)
SCORE_THRESHOLD   = 0.75
CAVAFACE_INPUT_HW = (112, 112)
DEFAULT_THRESHOLD = 0.45   # cosine similarity — ปรับได้, ยิ่งสูงยิ่ง strict

# ArcFace reference landmark positions for 112×112 aligned output
ARCFACE_DST = np.array([
    [38.2946, 51.6963],  # right eye
    [73.5318, 51.5014],  # left eye
    [56.0252, 71.7366],  # nose
    [41.5493, 92.3655],  # right mouth corner
    [70.7299, 92.2041],  # left mouth corner
], dtype=np.float32)


# ── Helpers ────────────────────────────────────────────────────────────────────
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
    padded[pt:pt+nh, pl:pl+nw] = resized
    tensor = (padded.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    return tensor, scale, pt, pl


def decode_boxes(raw, anchors, img_hw):
    H, W = img_hw
    center = anchors[:, 0:1, :] * np.array([[W, H]], dtype=np.float32)
    scale  = anchors[:, 1:2, :]
    K = raw.shape[1]
    mask = np.ones((K, 1), dtype=np.float32); mask[1] = 0.0
    return raw * scale + center * mask


# ── Face detection (returns aligned face + bounding box) ──────────────────────
def _detect(img_rgb: np.ndarray, det_sess: ort.InferenceSession, anchors: np.ndarray):
    """Internal: run BlazeFace and return (aligned_face, box_xyxy) or (None, None)."""
    inp, scale, pt, pl = resize_pad(img_rgb, DETECT_INPUT_HW)
    name = det_sess.get_inputs()[0].name
    c1, c2, s1, s2 = det_sess.run(None, {name: inp})

    coords = np.concatenate([c1[0], c2[0]], axis=0).reshape(-1, 8, 2)
    scores = np.concatenate([s1[0], s2[0]], axis=0).reshape(-1)
    scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -100.0, 100.0)))

    decoded = decode_boxes(coords, anchors, DETECT_INPUT_HW)

    mask = scores >= SCORE_THRESHOLD
    if not mask.any():
        return None, None

    best_idx = int(np.where(mask)[0][scores[mask].argmax()])

    # Bounding box (for drawing)
    cx, cy = decoded[best_idx, 0]
    bw, bh = decoded[best_idx, 1]
    box = np.array([
        (cx - bw / 2 - pl) / scale,
        (cy - bh / 2 - pt) / scale,
        (cx + bw / 2 - pl) / scale,
        (cy + bh / 2 - pt) / scale,
    ])

    # 5 landmarks → ArcFace alignment
    kps = decoded[best_idx, 2:7, :].copy()
    kps[:, 0] = (kps[:, 0] - pl) / scale
    kps[:, 1] = (kps[:, 1] - pt) / scale

    M, _ = cv2.estimateAffinePartial2D(kps, ARCFACE_DST, method=cv2.LMEDS)
    if M is None:
        return None, box

    aligned = cv2.warpAffine(img_rgb, M, (CAVAFACE_INPUT_HW[1], CAVAFACE_INPUT_HW[0]))
    return aligned, box


# ── CavaFace embedding ─────────────────────────────────────────────────────────
def get_embedding(face_rgb: np.ndarray, cava_sess: ort.InferenceSession) -> np.ndarray:
    inp  = (face_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    name = cava_sess.get_inputs()[0].name
    emb  = cava_sess.run(None, {name: inp})[0].reshape(-1)
    return emb / (np.linalg.norm(emb) + 1e-8)


def _match(emb: np.ndarray, database: dict, threshold: float) -> tuple[str, float]:
    """Return (best_name, best_score). best_name='Unknown' if below threshold."""
    best_name, best_score = "Unknown", -1.0
    for name, db_emb in database.items():
        score = float(np.dot(emb, db_emb))
        if score > best_score:
            best_score, best_name = score, name
    if best_score < threshold:
        best_name = "Unknown"
    return best_name, best_score


# ── Mode 1: Static image ───────────────────────────────────────────────────────
def recognize(image_path: str, db_path: str, threshold: float,
              detector_path: str, cavaface_path: str) -> None:

    db_file  = np.load(db_path)
    database = {name: db_file[name] for name in db_file.files}
    if not database:
        print("Database is empty. Run build_database.py first.")
        return
    print(f"Database loaded: {list(database.keys())}")

    anchors   = generate_anchors()
    det_sess  = ort.InferenceSession(detector_path)
    cava_sess = ort.InferenceSession(cavaface_path)

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Cannot read image: {image_path}")
        return
    img_rgb = img_bgr[:, :, ::-1]

    face, _ = _detect(img_rgb, det_sess, anchors)
    if face is None:
        print("No face detected in the image.")
        return

    emb = get_embedding(face, cava_sess)

    for name, db_emb in database.items():
        print(f"  {name}: {float(np.dot(emb, db_emb)):.3f}")

    best_name, best_score = _match(emb, database, threshold)
    print()
    if best_name != "Unknown":
        print(f"Result : {best_name}  (similarity={best_score:.3f})")
    else:
        print(f"Result : Unknown  (best similarity={best_score:.3f} < threshold={threshold})")


# ── Mode 2: Live camera ────────────────────────────────────────────────────────
def run_camera(camera_id: int, db_path: str, threshold: float,
               detector_path: str, cavaface_path: str) -> None:

    db_file  = np.load(db_path)
    database = {name: db_file[name] for name in db_file.files}
    if not database:
        print("Database is empty. Run build_database.py first.")
        return
    print(f"Database loaded: {list(database.keys())}")

    anchors   = generate_anchors()
    det_sess  = ort.InferenceSession(detector_path)
    cava_sess = ort.InferenceSession(cavaface_path)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Cannot open camera {camera_id}")
        return
    print(f"Camera {camera_id} opened. Press ESC or Q to quit.")

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            print("Camera read failed.")
            break

        frame_rgb = frame_bgr[:, :, ::-1]
        face, box = _detect(frame_rgb, det_sess, anchors)

        if face is not None and box is not None:
            emb = get_embedding(face, cava_sess)
            best_name, best_score = _match(emb, database, threshold)

            # Draw bounding box
            x1, y1, x2, y2 = box.astype(int)
            color = (0, 255, 0) if best_name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{best_name} ({best_score:.2f})" if best_name != "Unknown" else "Unknown"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame_bgr, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame_bgr, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.imshow("Face Recognition  [ESC / Q to quit]", frame_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face recognition (image or live camera)")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image",  type=str, help="Input image path (static mode)")
    mode.add_argument("--camera", type=int, metavar="ID",
                      help="Camera device ID for live mode (e.g. 0, 1, 2)")

    parser.add_argument("--db",        default="face_db.npz",
                        help="Database file from build_database.py (default: face_db.npz)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Cosine similarity threshold (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--detector",  default="FaceDetector.onnx")
    parser.add_argument("--cavaface",  default="CavaFace.onnx")
    args = parser.parse_args()

    if args.image:
        recognize(args.image, args.db, args.threshold, args.detector, args.cavaface)
    else:
        run_camera(args.camera, args.db, args.threshold, args.detector, args.cavaface)
