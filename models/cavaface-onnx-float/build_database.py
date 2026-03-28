"""
Build face recognition database from a photos folder.

Folder structure:
    photos/
    ├── Mind/
    │   ├── 001.jpg
    │   └── 002.jpg
    └── John/
        └── 001.jpg

Usage:
    python3 build_database.py [--photos photos/] [--db face_db.npz]

Output: face_db.npz  — used by recognize.py
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# ── Constants (shared with camera_demo.py) ─────────────────────────────────────
DETECT_INPUT_HW   = (256, 256)
SCORE_THRESHOLD   = 0.75
CAVAFACE_INPUT_HW = (112, 112)
IMG_EXTENSIONS    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ArcFace reference landmark positions for 112×112 aligned output
ARCFACE_DST = np.array([
    [38.2946, 51.6963],  # right eye
    [73.5318, 51.5014],  # left eye
    [56.0252, 71.7366],  # nose
    [41.5493, 92.3655],  # right mouth corner
    [70.7299, 92.2041],  # left mouth corner
], dtype=np.float32)


# ── Anchors ────────────────────────────────────────────────────────────────────
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


# ── Pre-processing ─────────────────────────────────────────────────────────────
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


# ── Anchor decode + NMS ────────────────────────────────────────────────────────
def decode_boxes(raw: np.ndarray, anchors: np.ndarray, img_hw: tuple) -> np.ndarray:
    H, W = img_hw
    center = anchors[:, 0:1, :] * np.array([[W, H]], dtype=np.float32)
    scale  = anchors[:, 1:2, :]
    K = raw.shape[1]
    mask = np.ones((K, 1), dtype=np.float32); mask[1] = 0.0
    return raw * scale + center * mask


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, scores, iou_thr):
    order = scores.argsort()[::-1].tolist()
    keep = []
    while order:
        i = order.pop(0); keep.append(i)
        order = [j for j in order if box_iou(boxes[i], boxes[j]) < iou_thr]
    return keep


# ── Face detector ──────────────────────────────────────────────────────────────
def detect_face(img_rgb: np.ndarray, det_sess: ort.InferenceSession, anchors: np.ndarray):
    """Return ArcFace-aligned face [112,112,3] or None if no face found."""
    inp, scale, pt, pl = resize_pad(img_rgb, DETECT_INPUT_HW)
    name = det_sess.get_inputs()[0].name
    c1, c2, s1, s2 = det_sess.run(None, {name: inp})

    coords = np.concatenate([c1[0], c2[0]], axis=0).reshape(-1, 8, 2)
    scores = np.concatenate([s1[0], s2[0]], axis=0).reshape(-1)
    scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -100.0, 100.0)))

    decoded = decode_boxes(coords, anchors, DETECT_INPUT_HW)

    mask = scores >= SCORE_THRESHOLD
    if not mask.any():
        return None

    best_idx = int(np.where(mask)[0][scores[mask].argmax()])

    # Extract 5 facial landmarks (right eye, left eye, nose, mouth×2)
    # and map from padded-256 space back to original image space
    kps = decoded[best_idx, 2:7, :].copy()
    kps[:, 0] = (kps[:, 0] - pl) / scale
    kps[:, 1] = (kps[:, 1] - pt) / scale

    # Affine-align to ArcFace reference positions → 112×112
    M, _ = cv2.estimateAffinePartial2D(kps, ARCFACE_DST, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(img_rgb, M, (CAVAFACE_INPUT_HW[1], CAVAFACE_INPUT_HW[0]))


# ── CavaFace embedding ─────────────────────────────────────────────────────────
def get_embedding(face_rgb: np.ndarray, cava_sess: ort.InferenceSession) -> np.ndarray:
    """Return L2-normalized 512-dim embedding."""
    inp  = (face_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    name = cava_sess.get_inputs()[0].name
    emb  = cava_sess.run(None, {name: inp})[0].reshape(-1)
    return emb / (np.linalg.norm(emb) + 1e-8)


# ── Main ───────────────────────────────────────────────────────────────────────
def build(photos_dir: str, db_path: str,
          detector_path: str, cavaface_path: str) -> None:

    photos_root = Path(photos_dir)
    if not photos_root.exists():
        raise FileNotFoundError(f"Photos folder not found: {photos_root}")

    print("Loading models …")
    anchors   = generate_anchors()
    det_sess  = ort.InferenceSession(detector_path)
    cava_sess = ort.InferenceSession(cavaface_path)

    database: dict[str, list[np.ndarray]] = {}

    # Each sub-folder = one person
    person_dirs = sorted([d for d in photos_root.iterdir() if d.is_dir()])
    if not person_dirs:
        raise RuntimeError(f"No sub-folders found in {photos_root}. "
                           "Create one folder per person, e.g. photos/Mind/")

    for person_dir in person_dirs:
        name  = person_dir.name
        files = [f for f in sorted(person_dir.iterdir())
                 if f.suffix.lower() in IMG_EXTENSIONS]
        if not files:
            print(f"  [{name}] no images found — skipped")
            continue

        embeddings = []
        for img_path in files:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"  [{name}] cannot read {img_path.name} — skipped")
                continue

            img_rgb = img_bgr[:, :, ::-1]
            face    = detect_face(img_rgb, det_sess, anchors)
            if face is None:
                print(f"  [{name}] no face in {img_path.name} — skipped")
                continue

            emb = get_embedding(face, cava_sess)
            embeddings.append(emb)
            print(f"  [{name}] {img_path.name} ✓")

        if embeddings:
            # Average all embeddings for this person, then re-normalize
            avg = np.mean(embeddings, axis=0)
            avg /= np.linalg.norm(avg) + 1e-8
            database[name] = avg
            print(f"  → {name}: {len(embeddings)} photo(s) enrolled\n")

    if not database:
        print("No faces enrolled. Check photos folder.")
        return

    # Save: arrays keyed by person name
    np.savez(db_path, **database)
    print(f"Database saved → {db_path}")
    print(f"Enrolled {len(database)} person(s): {list(database.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build face recognition database")
    parser.add_argument("--photos",   default="photos",
                        help="Folder with sub-folders per person (default: photos/)")
    parser.add_argument("--db",       default="face_db.npz",
                        help="Output database file (default: face_db.npz)")
    parser.add_argument("--detector", default="FaceDetector.onnx")
    parser.add_argument("--cavaface", default="CavaFace.onnx")
    args = parser.parse_args()

    build(args.photos, args.db, args.detector, args.cavaface)
