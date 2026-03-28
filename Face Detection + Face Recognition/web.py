import cv2
import numpy as np
import time
import threading
import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from PIL import Image
from flask import Flask, Response, render_template_string, request, jsonify
from snpehelper_manager import PerfProfile, Runtime, SnpeContext

# =============================================================================
# SCRFD Face Detection Logic
# =============================================================================

class SCRFD(SnpeContext):
    """
    SCRFD (Sample and Computation Redistributed Face Detection) implementation
    for SNPE-Helper framework.

    This class handles face detection using the SCRFD_2.5G model quantized for
    Snapdragon 6490 DSP execution.

    Model Input: 1x320x320x3 (NHWC format)
    Model Outputs: Multi-scale predictions with bboxes, scores, and landmarks
    """

    def __init__(self, dlc_path: str = "None",
                 input_layers: list = [],
                 output_layers: list = [],
                 output_tensors: list = [],
                 runtime: str = Runtime.CPU,
                 profile_level: str = PerfProfile.BALANCED,
                 enable_cache: bool = False,
                 input_size: tuple = (320, 320),
                 conf_threshold: float = 0.5,
                 nms_threshold: float = 0.4):
        """
        Initialize SCRFD face detector.

        Args:
            input_size: Model input size (height, width), default (320, 320)
            conf_threshold: Confidence threshold for face detection
            nms_threshold: NMS IoU threshold for removing duplicate detections
        """
        super().__init__(dlc_path, input_layers, output_layers, output_tensors,
                        runtime, profile_level, enable_cache)

        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        # SCRFD uses 3 feature pyramid levels with different strides
        self.feat_stride_fpn = [8, 16, 32]
        self.num_anchors = 2  # 2 anchors per location

        # Generate anchor centers for each feature pyramid level
        self._anchor_centers = {}
        self._num_anchors = {}
        for stride in self.feat_stride_fpn:
            feat_h = input_size[0] // stride
            feat_w = input_size[1] // stride
            self._num_anchors[stride] = feat_h * feat_w * self.num_anchors

            # Create anchor centers
            anchor_centers = np.stack(np.mgrid[:feat_h, :feat_w][::-1], axis=-1)
            anchor_centers = anchor_centers.astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))

            # Duplicate for num_anchors
            anchor_centers = np.stack([anchor_centers] * self.num_anchors, axis=1)
            anchor_centers = anchor_centers.reshape((-1, 2))
            self._anchor_centers[stride] = anchor_centers

    def preprocess(self, image):
        """
        Preprocess input image for SCRFD model.

        The model expects:
        - Input shape: 1x320x320x3 (NHWC)
        - Normalized to range [-1, 1]
        - RGB format

        Args:
            image: PIL Image or numpy array (HWC, BGR/RGB)
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Store original shape for postprocessing
        self.orig_shape = image.shape[:2]  # (height, width)

        # Resize to model input size
        input_image = cv2.resize(image, (self.input_size[1], self.input_size[0]))

        # Convert BGR to RGB if needed (OpenCV loads as BGR)
        if len(input_image.shape) == 3 and input_image.shape[2] == 3:
            # Assume BGR from OpenCV, convert to RGB
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        # Normalize to [-1, 1] range
        input_image = input_image.astype(np.float32)
        input_image = (input_image - 127.5) / 128.0

        # Flatten and set input buffer
        input_image_flat = input_image.flatten()
        self.SetInputBuffer(input_image_flat, "input.1")

        return

    def distance2bbox(self, points, distance, max_shape=None):
        """
        Decode distance prediction to bounding box.

        Args:
            points: Anchor center points, shape (n, 2)
            distance: Distance predictions, shape (n, 4)
            max_shape: Maximum valid coordinate values

        Returns:
            Decoded bboxes, shape (n, 4) in format [x1, y1, x2, y2]
        """
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]

        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])

        return np.stack([x1, y1, x2, y2], axis=-1)

    def distance2kps(self, points, distance, max_shape=None):
        """
        Decode distance prediction to keypoints (facial landmarks).

        Args:
            points: Anchor center points, shape (n, 2)
            distance: Distance predictions for landmarks, shape (n, 10)
            max_shape: Maximum valid coordinate values

        Returns:
            Decoded keypoints, shape (n, 5, 2)
        """
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, 0] + distance[:, i]
            py = points[:, 1] + distance[:, i + 1]

            if max_shape is not None:
                px = np.clip(px, 0, max_shape[1])
                py = np.clip(py, 0, max_shape[0])

            preds.append(np.stack([px, py], axis=-1))

        return np.stack(preds, axis=1)

    def nms(self, dets, scores):
        """
        Non-Maximum Suppression.

        Args:
            dets: Bounding boxes, shape (n, 4) in format [x1, y1, x2, y2]
            scores: Confidence scores, shape (n,)

        Returns:
            Indices of boxes to keep
        """
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= self.nms_threshold)[0]
            order = order[inds + 1]

        return keep

    def postprocess(self):
        """
        Postprocess model outputs to get face detections.

        Returns:
            detections: List of detections, each containing:
                - bbox: [x1, y1, x2, y2] in original image coordinates
                - score: confidence score
                - landmarks: 5 facial keypoints, shape (5, 2)
        """
        # Output tensor mapping based on model inspection
        # Format: {stride: {'score': tensor_id, 'bbox': tensor_id, 'kps': tensor_id}}
        output_mapping = {
            8: {'score': '446', 'bbox': '449', 'kps': '452'},   # 3200 anchors
            16: {'score': '466', 'bbox': '469', 'kps': '472'},  # 800 anchors
            32: {'score': '486', 'bbox': '489', 'kps': '492'}   # 200 anchors
        }

        all_bboxes = []
        all_scores = []
        all_kps = []

        # Process each feature pyramid level
        for stride in self.feat_stride_fpn:
            mapping = output_mapping[stride]
            num_pred = self._num_anchors[stride]

            # Get outputs from model
            score_output = self.GetOutputBuffer(mapping['score'])
            bbox_output = self.GetOutputBuffer(mapping['bbox'])
            kps_output = self.GetOutputBuffer(mapping['kps'])

            # Reshape outputs
            scores = score_output.reshape((num_pred, 1))
            bboxes = bbox_output.reshape((num_pred, 4))
            kps = kps_output.reshape((num_pred, 10))

            # Get anchor centers for this level
            anchor_centers = self._anchor_centers[stride]

            # Decode bbox predictions (distance to bbox)
            # For SCRFD, bbox predictions are distance * stride
            bboxes = bboxes * stride
            pos_bboxes = self.distance2bbox(anchor_centers, bboxes)

            # Decode keypoint predictions
            kps = kps * stride
            pos_kps = self.distance2kps(anchor_centers, kps)

            all_bboxes.append(pos_bboxes)
            all_scores.append(scores)
            all_kps.append(pos_kps)

        # Concatenate all levels
        all_bboxes = np.vstack(all_bboxes)
        all_scores = np.vstack(all_scores).squeeze()
        all_kps = np.vstack(all_kps)

        # Filter by confidence threshold
        valid_mask = all_scores > self.conf_threshold
        bboxes = all_bboxes[valid_mask]
        scores = all_scores[valid_mask]
        kps = all_kps[valid_mask]

        # Apply NMS
        if len(bboxes) > 0:
            keep = self.nms(bboxes, scores)
            bboxes = bboxes[keep]
            scores = scores[keep]
            kps = kps[keep]

        # Scale back to original image size
        scale_x = self.orig_shape[1] / self.input_size[1]
        scale_y = self.orig_shape[0] / self.input_size[0]

        detections = []
        for bbox, score, kp in zip(bboxes, scores, kps):
            detection = {
                'bbox': [
                    bbox[0] * scale_x,
                    bbox[1] * scale_y,
                    bbox[2] * scale_x,
                    bbox[3] * scale_y
                ],
                'score': float(score),
                'landmarks': kp * np.array([scale_x, scale_y])
            }
            detections.append(detection)

        return detections

    def draw_detections(self, image, detections, output_path="scrfd_result.jpg",
                       draw_landmarks=True, draw_scores=True):
        """
        Visualize face detections on image.

        Args:
            image: Input image (PIL Image or numpy array)
            detections: List of detections from postprocess()
            output_path: Path to save annotated image
            draw_landmarks: Whether to draw facial landmarks
            draw_scores: Whether to draw confidence scores
        """
        # Convert to numpy if PIL
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Make a copy to draw on
        vis_image = image.copy()

        # Draw each detection
        for det in detections:
            bbox = det['bbox']
            score = det['score']
            landmarks = det['landmarks']

            # Draw bounding box
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw score
            if draw_scores:
                text = f"{score:.2f}"
                cv2.putText(vis_image, text, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw landmarks (5 facial keypoints)
            if draw_landmarks:
                for i, (lx, ly) in enumerate(landmarks):
                    cv2.circle(vis_image, (int(lx), int(ly)), 2, (0, 0, 255), -1)

        # Save result
        cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        print(f"Detection result saved to {output_path}")
        print(f"Total faces detected: {len(detections)}")

        return vis_image

# =============================================================================
# ArcFace Face Recognition Logic
# =============================================================================

class ArcFace(SnpeContext):
    """
    ArcFace ResNet100 face recognition model.

    This model generates 512-dimensional embedding vectors from face images.
    It should be used AFTER face detection to process cropped face regions.

    Args:
        dlc_path: Path to the ArcFace DLC model file
        input_layers: List of input layer names (default: ["data"])
        output_layers: List of output layer names (default: ["pre_fc1"])
        output_tensors: List of output tensor IDs (default: ["fc1"])
        runtime: SNPE runtime (CPU or DSP)
        profile_level: Performance profile
        enable_cache: Whether to enable caching
        input_size: Input image size (default: (112, 112))
    """

    def __init__(self, dlc_path: str = "None",
                 input_layers: list = [],
                 output_layers: list = [],
                 output_tensors: list = [],
                 runtime: str = Runtime.CPU,
                 profile_level: str = PerfProfile.BALANCED,
                 enable_cache: bool = False,
                 input_size: tuple = (112, 112)):

        super().__init__(dlc_path, input_layers, output_layers, output_tensors,
                        runtime, profile_level, enable_cache)

        self.input_size = input_size
        self.embedding_dim = 512

    def preprocess(self, face_image):
        """
        Preprocess face image for ArcFace model.

        Args:
            face_image: Input face image (PIL Image or numpy array)
                       Should be a cropped face region, not full image

        Returns:
            None (sets input buffer internally)
        """
        # Convert PIL to numpy if needed
        if isinstance(face_image, Image.Image):
            face_image = np.array(face_image)

        # Resize to model input size (112x112)
        input_image = cv2.resize(face_image, (self.input_size[1], self.input_size[0]))

        # Convert BGR to RGB if needed (OpenCV loads as BGR)
        if len(input_image.shape) == 3 and input_image.shape[2] == 3:
            # Check if likely BGR (loaded with cv2)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        # Normalize to [-1, 1] range
        # ArcFace expects: (pixel - 127.5) / 128.0
        input_image = input_image.astype(np.float32)
        input_image = (input_image - 127.5) / 128.0

        # Flatten and set input buffer
        input_image_flat = input_image.flatten()
        self.SetInputBuffer(input_image_flat, "data")

    def postprocess(self):
        """
        Extract andnormalize the face embedding vector.

        Returns:
            dict: Dictionary containing:
                - 'embedding': Normalized 512-dimensional numpy array
                - 'raw_embedding': Un-normalized embedding (optional)
        """
        # Get the embedding from fc1 output
        embedding = self.GetOutputBuffer("fc1")

        # Reshape to 1D vector
        embedding = embedding.reshape(self.embedding_dim)

        # L2 normalization (critical for cosine similarity)
        normalized_embedding = self.normalize_embedding(embedding)

        return {
            'embedding': normalized_embedding,
            'raw_embedding': embedding.copy()
        }

    def normalize_embedding(self, embedding):
        """
        L2 normalize an embedding vector.

        Args:
            embedding: Numpy array of embedding

        Returns:
            Normalized embedding (unit vector)
        """
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    def get_embedding(self, face_image):
        """
        Complete pipeline: preprocess → execute → extract embedding.

        Args:
            face_image: Input face image (PIL Image or numpy array)

        Returns:
            Normalized 512-dimensional embedding vector
        """
        self.preprocess(face_image)

        if not self.Execute():
            print("Error: Failed to execute ArcFace model")
            return None

        result = self.postprocess()
        return result['embedding']

    @staticmethod
    def cosine_similarity(embedding1, embedding2):
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector (should be normalized)
            embedding2: Second embedding vector (should be normalized)

        Returns:
            Similarity score in range [-1, 1]
            Higher values indicate more similar faces
        """
        return np.dot(embedding1, embedding2)

    @staticmethod
    def euclidean_distance(embedding1, embedding2):
        """
        Compute Euclidean distance between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Distance value (lower is more similar)
        """
        return np.linalg.norm(embedding1 - embedding2)

    @staticmethod
    def compare_faces(embedding1, embedding2, threshold=0.4):
        """
        Compare two face embeddings to determine if they're the same person.

        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            threshold: Similarity threshold (default: 0.4)
                      >0.6: High confidence same person
                      0.4-0.6: Likely same person
                      <0.4: Different people

        Returns:
            dict: Dictionary containing:
                - 'match': Boolean indicating if faces match
                - 'similarity': Cosine similarity score
                - 'distance': Euclidean distance
        """
        similarity = ArcFace.cosine_similarity(embedding1, embedding2)
        distance = ArcFace.euclidean_distance(embedding1, embedding2)

        return {
            'match': similarity > threshold,
            'similarity': float(similarity),
            'distance': float(distance),
            'confidence': 'high' if similarity > 0.6 else ('medium' if similarity > 0.4 else 'low')
        }


# =============================================================================
# Face Database Logic
# =============================================================================

class FaceDatabase:
    """
    Simple face database for storing and retrieving face embeddings.
    Uses JSON for metadata and pickle for embeddings.
    """

    def __init__(self, db_path="face_database"):
        """
        Initialize face database.

        Args:
            db_path: Directory to store database files
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)

        self.metadata_file = self.db_path / "metadata.json"
        self.embeddings_file = self.db_path / "embeddings.pkl"

        self.metadata = {}
        self.embeddings = {}

        self.load()

    def load(self):
        """Load database from disk."""
        # Load metadata
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)

        # Load embeddings
        if self.embeddings_file.exists():
            with open(self.embeddings_file, 'rb') as f:
                self.embeddings = pickle.load(f)

    def save(self):
        """Save database to disk."""
        # Save metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        # Save embeddings
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(self.embeddings, f)

    def add_person(self, person_id, name, embedding, image_path=None):
        """
        Add a person to the database.

        Args:
            person_id: Unique identifier for the person
            name: Person's name
            embedding: 512-dimensional face embedding
            image_path: Optional path to reference image

        Returns:
            bool: Success status
        """
        # Store metadata
        self.metadata[person_id] = {
            'name': name,
            'enrolled_at': datetime.now().isoformat(),
            'image_path': str(image_path) if image_path else None,
            'embedding_shape': embedding.shape
        }

        # Store embedding
        self.embeddings[person_id] = embedding

        self.save()
        return True

    def remove_person(self, person_id):
        """Remove a person from the database."""
        if person_id in self.metadata:
            del self.metadata[person_id]
            del self.embeddings[person_id]
            self.save()
            return True
        return False

    def search(self, query_embedding, threshold=0.4, top_k=5):
        """
        Search for matching faces in the database.

        Args:
            query_embedding: Face embedding to search for
            threshold: Minimum similarity threshold
            top_k: Number of top matches to return

        Returns:
            list: List of matches sorted by similarity
        """
        if not self.embeddings:
            return []

        matches = []

        for person_id, db_embedding in self.embeddings.items():
            similarity = float(np.dot(query_embedding, db_embedding))

            if similarity >= threshold:
                matches.append({
                    'person_id': person_id,
                    'name': self.metadata[person_id]['name'],
                    'similarity': similarity,
                    'enrolled_at': self.metadata[person_id]['enrolled_at']
                })

        # Sort by similarity (descending)
        matches.sort(key=lambda x: x['similarity'], reverse=True)

        return matches[:top_k]

    def get_person(self, person_id):
        """Get person information by ID."""
        if person_id in self.metadata:
            return {
                **self.metadata[person_id],
                'person_id': person_id,
                'embedding': self.embeddings[person_id]
            }
        return None

    def list_all(self):
        """List all people in the database."""
        return [
            {
                'person_id': pid,
                **info
            }
            for pid, info in self.metadata.items()
        ]

    def __len__(self):
        """Get number of people in database."""
        return len(self.metadata)

# =============================================================================
# Web Application Logic
# =============================================================================

app = Flask(__name__)
output_frame = None
lock = threading.Lock()
face_results = []
system_ready = False

scrfd_model = None
arcface_model = None
database = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Face Recognition</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<style>
:root {
    --bg-1: #0f172a;
    --bg-2: #1e293b;
    --glass: rgba(255,255,255,0.08);
    --border: rgba(255,255,255,0.15);
    --primary: #6366f1;
    --success: #22c55e;
    --danger: #ef4444;
    --text-main: #f8fafc;
    --text-sub: #cbd5f5;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: Inter, system-ui, sans-serif;
    background:
        radial-gradient(1200px 600px at 10% 10%, #1e1b4b, transparent),
        linear-gradient(135deg, var(--bg-1), var(--bg-2));
    color: var(--text-main);
    min-height: 100vh;
}

.container {
    max-width: 1440px;
    margin: auto;
    padding: 32px;
}

h1 {
    text-align: center;
    font-size: 2.4rem;
    font-weight: 700;
}

.subtitle {
    text-align: center;
    color: var(--text-sub);
    margin: 8px 0 32px;
}

.main-content {
    display: grid;
    grid-template-columns: 3fr 1.2fr;
    gap: 24px;
}

@media (max-width: 1024px) {
    .main-content {
        grid-template-columns: 1fr;
    }
}

/* Video */
.video-panel {
    background: var(--glass);
    padding: 16px;
    border: 1px solid var(--border);
    backdrop-filter: blur(16px);
    box-shadow: 0 20px 60px rgba(0,0,0,.35);
}

#videoStream {
    width: 100%;
    background: #000;
}

.faces-panel {
    background: var(--glass);
    padding: 20px;
    border: 1px solid var(--border);
    backdrop-filter: blur(16px);
    max-height: 640px;
    overflow-y: auto;
}

.faces-panel h2 {
    margin-bottom: 16px;
}

.face-card {
    background: linear-gradient(180deg,
        rgba(255,255,255,.12),
        rgba(255,255,255,.05)
    );
    padding: 16px;
    margin-bottom: 14px;
    border: 1px solid var(--border);
    transition: .25s;
}

.face-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 30px rgba(0,0,0,.35);
}

.face-card.identified { border-left: 4px solid var(--success); }
.face-card.unknown    { border-left: 4px solid var(--danger); }

.face-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.face-name {
    font-weight: 600;
}

.face-similarity {
    font-size: .85rem;
    color: var(--text-sub);
}

/* Stats */
.stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 20px;
    margin-top: 28px;
}

.stat-box {
    background: var(--glass);
    padding: 22px;
    border: 1px solid var(--border);
    backdrop-filter: blur(16px);
    text-align: center;
}

.stat-value {
    font-size: 2rem;
    font-weight: 700;
    margin-top: 8px;
}

.stat-label {
    font-size: .75rem;
    letter-spacing: .15em;
    text-transform: uppercase;
    color: var(--text-sub);
}

</style>
</head>

<body>
<div class="container">
    <h1>Face Recognition</h1>

    <div class="main-content">
        <div class="video-panel">
            <img id="videoStream" src="{{ url_for('video_feed') }}">
        </div>

        <div class="faces-panel">
            <h2>Detected Faces</h2>
            <div id="facesList"></div>
        </div>
    </div>

    <div class="stats">
        <div class="stat-box">
            <div class="stat-label">People in Database</div>
            <div class="stat-value" id="dbCount">--</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Faces Detected</div>
            <div class="stat-value" id="facesCount">--</div>
        </div>
    </div>
</div>

<script>
function updateFaces() {
    fetch('/get_faces')
        .then(r => r.json())
        .then(data => {
            facesCount.textContent = data.faces.length;
            dbCount.textContent = data.db_size;

            facesList.innerHTML = data.faces.length === 0
                ? `<div style="opacity:.6;text-align:center">No faces detected</div>`
                : data.faces.map((face,i) => `
                    <div class="face-card ${face.identified?'identified':'unknown'}">
                        <div class="face-header">
                            <div class="face-name">
                                ${face.identified ? '✓ '+face.matches[0].name : '❓ Unknown'}
                            </div>
                            ${face.identified ?
                                `<div class="face-similarity">${(face.matches[0].similarity*100).toFixed(1)}%</div>` :
                                `<div class="face-similarity" style="color:var(--danger)">Not in DB</div>`
                            }
                        </div>
                        <div style="font-size:.85rem;opacity:.7">
                            Score: ${face.detection_score.toFixed(3)}
                        </div>
                    </div>
                `).join('');
        });
}

setInterval(updateFaces,1000);
updateFaces();
</script>
</body>
</html>
"""


def detection_thread(camera_id, scrfd, arcface, db, skip_frames=1, threshold=0.4):
    global output_frame, lock, face_results, system_ready

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return

    print(f"Webcam opened successfully")
    system_ready = True

    frame_count = 0
    last_faces = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            frame_count += 1

            if frame_count % (skip_frames + 1) == 1:
                scrfd.preprocess(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if scrfd.Execute():
                    detections = scrfd.postprocess()

                    faces = []
                    for det in detections:
                        x1, y1, x2, y2 = [int(v) for v in det['bbox']]

                        h, w = frame.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        if x2 > x1 and y2 > y1:
                            face_crop = frame[y1:y2, x1:x2]

                            embedding = arcface.get_embedding(face_crop)
                            if embedding is not None:
                                matches = db.search(embedding, threshold=0.9, top_k=3)

                                filtered_matches = [m for m in matches if m['similarity'] >= 0.5]

                                faces.append({
                                    'bbox': np.array(det['bbox']).astype(float).tolist(),
                                    'detection_score': float(det['score']),
                                    'landmarks': np.array(det['landmarks']).astype(float).tolist(),
                                    'embedding': embedding,
                                    'matches': filtered_matches,
                                    'identified': len(filtered_matches) > 0,
                                    'face_crop': face_crop
                                })

                    if len(faces) > 1:
                        person_best_match = {}

                        for i, face in enumerate(faces):
                            if face['identified'] and len(face['matches']) > 0:
                                person_id = face['matches'][0]['person_id']
                                similarity = face['matches'][0]['similarity']
                                detection_score = face['detection_score']

                                combined_score = (similarity * 0.7) + (detection_score * 0.3)

                                if person_id not in person_best_match:
                                    person_best_match[person_id] = (i, combined_score)
                                else:
                                    if combined_score > person_best_match[person_id][1]:
                                        person_best_match[person_id] = (i, combined_score)

                        best_indices = {idx for idx, _ in person_best_match.values()}
                        for i, face in enumerate(faces):
                            if face['identified'] and i not in best_indices:
                                face['matches'] = []
                                face['identified'] = False

                    last_faces = faces

            annotated = frame.copy()
            for face in last_faces:
                bbox = face['bbox']
                x1, y1, x2, y2 = [int(v) for v in bbox]

                if face['identified']:
                    color = (0, 255, 0)
                    name = face['matches'][0]['name']
                    similarity = face['matches'][0]['similarity']
                    label = f"{name} ({similarity:.2f})"
                else:
                    color = (0, 0, 255)
                    label = "Unknown"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
                cv2.putText(annotated, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                for lx, ly in face['landmarks']:
                    cv2.circle(annotated, (int(lx), int(ly)), 2, (255, 0, 0), -1)

            with lock:
                output_frame = annotated.copy()
                face_results = last_faces.copy()

    except Exception as e:
        print(f"Error in detection thread: {e}")
    finally:
        cap.release()


def generate_frames():
    """Generate frames for MJPEG streaming."""
    global output_frame, lock

    while True:
        with lock:
            if output_frame is None:
                time.sleep(0.1)
                continue

            (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encoded_image) + b'\r\n')

        time.sleep(0.033)


@app.route('/')
def index():
    """Main page."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_faces')
def get_faces():
    """Get current detected faces."""
    global face_results, lock, database

    with lock:
        faces = [
            {
                'bbox': f['bbox'],
                'detection_score': f['detection_score'],
                'matches': f['matches'],
                'identified': f['identified']
            }
            for f in face_results
        ]

    return jsonify({
        'faces': faces,
        'db_size': len(database)
    })


IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def build_database_from_folder(datasets_dir: str, scrfd: SCRFD, arcface: ArcFace,
                                database: FaceDatabase) -> None:
    """
    Pre-build face database from a datasets folder.

    Folder structure:
        datasets/
        ├── rooney/
        │   ├── rooney1.jpg
        │   └── rooney2.jpg
        └── titi/
            ├── 001.jpg
            └── 002.jpg

    Each sub-folder name becomes the person's name in the database.
    All images per person are averaged into a single representative embedding.

    Args:
        datasets_dir: Path to folder containing per-person sub-folders
        scrfd:        Initialized SCRFD face detector
        arcface:      Initialized ArcFace recognizer
        database:     FaceDatabase instance to populate
    """
    datasets_path = Path(datasets_dir)
    if not datasets_path.exists():
        print(f"  ⚠ Datasets folder not found: {datasets_dir} — skipping pre-enrollment")
        return

    person_dirs = sorted([d for d in datasets_path.iterdir() if d.is_dir()])
    if not person_dirs:
        print(f"  ⚠ No sub-folders found in {datasets_dir} — skipping pre-enrollment")
        return

    print(f"\nBuilding database from: {datasets_dir}")
    print("-" * 40)

    enrolled_count = 0

    for person_dir in person_dirs:
        name = person_dir.name
        person_id = name.lower().replace(' ', '_')

        # Skip if person already enrolled
        if database.get_person(person_id) is not None:
            print(f"  [{name}] already in database — skipped")
            continue

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

            # Detect face with SCRFD
            scrfd.preprocess(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            if not scrfd.Execute():
                print(f"  [{name}] SCRFD failed on {img_path.name} — skipped")
                continue

            detections = scrfd.postprocess()
            if not detections:
                print(f"  [{name}] no face in {img_path.name} — skipped")
                continue

            # Pick highest-score detection
            best = max(detections, key=lambda d: d['score'])
            x1, y1, x2, y2 = [int(v) for v in best['bbox']]
            h, w = img_bgr.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            face_crop = img_bgr[y1:y2, x1:x2]
            embedding = arcface.get_embedding(face_crop)
            if embedding is not None:
                embeddings.append(embedding)
                print(f"  [{name}] {img_path.name} ✓")

        if embeddings:
            # Average all embeddings then re-normalize → one representative vector
            avg_embedding = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(avg_embedding)
            avg_embedding = avg_embedding / (norm + 1e-8)
            database.add_person(person_id, name, avg_embedding)
            print(f"  → Enrolled: {name} ({len(embeddings)} photo(s))\n")
            enrolled_count += 1
        else:
            print(f"  [{name}] no valid faces found — not enrolled\n")

    print(f"Pre-enrollment done: {enrolled_count} new person(s) added")
    print(f"Total in database: {len(database)} person(s)")
    print("-" * 40)


def main():
    global scrfd_model, arcface_model, database

    parser = argparse.ArgumentParser(description='Web-Based Face Recognition System')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--db-path', default='face_database', help='Database directory')
    parser.add_argument('--datasets', default='', help='Path to datasets folder (sub-folders = person names)')
    parser.add_argument('--scrfd-dlc', default='../SCRFD (Face Detection)/Model/scrfd_quantized_6490.dlc')
    parser.add_argument('--arcface-dlc', default='../ArcFace (Face Recognition)/Model/arcface_quantized_6490.dlc')
    parser.add_argument('--runtime', default='DSP', choices=['CPU', 'DSP'])
    parser.add_argument('--threshold', type=float, default=0.4, help='Similarity threshold')
    parser.add_argument('--skip-frames', type=int, default=1, help='Process every N frames')
    parser.add_argument('--host', default='0.0.0.0', help='Web server host')
    parser.add_argument('--port', type=int, default=5000, help='Web server port')

    args = parser.parse_args()

    print("="*60)
    print("Web-Based Face Recognition System")
    print("="*60)

    # Initialize database
    print("\nInitializing database...")
    database = FaceDatabase(args.db_path)
    print(f"✓ Database loaded ({len(database)} people)")

    # Initialize models
    print("\nInitializing models...")
    runtime = Runtime.DSP if args.runtime == 'DSP' else Runtime.CPU

    scrfd_model = SCRFD(
        dlc_path=args.scrfd_dlc,
        input_layers=["input.1"],
        output_layers=[
            "Sigmoid_141", "Reshape_144", "Reshape_147",
            "Sigmoid_159", "Reshape_162", "Reshape_165",
            "Sigmoid_177", "Reshape_180", "Reshape_183"
        ],
        output_tensors=[
            "446", "449", "452",
            "466", "469", "472",
            "486", "489", "492"
        ],
        runtime=runtime,
        profile_level=PerfProfile.BURST
    )

    arcface_model = ArcFace(
        dlc_path=args.arcface_dlc,
        input_layers=["data"],
        output_layers=["pre_fc1"],
        output_tensors=["fc1"],
        runtime=runtime,
        profile_level=PerfProfile.BURST
    )

    if not scrfd_model.Initialize():
        print("Error: Failed to initialize SCRFD!")
        return 1

    if not arcface_model.Initialize():
        print("Error: Failed to initialize ArcFace!")
        return 1

    print("✓ Models initialized")

    # Pre-enroll faces from datasets folder (if provided)
    if args.datasets:
        build_database_from_folder(args.datasets, scrfd_model, arcface_model, database)
    else:
        print("\n(No --datasets folder specified — skipping pre-enrollment)")

    print("\nStarting webcam detection...")
    detection_process = threading.Thread(
        target=detection_thread,
        args=(args.camera, scrfd_model, arcface_model, database, args.skip_frames, args.threshold),
        daemon=True
    )
    detection_process.start()

    time.sleep(2)

    print(f"\n{'='*60}")
    print(f"🌐 Web Interface Starting...")
    print(f"{'='*60}")
    print(f"\n📺 Open your browser:")
    print(f"   http://localhost:{args.port}")
    print(f"\n   Or from another device:")
    print(f"   http://<your-ip>:{args.port}")
    print(f"\nPress Ctrl+C to stop")
    print(f"{'='*60}\n")

    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    return 0

if __name__ == "__main__":
    exit(main())