import cv2
import numpy as np
import time
from PIL import Image
from snpehelper_manager import PerfProfile, Runtime, SnpeContext
import argparse
import sys

class SCRFD(SnpeContext):

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
        super().__init__(dlc_path, input_layers, output_layers, output_tensors,
                        runtime, profile_level, enable_cache)

        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        self.feat_stride_fpn = [8, 16, 32]
        self.num_anchors = 2 

        self._anchor_centers = {}
        self._num_anchors = {}
        for stride in self.feat_stride_fpn:
            feat_h = input_size[0] // stride
            feat_w = input_size[1] // stride
            self._num_anchors[stride] = feat_h * feat_w * self.num_anchors

            anchor_centers = np.stack(np.mgrid[:feat_h, :feat_w][::-1], axis=-1)
            anchor_centers = anchor_centers.astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))

            anchor_centers = np.stack([anchor_centers] * self.num_anchors, axis=1)
            anchor_centers = anchor_centers.reshape((-1, 2))
            self._anchor_centers[stride] = anchor_centers

    def preprocess(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)

        self.orig_shape = image.shape[:2]

        input_image = cv2.resize(image, (self.input_size[1], self.input_size[0]))

        if len(input_image.shape) == 3 and input_image.shape[2] == 3:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        input_image = input_image.astype(np.float32)
        input_image = (input_image - 127.5) / 128.0

        input_image_flat = input_image.flatten()
        self.SetInputBuffer(input_image_flat, "input.1")

        return

    def distance2bbox(self, points, distance, max_shape=None):
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
        output_mapping = {
            8: {'score': '446', 'bbox': '449', 'kps': '452'},
            16: {'score': '466', 'bbox': '469', 'kps': '472'},
            32: {'score': '486', 'bbox': '489', 'kps': '492'}
        }

        all_bboxes = []
        all_scores = []
        all_kps = []

        for stride in self.feat_stride_fpn:
            mapping = output_mapping[stride]
            num_pred = self._num_anchors[stride]

            score_output = self.GetOutputBuffer(mapping['score'])
            bbox_output = self.GetOutputBuffer(mapping['bbox'])
            kps_output = self.GetOutputBuffer(mapping['kps'])

            scores = score_output.reshape((num_pred, 1))
            bboxes = bbox_output.reshape((num_pred, 4))
            kps = kps_output.reshape((num_pred, 10))

            anchor_centers = self._anchor_centers[stride]

            bboxes = bboxes * stride
            pos_bboxes = self.distance2bbox(anchor_centers, bboxes)

            kps = kps * stride
            pos_kps = self.distance2kps(anchor_centers, kps)

            all_bboxes.append(pos_bboxes)
            all_scores.append(scores)
            all_kps.append(pos_kps)

        all_bboxes = np.vstack(all_bboxes)
        all_scores = np.vstack(all_scores).squeeze()
        all_kps = np.vstack(all_kps)

        valid_mask = all_scores > self.conf_threshold
        bboxes = all_bboxes[valid_mask]
        scores = all_scores[valid_mask]
        kps = all_kps[valid_mask]

        if len(bboxes) > 0:
            keep = self.nms(bboxes, scores)
            bboxes = bboxes[keep]
            scores = scores[keep]
            kps = kps[keep]
        
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
        if isinstance(image, Image.Image):
            image = np.array(image)

        vis_image = image.copy()

        for det in detections:
            bbox = det['bbox']
            score = det['score']
            landmarks = det['landmarks']

            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if draw_scores:
                text = f"{score:.2f}"
                cv2.putText(vis_image, text, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if draw_landmarks:
                for i, (lx, ly) in enumerate(landmarks):
                    cv2.circle(vis_image, (int(lx), int(ly)), 2, (0, 0, 255), -1)

        cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        print(f"Detection result saved to {output_path}")
        print(f"Total faces detected: {len(detections)}")

        return vis_image

def main():
    parser = argparse.ArgumentParser(description="SCRFD SNPE Inference")
    parser.add_argument('--model', type=str, default='det_2.5g_quantized.dlc', help='Path to DLC model file')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='scrfd_result.jpg', help='Path to output image')
    parser.add_argument('--runtime', type=str, default='dsp', choices=['cpu', 'gpu', 'dsp', 'aip'], help='Runtime target')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.4, help='IOU threshold')
    parser.add_argument('--crop', action='store_true', help='Save cropped faces')

    args = parser.parse_args()

    runtime_map = {
        'cpu': Runtime.CPU,
        'gpu': Runtime.GPU,
        'dsp': Runtime.DSP,
        'aip': Runtime.AIP
    }
    
    selected_runtime = runtime_map[args.runtime]
    output_layers = [
        "Sigmoid_141", "Reshape_144", "Reshape_147",
        "Sigmoid_159", "Reshape_162", "Reshape_165",
        "Sigmoid_177", "Reshape_180", "Reshape_183"
    ]

    output_tensors = [
        "446", "449", "452",
        "466", "469", "472",
        "486", "489", "492"
    ]

    print(f"Model: {args.model}")
    print(f"Image: {args.image}")
    print(f"Runtime: {args.runtime}")
    print(f"Confidence: {args.conf}, NMS: {args.iou}")

    scrfd_model = SCRFD(
        dlc_path=args.model,
        input_layers=["input.1"],
        output_layers=output_layers,
        output_tensors=output_tensors,
        runtime=selected_runtime,
        profile_level=PerfProfile.BURST,
        enable_cache=False,
        input_size=(320, 320),
        conf_threshold=args.conf,
        nms_threshold=args.iou
    )

    if not scrfd_model.Initialize():
        print("!" * 50, "Failed to Initialize", "!" * 50)
        sys.exit(1)
    print("SCRFD model initialized successfully!")

    image = cv2.imread(args.image)
    if image is None:
        print(f"Failed to load image: {args.image}")
        sys.exit(1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    scrfd_model.preprocess(image_rgb)

    start_time = time.time()
    if not scrfd_model.Execute():
        print("!" * 50, "Failed to Execute", "!" * 50)
        sys.exit(1)
    end_time = time.time()
    print(f"Inference time: {(end_time - start_time) * 1000:.2f} ms")

    detections = scrfd_model.postprocess()

    scrfd_model.draw_detections(image_rgb, detections, output_path=args.output)

    print("\nDetection Details:")
    for i, det in enumerate(detections):
        print(f"Face {i+1}: Score: {det['score']:.3f}, Bbox: {[int(v) for v in det['bbox']]}")
        
        if args.crop:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                crop = image[y1:y2, x1:x2]
                crop_path = f"face_crop_{i}.jpg"
                cv2.imwrite(crop_path, crop)
                print(f"  Saved crop to {crop_path}")

if __name__ == "__main__":
    main()