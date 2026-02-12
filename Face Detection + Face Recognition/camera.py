import cv2
import numpy as np
import argparse
import threading
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from web import SCRFD, ArcFace, FaceDatabase
    from snpehelper_manager import Runtime, PerfProfile
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Live Webcam Face Recognition (Local Window)')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('--db-path', default='face_database', help='Database directory')
    parser.add_argument('--scrfd-dlc', default='../SCRFD (Face Detection)/Model/scrfd_quantized_6490.dlc', help='Path to SCRFD DLC')
    parser.add_argument('--arcface-dlc', default='../ArcFace (Face Recognition)/Model/arcface_quantized_6490.dlc', help='Path to ArcFace DLC')
    parser.add_argument('--runtime', default='DSP', choices=['CPU', 'DSP'], help='SNPE Runtime (CPU/DSP)')
    parser.add_argument('--threshold', type=float, default=0.4, help='Recognition similarity threshold')
    
    args = parser.parse_args()

    print("="*60)
    print("Live Webcam Face Recognition")
    print("="*60)

    print(f"Loading database from: {args.db_path}")
    db = FaceDatabase(args.db_path)
    print(f"✓ Database loaded: {len(db)} people enrolled")

    runtime = Runtime.DSP if args.runtime == 'DSP' else Runtime.CPU
    print(f"Initializing models on {args.runtime}...")

    scrfd = SCRFD(
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

    arcface = ArcFace(
        dlc_path=args.arcface_dlc,
        input_layers=["data"],
        output_layers=["pre_fc1"],
        output_tensors=["fc1"],
        runtime=runtime,
        profile_level=PerfProfile.BURST
    )

    if not scrfd.Initialize():
        print("❌ Failed to initialize SCRFD. Check model path.")
        return 1
    if not arcface.Initialize():
        print("❌ Failed to initialize ArcFace. Check model path.")
        return 1
    
    print("✓ Models initialized successfully")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"❌ Could not open camera {args.camera}")
        return 1
    
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("\nStarting Loop...")
    print("Controls:")
    print("  'q' - Quit")
    print("  'e' - Enroll the currently detected face (prompts for name in terminal)")
    print("="*60)

    input_size = (320, 320)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        inference_frame = frame.copy()
        
        scrfd.preprocess(cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB))
        
        detections = []
        if scrfd.Execute():
            raw_detections = scrfd.postprocess()
            
            h, w = frame.shape[:2]
            
            for det in raw_detections:
                x1, y1, x2, y2 = [int(v) for v in det['bbox']]
                score = float(det['score'])
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    face_crop = frame[y1:y2, x1:x2]
                    
                    embedding = arcface.get_embedding(face_crop)
                    
                    name = "Unknown"
                    similarity = 0.0
                    match_info = None

                    if embedding is not None:
                        matches = db.search(embedding, threshold=args.threshold, top_k=1)
                        if matches:
                            match = matches[0]
                            name = match['name']
                            similarity = match['similarity']
                            match_info = match

                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'score': score,
                        'name': name,
                        'similarity': similarity,
                        'embedding': embedding,
                        'landmarks': det['landmarks']
                    })

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            name = det['name']
            sim = det['similarity']
            
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{name} ({sim:.2f})" if name != "Unknown" else f"Unknown ({det['score']:.2f})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            for lx, ly in det['landmarks']:
                cv2.circle(frame, (int(lx), int(ly)), 2, (255, 255, 0), -1)

        cv2.imshow('Face Recognition (Live)', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            if len(detections) == 1:
                print("\n[ENROLLMENT] Face detected.")
                new_name = input("Enter name for this person: ").strip()
                if new_name:
                    face = detections[0]
                    person_id = f"user_{int(time.time())}"
                    db.add_person(person_id, new_name, face['embedding'])
                    print(f"✓ Enrolled {new_name} successfully!\n")
                else:
                    print("Enrollment cancelled (empty name).\n")
            elif len(detections) == 0:
                print("\n[ENROLLMENT] No face detected to enroll.\n")
            else:
                print("\n[ENROLLMENT] Multiple faces detected. Please make sure only one face is visible.\n")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
