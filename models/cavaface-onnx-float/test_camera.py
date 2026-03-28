"""
Quick camera + display test — no ML required.
Checks: camera opens, frames arrive, HDMI window shows.

Usage:
    pip install opencv-python
    python3 test_camera.py [--camera 0]

Press ESC or Q to quit.
"""
import argparse
import sys

import cv2


def test_camera(camera_id: int = 0) -> None:
    # 1. List all available camera devices (0-4)
    print("Scanning for cameras ...")
    available = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                available.append(i)
                print(f"  Camera {i}: OK")
            else:
                print(f"  Camera {i}: opened but no frame")
            cap.release()
        else:
            print(f"  Camera {i}: not found")

    if not available:
        print("\nERROR: No working camera found.")
        sys.exit(1)

    # 2. Open the requested camera
    print(f"\nOpening camera {camera_id} ...")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_id}")
        print(f"Available cameras: {available}")
        sys.exit(1)

    # 3. Print camera properties
    w   = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h   = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Resolution : {int(w)} x {int(h)}")
    print(f"FPS        : {fps}")
    print(f"\nDisplaying on HDMI — press ESC or Q to quit.\n")

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("ERROR: Frame read failed.")
            break

        frame_count += 1

        # Overlay info text
        cv2.putText(frame, f"Camera {camera_id}  {int(w)}x{int(h)}  frame#{frame_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Camera Test  [ESC/Q to quit]", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done — {frame_count} frames captured. Camera and display work!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera + HDMI display test")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    args = parser.parse_args()
    test_camera(args.camera)
