import numpy as np
import onnxruntime as ort

FACE_DETECTOR_PATH = "FaceDetector.onnx"
FACE_LANDMARK_PATH = "FaceLandmarkDetector.onnx"


def test_model(model_path: str, input_shape: list[int]) -> None:
    print(f"\n--- Testing: {model_path} ---")
    sess = ort.InferenceSession(model_path)

    inp = sess.get_inputs()[0]
    print(f"Input  : {inp.name} {input_shape} {inp.type}")

    dummy = np.zeros(input_shape, dtype=np.float32)
    results = sess.run(None, {inp.name: dummy})

    for i, (out, res) in enumerate(zip(sess.get_outputs(), results)):
        print(f"Output {i}: {out.name} {res.shape}")

    print("OK")


if __name__ == "__main__":
    test_model(FACE_DETECTOR_PATH, [1, 3, 256, 256])
    test_model(FACE_LANDMARK_PATH, [1, 3, 192, 192])
    print("\nAll models passed!")
