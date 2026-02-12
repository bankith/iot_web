import cv2
import numpy as np
from PIL import Image
from snpehelper_manager import SnpeContext, Runtime, PerfProfile
import argparse

class ArcFace(SnpeContext):

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
        if isinstance(face_image, Image.Image):
            face_image = np.array(face_image)

        input_image = cv2.resize(face_image, (self.input_size[1], self.input_size[0]))

        if len(input_image.shape) == 3 and input_image.shape[2] == 3:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        
        input_image = input_image.astype(np.float32)
        input_image = (input_image - 127.5) / 128.0

        input_image_flat = input_image.flatten()
        self.SetInputBuffer(input_image_flat, "data")

    def postprocess(self):
        embedding = self.GetOutputBuffer("fc1")

        embedding = embedding.reshape(self.embedding_dim)

        normalized_embedding = self.normalize_embedding(embedding)

        return {
            'embedding': normalized_embedding,
            'raw_embedding': embedding.copy()
        }

    def normalize_embedding(self, embedding):
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    def get_embedding(self, face_image):
        self.preprocess(face_image)

        if not self.Execute():
            print("Error: Failed to execute ArcFace model")
            return None

        result = self.postprocess()
        return result['embedding']

    @staticmethod
    def cosine_similarity(embedding1, embedding2):
        return np.dot(embedding1, embedding2)

    @staticmethod
    def euclidean_distance(embedding1, embedding2):
        return np.linalg.norm(embedding1 - embedding2)

    @staticmethod
    def compare_faces(embedding1, embedding2, threshold=0.4):
        similarity = ArcFace.cosine_similarity(embedding1, embedding2)
        distance = ArcFace.euclidean_distance(embedding1, embedding2)

        return {
            'match': similarity > threshold,
            'similarity': float(similarity),
            'distance': float(distance),
            'confidence': 'high' if similarity > 0.6 else ('medium' if similarity > 0.4 else 'low')
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ArcFace Face Recognition Example')
    parser.add_argument('--image1', type=str, required=True,
                       help='Path to first face image')
    parser.add_argument('--image2', type=str, default=None,
                       help='Path to second face image (for comparison)')
    parser.add_argument('--dlc', type=str, default='arcfaceresnet100-8_quantized_6490.dlc',
                       help='Path to ArcFace DLC model')
    parser.add_argument('--runtime', type=str, default='DSP', choices=['CPU', 'DSP'],
                       help='SNPE runtime')

    args = parser.parse_args()

    print("="*60)
    print("ArcFace Face Recognition")
    print("="*60)
    print("\nInitializing model...")

    runtime = Runtime.DSP if args.runtime == 'DSP' else Runtime.CPU

    arcface = ArcFace(
        dlc_path=args.dlc,
        input_layers=["data"],
        output_layers=["pre_fc1"],
        output_tensors=["fc1"],
        runtime=runtime,
        profile_level=PerfProfile.BURST,
        enable_cache=False
    )

    if not arcface.Initialize():
        print("Error: Failed to initialize ArcFace model!")
        exit(1)

    print("✓ Model initialized successfully!")

    print(f"\nProcessing: {args.image1}")
    image1 = cv2.imread(args.image1)
    if image1 is None:
        print(f"Error: Could not load image: {args.image1}")
        exit(1)

    embedding1 = arcface.get_embedding(image1)
    if embedding1 is None:
        print("Error: Failed to get embedding")
        exit(1)

    print(f"✓ Generated embedding (shape: {embedding1.shape})")
    print(f"  L2 norm: {np.linalg.norm(embedding1):.6f}")
    print(f"  First 5 values: {embedding1[:5]}")

    if args.image2:
        print(f"\nProcessing: {args.image2}")
        image2 = cv2.imread(args.image2)
        if image2 is None:
            print(f"Error: Could not load image: {args.image2}")
            exit(1)

        embedding2 = arcface.get_embedding(image2)
        if embedding2 is None:
            print("Error: Failed to get embedding")
            exit(1)

        print(f"✓ Generated embedding (shape: {embedding2.shape})")

        result = ArcFace.compare_faces(embedding1, embedding2)

        print(f"\n{'='*60}")
        print("Face Comparison Results:")
        print(f"{'='*60}")
        print(f"Match: {result['match']}")
        print(f"Cosine Similarity: {result['similarity']:.4f}")
        print(f"Euclidean Distance: {result['distance']:.4f}")
        print(f"Confidence: {result['confidence']}")
        print(f"{'='*60}")

        if result['match']:
            print("\n✓ Faces MATCH (likely same person)")
        else:
            print("\n✗ Faces DO NOT MATCH (likely different people)")

    print("\nDone!")