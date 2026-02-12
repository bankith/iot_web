# ArcFace Face Recognition

This folder contains the implementation of ArcFace for face recognition using SNPE on Qualcomm hardware.

## Description
ArcFace is a state-of-the-art face recognition model that maps face images to a 512-dimensional embedding space. The model is optimized for cosine similarity comparisons.

## Contents
- `arcface.py`: Main python script for inference and face comparison.
- `model/`: Directory containing the DLC model file(s).

## Dependencies
- `opencv-python`
- `numpy`
- `pillow`
- `snpehelper_manager` (located in parent directory)

## Usage

### Run Inference and Compare Two Faces
```bash
python arcface.py --image1 path/to/image1.jpg --image2 path/to/image2.jpg
```

### Arguments
- `--image1`: Path to the first face image (Required).
- `--image2`: Path to the second face image for comparison (Optional).
- `--dlc`: Path to the ArcFace DLC model file (Default: `arcfaceresnet100-8_quantized_6490.dlc`).
- `--runtime`: SNPE runtime to use (`CPU` or `DSP`, Default: `DSP`).

### Output
The script generates 512-d embeddings for the input images and, if a second image is provided, calculates:
- Cosine Similarity
- Euclidean Distance
- Match/No Match verdict based on a threshold.
