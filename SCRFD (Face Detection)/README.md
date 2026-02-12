# SCRFD Face Detection

This folder contains the implementation of SCRFD for high-performance face detection using SNPE.

## Description
SCRFD is an efficient face detection model that provides bounding boxes and 5-point facial landmarks for detected faces. It is designed to work well across various scales.

## Contents
- `scrfd.py`: Main python script for running inference on images.
- `Model/`: Directory containing the DLC model file(s).

## Dependencies
- `opencv-python`
- `numpy`
- `pillow`
- `snpehelper_manager` (located in parent directory)

## Usage

### Run Detection on an Image
```bash
python scrfd.py --image path/to/image.jpg
```

### Arguments
- `--image`: Path to the input image (Required).
- `--model`: Path to DLC model file (Default: `det_2.5g_quantized.dlc`).
- `--output`: Path to save the output image with visualized detections (Default: `scrfd_result.jpg`).
- `--runtime`: Runtime target (`cpu`, `gpu`, `dsp`, `aip`) (Default: `dsp`).
- `--conf`: Confidence threshold (Default: `0.5`).
- `--iou`: NMS/IOU threshold (Default: `0.4`).
- `--crop`: Flag to save cropped images of detected faces.

### Output
The script saves an output image with bounding boxes drawn around detected faces. It also prints:
- Inference time
- Detection details (score, bounding box coordinates)
