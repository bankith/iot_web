# YOLO26n Object Detection

This folder contains the implementation of YOLO26n for object detection using SNPE.

## Description
YOLO26n is a lightweight object detection model. This implementation is adapted for Qualcomm inference engines.

## Contents
- `yolo.py`: Main python script for object detection.
- `Model/`: Directory containing the DLC model file(s).

## Dependencies
- `opencv-python`
- `numpy`
- `snpehelper_manager` (located in parent directory)

## Usage

### Run Detection on an Image
```bash
python yolo.py --image path/to/image.jpg
```

### Arguments
- `--image`: Path to the input image (Default: `bus.jpg`).
- `--model`: Path to DLC model file (Default: `whisper-export/yolo26n_quantized.dlc`).
- `--output`: Path to save the output image (Default: `output.jpg`).
- `--runtime`: Runtime target (`cpu`, `gpu`, `dsp`, `aip`) (Default: `cpu`).
- `--conf`: Confidence threshold (Default: `0.5`).
- `--iou`: IOU/NMS threshold (Default: `0.5`).

### Output
The script saves an output image with bounding boxes around detected objects. It also prints:
- Inference execution time
- Confirmation of output file save
