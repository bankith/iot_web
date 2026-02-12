import sys
import cv2
import numpy as np
import time
import argparse

from snpehelper_manager import PerfProfile,Runtime,timer,SnpeContext

class YOLO26(SnpeContext):
    def __init__(self,dlc_path: str = "None",
                    input_layers : list = [],
                    output_layers : list = [],
                    output_tensors : list = [],
                    runtime : str = Runtime.CPU,
                    profile_level : str = PerfProfile.BALANCED,
                    enable_cache : bool = False,
                    conf_thres : float = 0.5,
                    iou_thres : float = 0.5):
        super().__init__(dlc_path,input_layers,output_layers,output_tensors,runtime,profile_level,enable_cache)
        self.classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        print (len(self.classes))
        self.color_palette = np.random.uniform(0, 255, size=(80, 3))
        self.size = 640
        self.rows = 8400
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def preprocess(self,image):
        img_height, img_width = self.size, self.size

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (self.size, self.size))
        image_data = np.array(image) / 255.0
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        self.SetInputBuffer(image_data,"images")
        return


    def draw_detections(self, img, box, score, class_id):
        x1, y1, w, h = box
        color = self.color_palette[class_id]

        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        label = f'{self.classes[class_id]}: {score:.2f}'

        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                      cv2.FILLED)

        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def postprocess(self, image):
        confidence_thres = self.conf_thres
        output = self.GetOutputBuffer("output0") # Shape: [1, 300, 6]
        # output is likely already NMSed and contains [x1, y1, x2, y2, score, class_id]
        
        detections = output[0] # [300, 6]
        
        img_height, img_width = image.shape[:2]
        # Coordinates are usually normalized or relative to 640x640
        # If they are in 0-640 range:
        x_factor = img_width / self.size
        y_factor = img_height / self.size

        for i in range(len(detections)):
            det = detections[i]
            score = det[4]
            
            if score < confidence_thres:
                continue
                
            x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
            class_id = int(det[5])
            
            # Map back to original image size
            left = int(x1 * x_factor)
            top = int(y1 * y_factor)
            width = int((x2 - x1) * x_factor)
            height = int((y2 - y1) * y_factor)
            
            self.draw_detections(image, [left, top, width, height], score, class_id)

        return image

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO26 SNPE Inference")
    parser.add_argument('--model', type=str, default='whisper-export/yolo26n_quantized.dlc', help='Path to DLC model file')
    parser.add_argument('--image', type=str, default='bus.jpg', help='Path to input image')
    parser.add_argument('--output', type=str, default='output.jpg', help='Path to output image')
    parser.add_argument('--runtime', type=str, default='cpu', choices=['cpu', 'gpu', 'dsp', 'aip'], help='Runtime target')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5, help='IOU threshold')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    runtime_map = {
        'cpu': Runtime.CPU,
        'gpu': Runtime.GPU,
        'dsp': Runtime.DSP,
        'aip': Runtime.AIP
    }
    
    selected_runtime = runtime_map[args.runtime]

    print(f"Model: {args.model}")
    print(f"Image: {args.image}")
    print(f"Runtime: {args.runtime} ({selected_runtime})")
    print(f"Confidence: {args.conf}, IOU: {args.iou}")

    model_object = YOLO26(
        dlc_path=args.model,
        input_layers=["images"],
        output_layers=["output0"],
        output_tensors=["output0"],
        runtime=selected_runtime,
        profile_level=PerfProfile.BURST,
        enable_cache=False,
        conf_thres=args.conf,
        iou_thres=args.iou)

    ret = model_object.Initialize()
    if(ret != True):
        print("!"*50,"Failed to Initialize","!"*50)
        exit(0)
    print ("====== Initialize done")
    
    image = cv2.imread(args.image)
    if image is None:
        print(f"Failed to load image: {args.image}")
        exit(1)
        
    model_object.preprocess(image)

    # execute
    start = time.time()
    if(model_object.Execute() != True):
        print("!"*50,"Failed to Execute","!"*50)
        exit(0)
    end = time.time()
    print ("== execute time in ms: ", (end - start) * 1000)
    
    output_img = model_object.postprocess(image)
    path = args.output
    cv2.imwrite(path, output_img)
    print ("wrote output to " + path)