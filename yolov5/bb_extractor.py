import os
import cv2
import sys
import torch
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DetectMultiBackend(ROOT / "runs/train/exp2/weights/best.pt", device=device, data="coco128.yaml")
model.warmup(imgsz=(1, 3, 640, 640))
stride, names, pt = model.stride, model.names, model.pt

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess the frame
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    img = img.copy()  # Make a copy of the array
    img = torch.from_numpy(img).to(device)
    img = img.half() if model.fp16 else img.float()  
    img /= 255
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    # Perform inference
    pred = model(img)

    # Apply Non-Maximum Suppression
    pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.5, classes=None, agnostic=False, multi_label=False, max_det=2)

    # Process the results
    for det in pred:  # per image
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()

            # Print the results
            for *xyxy, conf, cls in reversed(det):
                label = names[int(cls)]
                confidence = conf.item()
                bboxes = xyxy.tolist()
                print(f'Class: {label}, Confidence: {confidence}, Bounding Boxes: {bboxes}')

    cv2.imshow('Webcam Inference', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()