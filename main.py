import os
import torch
import cv2
from PIL import Image
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# Model
model = torch.hub.load('ultralytics/yolov5', 
                       'custom', 
                       path=ROOT / "yolov5/runs/train/exp2/weights/best.pt"
                       )  # local model

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert OpenCV BGR image to RGB
    rgb_frame = frame

    # Draw a vertical line in the middle of the frame
    length = frame.shape[0]
    width = frame.shape[1]
    cv2.line(frame, (int(width/2), 0), (int(width/2), length), (0, 255, 0), 3)

    # Convert to PIL image
    pil_image = Image.fromarray(rgb_frame)

    # Inference
    results = model(pil_image, size=640)  # single image
    for box in results.xyxy[0]:
        x_min, y_min, x_max, y_max, confidence, class_pred = box.tolist()
        print(f"Bounding Box Coordinates: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
        print(f"Confidence: {confidence}, Class: {class_pred}")

    # Display the results
    cv2.imshow('YOLOv5 Webcam Inference', results.render()[0])

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
