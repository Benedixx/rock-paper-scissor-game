import os
import sys
import platform
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode

@smart_inference_mode()
def webcam_inference(
    weights= ROOT/'weights/best.pt',  # model.pt path(s)
    source=0,  # file/dir/URL/glob, 0 for webcam
    imgsz=(512,512),  # inference size (pixels)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.5,  # NMS IOU threshold
    data= ROOT/'Rock-Paper-Scissor-4/data.yaml',  # path to data.yaml
):
    source = str(source)
    webcam= source.isnumeric() or source.endswith('.streams')
    
    device= select_device("")
    model= DetectMultiBackend(weights, device=device, dnn=False, data=ROOT/'data', fp16=False)
    stride, names, pt= model.stride, model.names, model.pt
    imgz= check_img_size(imgsz, s=stride)
    
    batch_size= 1
    view_img= check_imshow(warn=True)
    dataset= LoadStreams(source, img_size=imgz, stride=stride, auto=pt, vid_stride=1)
    batch_size= len(dataset)
    
    model.warmup(imgsz=(1 if pt or model.triton else batch_size, 3, *imgz))
    seen, windows, dt= 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im= torch.from_numpy(im).to(model.device)
            im= im.half() if model.fp16 else im.float()
            im/= 255.0
            if len(im.shape) == 3:
                im= im[None]
            if model.xml and im.shape[0] > 1:
                ims= torch.chunk(im, chunks=im.shape[0], dim=0)
    
        with dt[1]:
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=False, visualize=False).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=False, visualize=False).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred= model(im, augment=False, visualize=False)
            
        with dt[2]:
            pred= non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=1000)
            
        for i, det in enumerate(pred):
            seen+= 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)
        
            p = Path(p)  # to Path
            s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=3, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"
                    x_min, y_min, x_max, y_max = xyxy
                    
                    if  view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = f"{names[c]} {conf:.2f}"
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        

        
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                length = im0.shape[0]
                width = im0.shape[1]
                cv2.line(im0, (int(width/2), 0), (int(width/2), length), (0, 255, 0), 3)
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
    
    LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    return x_min, y_min, x_max, y_max, confidence, cls

if __name__ == "__main__":
    # Set your desired parameters
    weights_path = ROOT / 'runs/train/exp/weights/best.pt'
    data_path = ROOT / 'Rock-Paper-Scissor-4/data.yaml'
    source = 0  # Change this to the appropriate source, e.g., path to video file or URL

    # Run the webcam_inference function
    webcam_inference(weights=weights_path, data=data_path, source=source)