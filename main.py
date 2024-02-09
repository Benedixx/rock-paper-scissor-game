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

left_box_found = False
right_box_found = False
game_start = False
game_finish = False
left_hand= None
right_hand= None

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert OpenCV BGR image to RGB
    rgb_frame = frame

    # Draw a vertical line in the middle of the frame
    length = frame.shape[0]
    width = frame.shape[1]
    cv2.line(frame, (int(width/2), 0), (int(width/2), length), (0, 255, 0), 3)
    if not game_start and not game_finish :
        cv2.putText(frame, "Press 's' to start the game", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 5, cv2.LINE_AA)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            game_start = True
    

    # Convert to PIL image
    pil_image = Image.fromarray(rgb_frame)

    # Inference
    results = model(pil_image, size=512)
    if game_start == True:
        for box in results.xyxy[0]:
            x_min, y_min, x_max, y_max, confidence, class_pred = box.tolist()
            if box[0] > width/2 and not right_box_found:
                print(f"Bounding Box right Coordinates: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
                right_hand = class_pred
                print("Right class: ", right_hand)
                right_box_found = True
            elif box[0] <= width/2 and not left_box_found:
                print(f"Bounding Box left Coordinates: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
                left_hand = class_pred
                print("Left class: ", left_hand)
                left_box_found = True
        
        
    if left_box_found == True and right_box_found == True:
        if right_hand == left_hand:
            # cv2.putText(frame, "DRAW", (int(width/2), int(length/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 5, cv2.LINE_AA)
            # cv2.putText(frame, "Press 's' to play again", (int(width/3), int(length/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 5, cv2.LINE_AA)
            print("DRAW")
            print("Press 's' to play again")
            game_start = False
            game_finish = True
            left_hand= None
            right_hand= None
            left_box_found = False
            right_box_found = False
        elif right_hand == 0 and left_hand == 2:
            # cv2.putText(frame, "Right hand wins", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 5, cv2.LINE_AA)
            # cv2.putText(frame, "Press 's' to play again", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 5, cv2.LINE_AA)
            print("Right hand wins")
            print("Press 's' to play again")
            game_start = False
            game_finish = True
            left_hand= None
            right_hand= None
            left_box_found = False
            right_box_found = False
        elif right_hand == 2 and left_hand == 0:
            # cv2.putText(frame, "Left hand wins", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 5, cv2.LINE_AA)
            # cv2.putText(frame, "Press 's' to play again", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 5, cv2.LINE_AA)
            print("Left hand wins")
            print("Press 's' to play again")
            game_start = False
            game_finish = True
            left_hand= None
            right_hand= None
            left_box_found = False
            right_box_found = False
        elif right_hand == 0 and left_hand == 1:
            # cv2.putText(frame, "Left hand wins", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 5, cv2.LINE_AA)
            # cv2.putText(frame, "Press 's' to play again", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 5, cv2.LINE_AA)
            print("Left hand wins")
            print("Press 's' to play again")
            game_start = False
            game_finish = True    
            left_hand= None
            right_hand= None
            left_box_found = False
            right_box_found = False
        elif right_hand == 1 and left_hand == 0:
            # cv2.putText(frame, "Right hand wins", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 5, cv2.LINE_AA)
            # cv2.putText(frame, "Press 's' to play again", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 5, cv2.LINE_AA)
            print("Right hand wins")
            print("Press 's' to play again")
            game_start = False
            game_finish = True 
            left_hand= None
            right_hand= None
            left_box_found = False
            right_box_found = False
        elif right_hand == 1 and left_hand == 2:
            # cv2.putText(frame, "Left hand wins", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 5, cv2.LINE_AA)
            # cv2.putText(frame, "Press 's' to play again", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 5, cv2.LINE_AA)
            print("Left hand wins")
            print("Press 's' to play again")
            game_start = False
            game_finish = True
            left_hand= None
            right_hand= None
            left_box_found = False
            right_box_found = False
        elif right_hand == 2 and left_hand == 1:
            # cv2.putText(frame, "Right hand wins", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 5, cv2.LINE_AA)
            # cv2.putText(frame, "Press 's' to play again", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 5, cv2.LINE_AA)
            print("Right hand wins")
            print("Press 's' to play again")
            game_start = False
            game_finish = True
            left_hand= None
            right_hand= None
            left_box_found = False
            right_box_found = False   
            
    if cv2.waitKey(1) & 0xFF == ord('s'):
        game_start = True
        game_finish = False
        

        # Display the results
    cv2.imshow('YOLOv5 Webcam Inference', results.render()[0])

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting the game")
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
