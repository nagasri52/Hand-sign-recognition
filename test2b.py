import cv2
import numpy as np
import math
import torch
from cvzone.HandTrackingModule import HandDetector
from PIL import Image

# Load the YOLOv5 classification model
model_path = r"C:\Users\nagas\OneDrive\Desktop\New folder\Models\YoloV5m-best.pt"
model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, source="local")

# Initialize webcam & hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

# Define class labels (Make sure these match your training labels)
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:  # Ensure valid crop
            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Convert to PIL image for YOLO model
            img_pil = Image.fromarray(cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB))

            # Run YOLO classification
            results = model(img_pil)
            if results and results.pandas().xyxy[0].shape[0] > 0:
                pred_class = int(results.pandas().xyxy[0]['class'][0])  # Get predicted class index
                label = labels[pred_class]

                # Draw output label
                cv2.rectangle(imgOutput, (x - offset, y - offset - 50), 
                              (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, label, (x, y - 26), 
                            cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset), 
                              (x + w + offset, y + h + offset), (255, 0, 255), 4)

        cv2.imshow("Processed Hand Image", imgWhite)  # Show processed image

    cv2.imshow("Webcam Feed", imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
