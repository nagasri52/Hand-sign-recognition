
import cv2
import torch
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

# Load the trained PyTorch model
model_path = r"C:\Users\nagas\OneDrive\Desktop\New folder\Models\indian_sign_language_resnet2.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet50 model structure and weights
model = models.resnet50(pretrained=False)
num_classes = 36  # Adjust based on your dataset
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

# Define image transformations for color input
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet50 input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for color images
])

# Initialize webcam & hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

# Define class labels
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'k', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        
        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
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
            
            # Convert OpenCV image to PIL, apply transformations, and predict

            
            # img_pil = Image.fromarray(cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB))
            # img_tensor = transform(img_pil).unsqueeze(0).to(device)


            img_pil = Image.fromarray(cv2.cvtColor(imgResize, cv2.COLOR_BGR2RGB))
            img_tensor = transform(img_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                _, predicted = torch.max(outputs, 1)
                index = predicted.item()

            # Draw output label
            cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
        
        cv2.imshow("ImageWhite", imgWhite)  # Show processed image

    cv2.imshow("Image", imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()