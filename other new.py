import cv2
import torch
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

# Load the trained PyTorch model
model_path = r"c:\Users\nagas\Downloads\indian_sign_language_resnet.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet50 model structure and weights
model = models.resnet50(pretrained=False)
num_classes = 36  # Adjust based on dataset
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
])

# Initialize webcam & hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  # Detects two hands

offset = 20
imgSize = 300

# Define class labels
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    predictions = []  # Store predictions for both hands

    if hands and len(hands) == 2:  # Ensure two hands are detected
        for i, hand in enumerate(hands):
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

                # Convert OpenCV image to PIL, apply transformations, and predict
                img_pil = Image.fromarray(cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB))
                img_tensor = transform(img_pil).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(img_tensor)
                    _, predicted = torch.max(outputs, 1)
                    predicted_letter = labels[predicted.item()]
                    predictions.append(predicted_letter)  # Store letter prediction
            
            # Display cropped images
            cv2.imshow(f"Hand {i+1}", imgWhite)

            # Draw bounding box with labels for each hand
            color = (0, 255, 0) if i == 0 else (0, 0, 255)  # Green for Hand 1, Red for Hand 2
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), color, 4)
            cv2.putText(imgOutput, f"Hand {i+1}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # Combine results from both hands
        if len(predictions) == 2:
            final_prediction = predictions[0] + predictions[1]  # Simple concatenation logic
            
            # Display final prediction on screen
            cv2.putText(imgOutput, f"Prediction: {final_prediction}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)

    cv2.imshow("Image", imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
