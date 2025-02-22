import cv2
import torch
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
import torchvision.transforms as transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet




import torch
import torchvision.models as models

import torch
from torchvision import models

model = models.efficientnet_b0(pretrained=False)  # Ensure correct architecture
state_dict = torch.load("r,C:\Users\nagas\OneDrive\Desktop\New folder\Models\indian_sign_language_resnet_B.pth")  # Load your state_dict
model.load_state_dict(state_dict, strict=False)  # Set strict=False to ignore mismatched keys

for key in state_dict.keys():
    print(key)  # Print keys to compare with model



# --- Corrected Model Loading Section ---
# model_path = r"C:\Users\nagas\OneDrive\Desktop\New folder\Models\indian_sign_language_resnet_B.pth"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Recreate the exact model structure used during training
# model = EfficientNet.from_pretrained('efficientnet-b0')  # Ensure this matches your training
# num_classes = 26  # Ensure this matches your training
# num_features = model._fc.in_features
# model._fc = torch.nn.Linear(num_features, num_classes)

# try:
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     print("Model loaded successfully!") # added print statement to show success.
# except Exception as e:
#     print(f"Error loading model: {e}")
#     exit() # exit if model load fails.

# model.eval().to(device)
# # --- End of Corrected Model Loading Section ---

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize webcam and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300
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

            img_pil = Image.fromarray(cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB))
            img_tensor = transform(img_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img_tensor)
                _, predicted = torch.max(outputs, 1)
                index = predicted.item()

            cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()