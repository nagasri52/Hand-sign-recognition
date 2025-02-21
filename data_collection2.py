import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Initialize camera and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300
folder = r"C:\Users\nagas\OneDrive\Desktop\New folder\J"
counter = 0

while True:
    success, img = cap.read()
    if not success:
        continue

    # Convert the entire image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert grayscale back to 3 channels
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Detect hands (without drawing)
    hands, _ = detector.findHands(img, draw=False)  

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Extract hand region from original color image
        img_hand = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Overlay the colored hand on the grayscale image
        gray_3ch[y - offset:y + h + offset, x - offset:x + w + offset] = img_hand

        # Manually draw landmarks and connections
        for lm in hand["lmList"]:
            cx, cy = lm[:2]
            cv2.circle(gray_3ch, (cx, cy), 5, (0, 255, 0), -1)  # Green circles for landmarks

        for connection in detector.fingers:
            for i in range(len(connection) - 1):
                pt1 = tuple(hand["lmList"][connection[i]][:2])
                pt2 = tuple(hand["lmList"][connection[i + 1]][:2])
                cv2.line(gray_3ch, pt1, pt2, (0, 255, 255), 2)  # Yellow lines for connections

        # Prepare white image for cropping
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

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

        # Display images
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # Show final processed image
    cv2.imshow("Image", gray_3ch)

    # Save image on keypress
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(f"Saved image {counter}")
