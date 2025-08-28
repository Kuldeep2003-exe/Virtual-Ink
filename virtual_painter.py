import cv2
import numpy as np
import os
import HandTrackingModule as htm  # make sure handtracking.py is in the same folder

#######################
brushThickness = 15
eraserThickness = 100
########################

# Load header images (toolbar icons)
folderPath = "Header"
myList = sorted(os.listdir(folderPath))
overlayList = [cv2.resize(cv2.imread(f'{folderPath}/{imPath}'), (1280, 125)) for imPath in myList]

header = overlayList[0]   # default toolbar image
drawColor = (255, 0, 255)

# Try default camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.HandDetector(detectionCon=0.7, maxHands=1)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)

    # Detect hand
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)  # ✅ FIXED unpacking

    if lmList and len(lmList) > 12:
        # fingertip coordinates
        x1, y1 = lmList[8][1:3]
        x2, y2 = lmList[12][1:3]

        fingers = detector.fingersUp()

        # Selection Mode (index & middle fingers up)
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]; drawColor = (255, 0, 255)
                elif 550 < x1 < 750:
                    header = overlayList[1]; drawColor = (255, 0, 0)
                elif 800 < x1 < 950:
                    header = overlayList[2]; drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]; drawColor = (0, 0, 0)

        # Drawing Mode (only index finger up)
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):  # eraser
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:  # brush
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    # Merge canvas with video feed
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Show toolbar
    img[0:125, 0:1280] = header

    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()