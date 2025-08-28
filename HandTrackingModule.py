import cv2
import mediapipe as mp
import time
import math
import numpy as np


class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5, modelComplexity=1):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.modelComplexity = modelComplexity

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            model_complexity=self.modelComplexity,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

        self.results = None
        self.lmList = []

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList, yList = [], []
        bbox = ()
        self.lmList = []

        if self.results and self.results.multi_hand_landmarks:
            if handNo >= len(self.results.multi_hand_landmarks):
                return self.lmList, bbox

            myHand = self.results.multi_hand_landmarks[handNo]
            h, w, _ = img.shape

            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            if xList and yList:
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = (xmin, ymin, xmax, ymax)
                if draw:
                    cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):
        fingers = [0, 0, 0, 0, 0]
        if not self.lmList or len(self.lmList) <= max(self.tipIds):
            return fingers

        # Thumb
        fingers[0] = 1 if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1] else 0

        # Other four fingers
        for i in range(1, 5):
            fingers[i] = 1 if self.lmList[self.tipIds[i]][2] < self.lmList[self.tipIds[i] - 2][2] else 0

        return fingers

    def findDistance(self, p1, p2, img=None, draw=True, r=15, t=3):
        if not self.lmList or p1 >= len(self.lmList) or p2 >= len(self.lmList):
            return None, img, []
