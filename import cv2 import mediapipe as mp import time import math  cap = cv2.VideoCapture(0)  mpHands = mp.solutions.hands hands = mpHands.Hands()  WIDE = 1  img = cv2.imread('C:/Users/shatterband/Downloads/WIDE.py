import cv2
import mediapipe as mp
import time
import math

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()

WIDE = 1

img = cv2.imread('C:/Users/shatterband/Downloads/4fs_D4Uieh0.jpg') # image to read

while True:
    succsess, img = cap.read()
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if not results.multi_hand_landmarks is None:
        for handLms in results.multi_hand_landmarks:
            WIDE = 1 + math.sqrt((handLms.landmark[8].x - handLms.landmark[4].x)**2 + (handLms.landmark[8].y - handLms.landmark[4].y)**2)
    else :
        WIDE = 1
    imgWIDE = cv2.resize(img, (int(512*WIDE**4), 512))
    time.sleep(0.1)
    GRAY = cv2.cvtColor(imgWIDE, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(imgWIDE, 125, 175)
    cv2.imshow("picture", canny)

    cv2.waitKey(1)
