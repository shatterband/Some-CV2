import cv2
import mediapipe as mp
import time
import math
import numpy as np

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()

mpPose = mp.solutions.pose
pose = mpPose.Pose()

mpFaceDetection = mp.solutions.face_detection
mpFaceMesh = mp.solutions.face_mesh
faceDetection = mpFaceDetection.FaceDetection()
faceMesh = mpFaceMesh.FaceMesh()


mpDraw = mp.solutions.drawing_utils

fuck = ""

while True:
    succsess, img = cap.read()
    
    ih, iw, ic = img.shape
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    resultsP = pose.process(imgRGB)
    resultsF = faceDetection.process(imgRGB)
    resultFM = faceMesh.process(imgRGB)
    

    blank = np.zeros((ih,iw,3), dtype='uint8')
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(blank, handLms, mpHands.HAND_CONNECTIONS)
    if resultsP.pose_landmarks:
        mpDraw.draw_landmarks(blank, resultsP.pose_landmarks, mpPose.POSE_CONNECTIONS)
    if resultsF.detections:
        for detection in resultsF.detections:
            mpDraw.draw_detection(blank, detection)
    if resultFM.multi_face_landmarks:
        for faceLms in resultFM.multi_face_landmarks:
            mpDraw.draw_landmarks(blank, faceLms, mpFaceMesh.FACEMESH_CONTOURS)
    
    blankFlip = cv2.flip(blank, 1)
    
    cv2.imshow("Image", blankFlip)
    cv2.waitKey(1)
    
    pTime = time.time()