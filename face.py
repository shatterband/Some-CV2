import cv2
import mediapipe as mp
import time
import math

cap = cv2.VideoCapture(0)

mpDraw = mp.solutions.drawing_utils
mpFaceDetection = mp.solutions.face_detection
mpFaceMesh = mp.solutions.face_mesh
faceDetection = mpFaceDetection.FaceDetection()
faceMesh = mpFaceMesh.FaceMesh()

blick = 0

while True:
    succsess, img = cap.read()
    ih, iw, ic = img.shape
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = faceDetection.process(imgRGB)
    '''
    if results.detections:
        for detection in results.detections:
            mpDraw.draw_detection(img, detection)
    #'''
    result_ = faceMesh.process(imgRGB)
    cv2.putText(img, str(int(blick)), (10, 50),
    cv2.FONT_HERSHEY_PLAIN, 3 , (255,255,255), 4)
    if result_.multi_face_landmarks:
        for faceLms in result_.multi_face_landmarks:
            #mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS)
            for lm in [faceLms.landmark[164], faceLms.landmark[167], faceLms.landmark[393],faceLms.landmark[92],faceLms.landmark[165], faceLms.landmark[391], faceLms.landmark[322], faceLms.landmark[200], faceLms.landmark[199]]:
                x, y = int(lm.x *iw), int(lm.y*ih)
                cv2.circle(img, (x, y), 10, (0,0,0), cv2.FILLED)
            x, y = int(faceLms.landmark[1].x*iw), int(faceLms.landmark[1].y*ih)
            cv2.circle(img, (x, y), 30, (0,0,255), cv2.FILLED)
        cv2.imshow("Face", img)
    cv2.waitKey(1)
