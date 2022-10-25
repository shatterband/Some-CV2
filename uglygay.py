import cv2
import mediapipe as mp
import time
import math
import imutils

cap = cv2.VideoCapture(0)

mpDraw = mp.solutions.drawing_utils
mpFaceDetection = mp.solutions.face_detection
mpFaceMesh = mp.solutions.face_mesh
faceDetection = mpFaceDetection.FaceDetection()
faceMesh = mpFaceMesh.FaceMesh()



def denormalize(landmark, img):
    ih, iw, ic = img.shape
    x, y = landmark.x, landmark.y
    return (int(x*iw),int(y*ih))

def find_angle(vector_x, vector_y):
    len = math.hypot(vector_x, vector_y)
    vector_x = vector_x/len
    vector_y = vector_y/len
    if vector_x > 0:
        angle = 180 + math.degrees(math.acos(vector_y))
    else:
        angle = 180 - math.degrees(math.acos(vector_y))
    return angle

angle = 0

while True:
    succsess, img = cap.read()
    
    ih, iw, ic = img.shape
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    result = faceMesh.process(imgRGB)
    
    if result.multi_face_landmarks:
        for faceLms in result.multi_face_landmarks:
            max_x, max_y, min_x, min_y = 0,0,1,1
            for landmark in faceLms.landmark:
                if landmark.x > max_x:
                    max_x = landmark.x
                if landmark.x < min_x:
                    min_x = landmark.x
                if landmark.y > max_y:
                    max_y = landmark.y
                if landmark.y < min_y:
                    min_y = landmark.y
            #mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS)
            cv2.line(img, denormalize(faceLms.landmark[10], img) , denormalize(faceLms.landmark[152], img) , (0,255,0), 1)
            cv2.line(img, denormalize(faceLms.landmark[454], img) , denormalize(faceLms.landmark[234], img) , (0,255,0), 1)
            angle = find_angle(faceLms.landmark[10].x-faceLms.landmark[152].x, faceLms.landmark[10].y-faceLms.landmark[152].y)
            max_x, max_y, min_x, min_y = int(max_x*iw), int(max_y*ih), int(min_x*iw), int(min_y*ih)
            img = img[min_y:max_y , min_x:max_x]
    
        
    
    imgR = imutils.rotate(img, angle=-angle)
    cv2.imshow("Face", imgR)
    cv2.waitKey(1)
