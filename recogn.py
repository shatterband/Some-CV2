import numpy as np
import cv2
import face_recognition
import datetime

cap = cv2.VideoCapture(0)
my_masster = face_recognition.load_image_file('my_masster.jpg')

my_masster_face_encoding = face_recognition.face_encodings(my_masster)

print(my_masster_face_encoding)

while True:
    succsess, img = cap.read()
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    imgRGB_face_encoding = face_recognition.face_encodings(img)
    
    if imgRGB_face_encoding:
        is_it_you = face_recognition.compare_faces(my_masster_face_encoding, imgRGB_face_encoding[0])
    
        if is_it_you[0]:
            cv2.putText(img, "it's you, my master", (10, 50),cv2.FONT_HERSHEY_PLAIN, 3 , (0,0,0), 4)
    
    cv2.imshow("It is your ugly face", img)
    
    print(is_it_you, datetime.datetime.now())
    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()