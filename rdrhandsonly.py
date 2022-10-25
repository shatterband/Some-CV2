import cv2
import mediapipe as mp
import time
import math
import numpy as np
import mouse
import keyboard

cap = cv2.VideoCapture(0)

nonSens = 10

mpDraw = mp.solutions.drawing_utils

pressedL = False
pressedR = False
pressedf = False
pressedT = False

pressedw = False
pressedr = False
pressede = False
pressedS = False

def fingerCount(hand):
    pointing = math.hypot(hand.landmark[8].x - hand.landmark[5].x, hand.landmark[8].y - hand.landmark[5].y)
    middle = math.hypot(hand.landmark[12].x - hand.landmark[9].x, hand.landmark[12].y - hand.landmark[9].y)
    nameless = math.hypot(hand.landmark[16].x - hand.landmark[13].x, hand.landmark[16].y - hand.landmark[13].y)
    lilbud = math.hypot(hand.landmark[20].x - hand.landmark[17].x, hand.landmark[20].y - hand.landmark[17].y)
    
    clsd = max(math.hypot(hand.landmark[13].x - hand.landmark[5].x, hand.landmark[13].y - hand.landmark[5].y), math.hypot(hand.landmark[11].x - hand.landmark[10].x, hand.landmark[11].y - hand.landmark[10].y)*2)

    pointing = pointing > clsd
    middle = middle > clsd
    nameless = nameless > clsd
    lilbud = lilbud > clsd
    
    return [pointing, middle, nameless, lilbud]


mpHands = mp.solutions.hands
hands = mpHands.Hands()

def denormalize(landmark, img):
    ih, iw, ic = img.shape
    x, y = landmark.x, landmark.y
    return (int(x*iw),int(y*ih))


while True:
    succsess, img = cap.read()

    rHand = None
    lHand = None

    ih, iw, ic = img.shape
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    blank = np.zeros((ih,iw,3), dtype='uint8')

    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks) == 2:
            if results.multi_hand_landmarks[0].landmark[0].x > results.multi_hand_landmarks[1].landmark[0].x:
                rHand = results.multi_hand_landmarks[1]
                lHand = results.multi_hand_landmarks[0]
            else:
                rHand = results.multi_hand_landmarks[0]
                lHand = results.multi_hand_landmarks[1]
                
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(blank, handLms, mpHands.HAND_CONNECTIONS)
            
    rCenter = (iw//4, ih//2)
    
    if not None in (rHand, lHand):
        cv2.circle(blank, denormalize(rHand.landmark[0], blank), 5, (255,0,0), cv2.FILLED)
        cv2.circle(blank, denormalize(lHand.landmark[0], blank), 5, (0,255,0), cv2.FILLED)
        
        moveMouse = denormalize(rHand.landmark[9], blank)
        if moveMouse[0] < iw//2:
            mouse.move(-(moveMouse[0] - rCenter[0])//nonSens, (moveMouse[1] - rCenter[1])//nonSens, absolute=False, duration=0)
        cv2.circle(blank, moveMouse, 7, (255,255,0), cv2.FILLED)        
        
        rFingers = fingerCount(rHand)
        lFingers = fingerCount(lHand)
        
        #Rhand : LMB, RMB, f, TAB
        #Lhand : w, r, e, Shift
        
        if rFingers[0] and not pressedL: #LMB
            mouse.press(button="left")
            pressedL = True
        elif not rFingers[0] and pressedL:
            mouse.release(button='left')
            pressedL = False
        if rFingers[1] and not pressedR: #RMB
            mouse.press(button="right")
            pressedR = True
        elif not rFingers[1] and pressedR:
            mouse.release(button='right')
            pressedR = False
        if rFingers[2] and not pressedf: # f
            keyboard.press('f')
            pressedf = True
        elif not rFingers[2] and pressedf:
            keyboard.release('f')
            pressedf = False
        if rFingers[3] and not pressedT: # TAB
            keyboard.press("tab")
            pressedT = True
        elif not rFingers[3] and pressedT:
            keyboard.release("tab")
            pressedT = False
            
        if lFingers[0] and not pressedw: # w
            keyboard.press("w")
            pressedw = True
        elif not lFingers[0] and pressedw:
            keyboard.release("w")
            pressedw = False
        if lFingers[1] and not pressedr: # r
            keyboard.press("r")
            pressedr = True
        elif not lFingers[1] and pressedr:
            keyboard.release("r")
            pressedr = False
        if lFingers[2] and not pressede: # e
            keyboard.press("e")
            pressede = True
        elif not lFingers[2] and pressede:
            keyboard.release("e")
            pressede = False
        if lFingers[3] and not pressedS: # Shift
            keyboard.press("shift")
            pressedS = True
        elif not lFingers[3] and pressedS:
            keyboard.release("shift")
            pressedS = False


    cv2.circle(blank, rCenter, 3, (255,255,0), cv2.FILLED)
    
    cv2.line(blank, (iw//2, 0), (iw//2, ih), (128,128,128), 2)
    
    blankFlip = cv2.flip(blank, 1)
    
    cv2.imshow("I use only my hands", blankFlip)
    cv2.waitKey(1)