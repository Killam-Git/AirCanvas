import os
import numpy as np
import cv2
import HandTrackingModule as htm

UIPath = "UI"
mylist = sorted(os.listdir(UIPath))
print(mylist)
overlayList = []

for imgPath in mylist:
    img = cv2.imread(f'{UIPath}/{imgPath}')
    overlayList.append(img)
print(len(overlayList))
header = overlayList[0]
draw_color = (255,255,255)

video = cv2.VideoCapture(0)
video.set(3, 1280)
video.set(4,720)

hand_detector = htm.handDetector(detectionCon=0.7, maxHands=1)
xp, yp = 0,0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

brushThickness = 15
IndexFingerXP, IndexFingerYP = 0,0

isEraser = False

while True:
    success, img = video.read()
    img = cv2.flip(img, 1)
    img = hand_detector.findHands(img)
    LandMarklist = hand_detector.findPosition(img, draw=False)

    if len(LandMarklist) != 0:
        print(LandMarklist[8])
        IndexFingerX, IndexFingerY = LandMarklist[8][1:]
        MiddleFingerX, MiddleFingerY = LandMarklist[12][1:]

        fingers = hand_detector.fingersUp()

        if fingers[1] and fingers[2]:
            cv2.rectangle(img, (IndexFingerX, IndexFingerY -25), (MiddleFingerX, MiddleFingerY + 25), draw_color, cv2.FILLED)
            IndexFingerXP, IndexFingerYP = IndexFingerX, IndexFingerY
            print("Selecting")

            if IndexFingerY < 125:

                if IndexFingerY < 200 < IndexFingerX < 330:
                    header = overlayList[0]
                    isEraser = False
                    # draw_color = (255,255,255)
                    draw_color = (255,255,255)

                if IndexFingerY < 400 < IndexFingerX < 630:
                    header = overlayList[1]
                    isEraser = False
                    draw_color = (0,0,255)
                
                if IndexFingerY < 600 < IndexFingerX < 930:
                    header = overlayList[2]
                    isEraser = False
                    draw_color = (0,255,0)

                if IndexFingerY < 840 < IndexFingerX < 1100:
                    header = overlayList[3]
                    isEraser = False
                    draw_color = (255,0,0)


                if IndexFingerY < 1150 < IndexFingerX < 1250:
                    header = overlayList[4]
                    isEraser = True
                    draw_color = (0,0,0)
        
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (IndexFingerX, IndexFingerY), 15, draw_color, cv2.FILLED)
            print("Drawing now")

            if IndexFingerXP == 0 and IndexFingerYP ==0:
                IndexFingerXP, IndexFingerYP = IndexFingerX, IndexFingerY
            
            if(isEraser):
                brushThickness = 70
            else:
                brushThickness = 15
            
            cv2.line(img, (IndexFingerXP, IndexFingerYP),(IndexFingerX, IndexFingerY), draw_color, brushThickness)
            cv2.line(imgCanvas, (IndexFingerXP, IndexFingerYP),(IndexFingerX, IndexFingerY), draw_color, brushThickness)

            
            IndexFingerXP, IndexFingerYP = IndexFingerX, IndexFingerY
            
        if all (x >= 1 for x in fingers):
            imgCanvas = np.zeros((720, 1280, 3), np.uint8)
            print("All up")


    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInverse = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInverse = cv2.cvtColor(imgInverse, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInverse)
    img = cv2.bitwise_or(img, imgCanvas)


    img[0:125, 0:1280] = header
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Webcam",img)
    # cv2.imshow("Canvas",imgCanvas)
    cv2.waitKey(1)
