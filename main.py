import os

import cv2
import numpy as np
import math
import mediapipe as mp

class HandDetector:

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []
    def findHands(self, img, draw=True, flipType=True):
        imgdraw = np.ones((img.shape[0], img.shape[1],3), np.uint8) * 255
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                ## lmList
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                ## draw
                if draw:

                    self.mpDraw.draw_landmarks(imgdraw, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
                    #cv2.rectangle(imgdraw, (bbox[0] - 20, bbox[1] - 20),
                                  #(bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  #(255, 0, 255), 2)
                    cv2.putText(imgdraw, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)
        if draw:
            return allHands, imgdraw
        else:
            return allHands

        cap = cv2.VideoCapture(0)
        detector = HandDetector(maxHands=1)
        offset = 20
        imgSize = 300

        folder_dir = "D:\Data\480"
        counter = 0
        for video in folder_dir:
            video_dir = os.path.join(folder_dir, video)
            print(video_dir)
            '''
            writer = cv2.VideoWriter('D:\Dao.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (1920, 1080))
            while True:
                try:
                    success, img = cap.read()
                    ret, frame = cap.read()
                    writer.write(frame)
                    hands, img = detector.findHands(img)
                    if hands:
                        hand = hands[0]
                        x, y, w, h = hand['bbox']
                        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
                        imgCropShape = imgCrop.shape
    
                        aspectRatio = h / w
    
                        if aspectRatio > 1:
                            k = imgSize / h
                            wCal = math.ceil(k * w)
                            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                            imgResizeShape = imgResize.shape
                            wGap = math.ceil((imgSize - wCal) / 2)
                            imgWhite[:, wGap:wCal + wGap] = imgResize
    
                        else:
                            k = imgSize / w
                            hCal = math.ceil(k * h)
                            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                            imgResizeShape = imgResize.shape
                            hGap = math.ceil((imgSize - hCal) / 2)
                            imgWhite[hGap:hCal + hGap, :] = imgResize
    
                        cv2.imshow("ImageCrop", imgCrop)
                        cv2.imshow("ImageWhite", imgWhite)
    
                    cv2.imshow("Image", img)
                    key = cv2.waitKey(1)
                    if key == ord("s"):
                        # counter += 1
                        # cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
                        # print(counter)
                        cap.release()
                        writer.release()
                        cv2.destroyAllWindows()
                except:
                    continue
    '''