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
    def findHands(self, img, draw=True):
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

                allHands.append(myHand)
                # draw
                if draw:
                    self.mpDraw.draw_landmarks(imgdraw, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        if draw:
            return allHands, imgdraw
        else:
            return allHands
    def findHands(self, img, draw=True):
        imgDraw = np.ones((img.shape[0], img.shape[1],3), np.uint8) * 255
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

                allHands.append(myHand)
                # draw
                if draw:
                    self.mpDraw.draw_landmarks(imgDraw, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        if draw:
            return allHands, imgDraw, img
        else:
            return allHands

dataDir = "D:/d"
print(dataDir)
boneDir = "D:/BoneData"
handDir = "D:/HandData"
counter = 16

for charDir in os.listdir(dataDir):
    videoDir = dataDir + "/" + charDir
    print(videoDir)
    video_bone_dir = boneDir + "/" + charDir
    #os.mkdir(video_bone_dir)
    video_hand_dir = handDir + "/" + charDir
    #os.mkdir(video_hand_dir)
    for video in os.listdir(videoDir):
        each_video_dir = videoDir + "/" + video
        print(each_video_dir)
        each_video_bone_dir = video_bone_dir + "/" + video.rstrip(".mp4")
        os.mkdir(each_video_bone_dir)
        print(each_video_bone_dir)
        each_video_hand_dir = video_hand_dir + "/" + video.rstrip(".mp4")
        os.mkdir(each_video_hand_dir)
        print(each_video_hand_dir)

        cap = cv2.VideoCapture(each_video_dir)
        frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(frame_total)

        detector = HandDetector(maxHands=1)
        offset = 20
        imgSize = 300
        count =0
        while True:
            try:
                ret, frame = cap.read()
                img = frame
                hands, imgBone, imgHand = detector.findHands(img)
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']
                    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                    imgBlack = np.zeros((imgSize, imgSize, 3), np.uint8)
                    imgBoneCrop = imgBone[y - offset:y + h + offset, x - offset:x + w + offset]
                    imgHandCrop = imgHand[y - offset:y + h + offset, x - offset:x + w + offset]
                    imgCropShape = imgBoneCrop.shape
                    aspectRatio = h / w
                    if aspectRatio > 1:
                        k = imgSize / h
                        wCal = math.ceil(k * w)
                        imgBoneResize = cv2.resize(imgBoneCrop, (wCal, imgSize))
                        imgHandResize = cv2.resize(imgHandCrop, (wCal, imgSize))
                        imgResizeShape = imgBoneResize.shape
                        wGap = math.ceil((imgSize - wCal) / 2)
                        imgWhite[:, wGap:wCal + wGap] = imgBoneResize
                        imgBlack[:, wGap:wCal + wGap] = imgHandResize
                    else:
                        k = imgSize / w
                        hCal = math.ceil(k * h)
                        imgBoneResize = cv2.resize(imgBoneCrop, (imgSize, hCal))
                        imgHandResize = cv2.resize(imgHandCrop, (imgSize, hCal))
                        imgResizeShape = imgBoneResize.shape
                        hGap = math.ceil((imgSize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgBoneResize
                        imgBlack[hGap:hCal + hGap, :] = imgHandResize
                    cv2.imshow("ImageHand", imgBlack)
                    hand_frame_dir = each_video_hand_dir + '/' + 'hand' + str(counter) + '_' + str(count) + '.jpg'

                    cv2.imwrite(hand_frame_dir, imgBlack)
                    print(hand_frame_dir)
                    bone_frame_dir = each_video_bone_dir + '/' + 'bone' + str(counter) + '_' + str(count) + '.jpg'
                    cv2.imwrite(bone_frame_dir, imgWhite)
                    print(bone_frame_dir)
                count = count + 1
                cv2.imshow("Video", frame)
                key = cv2.waitKey(1)

            except:
                continue

            print(count)
            if count == frame_total-60:
                break
        counter = counter+1