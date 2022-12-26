import os
import cv2

folder_dir = "D:/Anh chua xu ly/BoneData"
for folder in os.listdir(folder_dir):
    img_dir = folder_dir + "/" + folder + "/" + "rotate"
    for each_folder in os.listdir(img_dir):
        each_folder_dir = img_dir + "/" + each_folder
        for image in os.listdir(each_folder_dir):
            img_link = each_folder_dir + "/" + image
            img = cv2.imread(img_link)
            img = cv2.rotate(img, cv2.ROTATE_180)
            cv2.imwrite(img_link, img)
            print("done")