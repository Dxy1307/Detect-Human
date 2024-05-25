from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

import glob
import matplotlib.patches as patches
import cv2
import imutils
import matplotlib.pyplot as plt

 
# Khởi tạo một bộ mô tả đặc trưng HOG
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

for i, imagePath in enumerate(glob.glob('img/he_2023/*.jpg')):
    if i <= 6:
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width = min(1000, image.shape[1]))
        orig = image.copy()
        
        plt.figure(figsize = (8, 6))
        # 1. Bounding box với ảnh gốc
        # Khởi tạo plot
        ax1 = plt.subplot(1, 2, 1)
        
        # Phát hiện người trong ảnh
        (rects, weights) = hog.detectMultiScale(img = image, winStride = (4, 4),
                                               padding = (8, 8), scale = 1.05)
        print('weights: ', weights)
        # Vẽ các bounding box xung quanh ảnh gốc
        for (x, y, h, w) in rects:
            cv2.rectangle(orig, (x, y), (x+w, y+h), (0, 0, 255), 2)
            # rectFig = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')
            # ax1.imshow(orig)
            # ax1.add_patch(rectFig)
            # plt.title('Ảnh trước non max suppression')

        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        print('rects: ', rects.shape)
        # Sử dụng non max suppression để lấy ra bounding box cuối cùng với ngưỡng threshold = 0.65
        pick = non_max_suppression(rects, probs = None, overlapThresh=0.65)
        
        # 2. Bounding box với ảnh suppression
        # Khởi tạo plot
        ax2 = plt.subplot(1, 2, 2)
        # Vẽ bounding box cuối cùng trên ảnh
        for (xA, yA, xB, yB) in pick:
            w = xB-xA
            h = yB-yA
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
            # Hiển thị hình ảnh
            # plt.imshow(image)
            # plt.title('Ảnh sau non max suppression')
            # rectFig = patches.Rectangle((xA, yA),w,h,linewidth=1,edgecolor='r',facecolor='none')
            # ax2.add_patch(rectFig)
            
        # Lấy thông tin ảnh
        filename = imagePath[imagePath.rfind("\\") + 1:]
        print("[INFO] {}: {} original boxes, {} after suppression".format(
            filename, len(rects), len(pick)))

        cv2.imshow("Before NMS", orig)
        cv2.imshow("After NMS", image)
        cv2.waitKey(0)