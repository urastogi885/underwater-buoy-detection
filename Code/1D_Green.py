import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

video = cv2.VideoCapture('detectbuoy.avi')
count = 0

def mouse_click(event, x, y, flag, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        points.append([x, y])

while video:
    ret , frame = video.read()
    if not ret:
        break
    
    directory = 'E:\Studies\Maryland\ENPM 673 Perception\Project 3\Green'
    points = []
    # resize image to easily crop the ROI
    frame = cv2.resize(frame, (1020,720), interpolation = cv2.INTER_AREA)

    
    cv2.namedWindow("frame", 1)
    cv2.setMouseCallback("frame", mouse_click)
    cv2.imshow('frame', frame)
    cv2.waitKey(0)

    points = np.array(points, np.int32)
    points = points.reshape((-1, 1, 2))
    


    # get buoy from dataset
    rect = cv2.boundingRect(points)
    x, y, w, h = rect
    capture = frame[y:y + h, x:x + w].copy()
    os.chdir(directory)
    cv2.imwrite('green'+str(count)+'.png', capture)
    frame = cv2.polylines(frame, [points], True, (0, 0, 0))
    cv2.imshow('cropped_image', capture)
    cv2.imshow('frame', frame)
    bg = np.ones_like(capture, np.uint8) * 255
    cv2.waitKey(0)
    count += 1
    
images=[]
path = "E:/Studies/Maryland/ENPM 673 Perception/Project 3/Green/"
for image in os.listdir(path):
    images.append(image)
    



video.release()
cv2.destroyAllWindows()
