import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

video = cv2.VideoCapture('detectbuoy.avi')
count = 0

def mouse_click(event, x, y, flag, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        pts.append([x, y])


def plot_histogram(img, title, mask=None):
    chans = cv2.split(img)
    colors = ("b" , "g" ,"r")
    plt.figure()
    plt.title(title)
    plt.xlabel("Bins")
    plt.ylabel("Number of Pixels")

    for (chan,color) in zip(chans,colors):
        histogram = cv2.calcHist([chan],[0],mask,[256],[0,256])
       
        plt.plot(histogram,color = color)
        plt.xlim([0,256])
  
# count = 148
# img_arr = []
# for file_name in glob('images/Yellow/*.png'):
#     img_arr.append(file_name)
# img_arr.sort()
# for file_name in img_arr:
#     file = file_name.split('/')[2]
#     try:
#         if int(file[6:9]) > count:
#             count += 1
#             img = cv2.imread(file_name)
#             cv2.imwrite('Yellow'+str(count)+'.png', img)
#     except:
#         print('Not integer')
    
    
while video:
    ret , video_frame = video.read()
    if not ret:
        break

    # create folder in the directory where you want to store buoy dataset    
    directory = 'E:\Studies\Maryland\ENPM 673 Perception\Project 3\Green_test'
    pts = []
    
    # resizing frame
    re_frame = cv2.resize(video_frame, (1020,720), interpolation = cv2.INTER_AREA)

    # Defining the mouse call back function for cropping
    cv2.namedWindow("frame", 1)
    cv2.setMouseCallback("frame", mouse_click)
    cv2.imshow('frame', re_frame)
    cv2.waitKey(0)

    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1, 1, 2))
    rectangle = cv2.boundingRect(pts)
    x, y, w, h = rectangle
    capture = re_frame[y:y + h, x:x + w].copy()
    
    # Saving Unmasked cropped Images of buoy
    cv2.imwrite('Green_Unmasked_'+str(count)+'.png', capture)
  
    # Directory for saving images
    os.chdir(directory)
    final_frame = cv2.polylines(re_frame, [pts], True, (0, 0, 0))
    cv2.imshow('frame', final_frame)
    
    
    # Applying Masking
    mask = np.zeros(capture.shape[:2], np.uint8)
    points = pts - pts.min(axis=0)
    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
    masked = cv2.bitwise_and(capture, capture, mask=mask)
    cv2.imshow("Masked image",masked)
    cv2.imwrite('Green'+str(count)+'.png', masked)
    cv2.waitKey(0)
    count += 1
    
    
    images=[]
     
    img_save = "E:/Studies/Maryland/ENPM 673 Perception/Project 3/Green_test/"
    for img in os.listdir(img_save):
        images.append(img)
        
    for img in images:
        img_read = cv2.imread("%s%s"%(img_save,img))
        img_blur = cv2.GaussianBlur(img_read,(5,5),0)
        
    plot_histogram(capture,"Green Buoy", mask=mask)  
    
   
  
video.release()
cv2.destroyAllWindows()
