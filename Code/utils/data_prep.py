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


while video:
    ret , video_frame = video.read()
    if not ret:
        break

    # create folder in the directory where you want to store buoy dataset    
    directory = 'E:\Studies\Maryland\ENPM 673 Perception\Project 3\Yellow_Unmasked'
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
    cv2.imwrite('Yellow_Unmasked_'+str(count)+'.png', capture)
  
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
    cv2.imwrite('Yellow'+str(count)+'.png', masked)
    cv2.waitKey(0)
    count += 1

    images=[]
    
    img_save = "E:/Studies/Maryland/ENPM 673 Perception/Project 3/Orange_Unmasked/"
    for img in os.listdir(img_save):
        images.append(img)
        
    
        
    hist_b = np.zeros((256,1))
    hist_g = np.zeros((256,1))
    hist_r = np.zeros((256,1))
    for img in images:
        img_read = cv2.imread("%s%s"%(img_save,img))
        img_blur = cv2.GaussianBlur(img_read,(5,5),0)
        
        color = ('b' , 'g' , 'r') 
        for i,col in enumerate(color):
            if col == 'b':
                histr_b = cv2.calcHist([img_blur],[i],None,[256],[0,256])
                hist_b = np.column_stack((hist_b,histr_b))
    
            if col == 'g':
                histr_g = cv2.calcHist([img_blur],[i],None,[256],[0,256])
                hist_g = np.column_stack((hist_g,histr_g))     
                
            if col == 'r':
                histr_r = cv2.calcHist([img_blur],[i],None,[256],[0,256])
                hist_r = np.column_stack((hist_r,histr_r))
        
    
    
    avg_hist_r = np.sum(hist_r, axis=1) / (hist_r.shape[1]-1)
    avg_hist_g = np.sum(hist_g, axis=1) / (hist_g.shape[1]-1)
    avg_hist_b = np.sum(hist_b, axis=1) / (hist_b.shape[1]-1)
    plt.plot(avg_hist_r,color = 'r')
    plt.plot(avg_hist_g,color = 'g')
    plt.plot(avg_hist_b,color = 'b')
    plt.title('Histogram for Orange Buoy')
    plt.xlabel("Bins")
    plt.ylabel("Number of Pixels")
    plt.show()

video.release()
cv2.destroyAllWindows()






