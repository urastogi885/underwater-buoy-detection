import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

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
    plt.ylabel("# of Pixels")
    #hist_b = np.zeros((256,1))
    #hist_g = np.zeros((256,1))
    #hist_r = np.zeros((256,1))
    for (chan,color) in zip(chans,colors):
        histogram = cv2.calcHist([chan],[0],mask,[256],[0,256])
       # hist = np.column_stack((hist,histogram))
        #avg = np.sum(hist, axis=1) / (hist.shape[1]-1)
        plt.plot(histogram,color = color)
        plt.xlim([0,256])
    
while video:
    ret , video_frame = video.read()
    if not ret:
        break

    # create folder in the directory where you want to store buoy dataset    
    directory = 'E:\Studies\Maryland\ENPM 673 Perception\Project 3\Green'
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
    
'''    
    images=[]
    img_save = "E:/Studies/Maryland/ENPM 673 Perception/Project 3/Green - Copy/"
    for img in os.listdir(img_save):
        images.append(img)
    
    for img in images:
        img_read = cv2.imread("%s%s"%(img_save,img))
        img_blur = cv2.GaussianBlur(img_read,(5,5),0)
    plot_histogram(img_blur,"kjkj")
    plot_histogram(masked, img_blur,"tftftt", mask=mask)
    
'''

    
 


    
video.release()
cv2.destroyAllWindows()
