import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

def gaussian(x, mu, sig):
    gauss =  ((1/(sig*math.sqrt(2*math.pi)))*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))
    return gauss

images=[]
   
img_save = "E:/Studies/Maryland/ENPM 673 Perception/Project 3/Yellow_Unmasked/"
for img in os.listdir(img_save):
    images.append(img)
    
for img in images:
    img_read = cv2.imread("%s%s"%(img_save,img))
    img_blur = cv2.GaussianBlur(img_read,(5,5),0)


(mean, std) = cv2.meanStdDev(img_blur)
x_values = list(range(0, 255))
    
green_mean = mean[1]
green_std = std[1]
red_mean = mean[2]
red_std = std[2]

gauss_green= gaussian(x_values, green_mean, green_std)
gauss_red = gaussian(x_values, red_mean, red_std)
 
green = plt.plot(gauss_green)
plt.show(green)

red = plt.plot(gauss_red)
plt.show(red)



