import cv2
import pylab as pl
from roipoly import roipoly
import crop_image

img = cv2.imread('/home/sayani_roy/Documents/Perception-Proj-3/code/frames/frame4.jpg')
b, g, r = cv2.split(img)
img = cv2.merge([r, g, b])

pl.imshow(img)
ROI1 = roipoly(roicolor='r')
# all_pos = []
# print(all_pos)

pl.imshow(img)
ROI1.displayROI()

# ROI1.displayMean(img)
pl.show()

img = cv2.merge([b, g, r])

if ROI1.buoy_points is not None:
    cropped_img = crop_image.crop(img.copy(), ROI1.buoy_points)
    # cv2.imshow("original", img)
    cv2.imshow('buoy', cropped_img)
    # cv2.waitKey(0)
    cv2.imwrite("/home/sayani_roy/Documents/Perception-Proj-3/code/yellow_buoys/frame3.jpg", cropped_img)




