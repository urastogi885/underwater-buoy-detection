from math import sqrt
import numpy as np
import cv2


def crop(img, points):
    # img = cv2.imread('/home/sayani_roy/Documents/Perception-Proj-3/code/frames/frame190.jpg')
    height = img.shape[0]
    width = img.shape[1]

    # mask = np.zeros((height, width, 3), dtype=np.uint8)
    # points = np.array([[[357, 201], [340, 209], [326, 230], [334, 251], [347, 258],
    #                     [379, 256], [387, 242], [391, 225], [381, 210], [358, 200]]])
    center_x = 0
    center_y = 0
    for point in points:
        center_x += point[0]
        center_y += point[1]

    center_x, center_y = (center_x // 8), (center_y // 8)
    radius = sqrt((center_x - points[3][0]) ** 2 + (center_y - points[3][1]) ** 2)
    for y in range(height):
        for x in range(width):
            if (x - center_x) ** 2 + (y - center_y) ** 2 > radius ** 2:
                img[y][x] = (0, 0, 0)

    # res = cv2.bitwise_and(img, img, mask = mask)
    #
    # rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
    # cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    # cv2.imshow('image', img)
    # cv2.imshow("cropped" , cropped )
    # cv2.imshow("same size" , res)
    # cv2.waitKey(0)
    return img