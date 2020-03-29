import cv2
import numpy as np

# TODO: Provide correct path to the project video
# ../ means go to the outer folder
# video = cv2.VideoCapture('../../videos/detectbuoy.avi')
# while True:
#     ret, video_frame = video.read()
#     # If no video frame is generated or the video has ended
#     if not ret:
#         break
#     cv2.imshow("Check", video_frame)
#     cv2.waitKey(0)
#
# video.release()

# This image is extracted from the video. It is the first frame of the video
# Apply blob detector on this image
ref_img = cv2.imread('images/ref_img.png')
# TODO: Define blob detector parameters to detect circles. Refer:
#  https://www.learnopencv.com/blob-detection-using-opencv-python-c/
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = True
params.blobColor = 0
params.filterByCircularity = True
params.minCircularity = 0.8
params.maxCircularity = 1
detector = cv2.SimpleBlobDetector_create(params)
# TODO: Define blob detector and get image with rich blobs
key_points = detector.detect(ref_img)
kp_img = cv2.drawKeypoints(ref_img, key_points, np.array([]), color=(0, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Blobs", kp_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
