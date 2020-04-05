import cv2
import numpy as np
from sys import argv
from utils import color_segmentation

script, input_video_location, dataset_location, output_destination = argv

if __name__ == '__main__':
    video = cv2.VideoCapture(str(input_video_location))  # Reading the video file
    video_format = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    video_output = cv2.VideoWriter(str(output_destination), video_format, 20.0, (640, 480))
    # Train GMM models for various buoys
    yellow_gmm = color_segmentation.gmm_train(str(dataset_location) + 'Yellow', 1, 50)
    green_gmm = color_segmentation.gmm_train(str(dataset_location) + 'Green', 1, 50)
    orange_gmm = color_segmentation.gmm_train(str(dataset_location) + 'Orange', 4, 200)
    # Start iterating over each video frame
    while True:
        video_frame_exists, img = video.read()
        if not video_frame_exists:
            break
        # print(img.shape)
        # break
        # Create copy of video frame to show buoy detection
        img_result = img.copy()
        # Fitting image
        y_seg, y_img_shape, y_img_thresh, y_img_thresh_slack = color_segmentation.gmm_fit(img, None, yellow_gmm, 1,
                                                                                          10e-3)
        g_seg, g_img_shape, g_img_thresh, g_img_thresh_slack = color_segmentation.gmm_fit(img, None, green_gmm, 20, 10)
        o_seg, o_img_shape, o_img_thresh, o_img_thresh_slack = color_segmentation.gmm_fit(img, 2, orange_gmm, 10e-15,
                                                                                          10e-50)
        seg = [y_seg, g_seg, o_seg]
        img_shape = [y_img_shape, g_img_shape, o_img_shape]
        img_thresh = [y_img_thresh, g_img_thresh, o_img_thresh]
        img_thresh_slack = [y_img_thresh_slack, g_img_thresh_slack, o_img_thresh_slack]

        #  Find bounding box
        color = [(0, 255, 255), (0, 255, 0), (0, 0, 255)]
        img_total = [img.copy(), img.copy(), img.copy()]
        img_show = []
        for i in range(len(seg)):
            contours, _ = cv2.findContours(img_thresh[i], 1, 2)  # opencv3: return contours, cnts, hierarchy.
            x = np.shape(img)[0] - 1
            y = np.shape(img)[1] - 1
            w = h = 0
            if contours:
                area = 100
                for cnt in contours:
                    if cv2.contourArea(cnt) > area:
                        area = cv2.contourArea(cnt)
                        x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(seg[i], (x - int(0.5 * w), y - int(0.5 * h)), (x + int(1.5 * w), y + int(1.5 * h)),
                              (0, 255, 0), 2)

            mask = np.zeros((img_shape[i][0], img_shape[i][1])).astype(np.uint8)
            cv2.rectangle(mask, (x - int(0.5 * w), y - int(1.5 * h)), (x + int(1.5 * w), y + int(1.5 * h)), 255, -1)
            mask = cv2.bitwise_and(mask, img_thresh_slack[i].astype(np.uint8))
            img_show.append(cv2.bitwise_and(img_total[i], img_total[i], mask=mask))
            # Fit circle
            contours, _ = cv2.findContours(mask, 1, 2)
            if contours:
                area = 0
                center = (0, 0)
                radius = 0
                for cnt in contours:
                    if cv2.contourArea(cnt) > area:
                        area = cv2.contourArea(cnt)
                        (x, y), radius = cv2.minEnclosingCircle(cnt)
                        center = (int(x), int(y))
                        radius = int(radius)
                cv2.circle(img_result, center, radius, color[i], 2)

        # cv2.imshow("Result", img_result)
        # key = cv2.waitKey(1)
        # if key == 27:
        #     break
        video_output.write(img_result)
    # Destroy all open-cv windows
    video.release()
    video_output.release()
    cv2.destroyAllWindows()
