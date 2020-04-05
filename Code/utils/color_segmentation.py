import cv2
import numpy as np
from glob import glob
from utils.gaussian_mixture import gaussian, GaussianMixture


def load_images(location):
    filename_list = []
    images = []
    for filename in glob(location):
        filename_list.append(filename)
    filename_list.sort()
    for file in filename_list:
        # print(file)
        img = cv2.imread(file)
        if img is not None:
            images.append(img)
    return images


def gmm_train(address, k, iteration):
    images = load_images(address + '/*.png')

    hist_bgr = np.zeros((256, 256, 256))
    for img in images:
        bgr_update = cv2.calcHist([img], [0, 1, 2], None, [256, 256, 256], [50, 256, 50, 256, 50, 256])
        hist_bgr += bgr_update
    hist_bgr /= len(images)
    hist_bgr /= hist_bgr.sum()

    num_data = 2
    train = np.zeros((1, 3))  # Initialize training data
    for scale in range(1, num_data):
        scale /= num_data
        scale *= hist_bgr.max()
        train_tmp = np.stack(np.where(hist_bgr >= scale), axis=1)
        if scale == 1 / num_data:
            train = train_tmp
        else:
            train = np.concatenate((train, train_tmp))
    train /= 255  # Training data normalization
    gmm_model = GaussianMixture(train, k)

    for i in range(iteration):
        gmm_model.train()
    print(address, 'trained')
    return gmm_model


def gmm_fit(img, k1, model, threshold1, threshold2):
    img_shape = np.shape(img)
    # Apply gaussian blur to image
    blur = cv2.GaussianBlur(img, (11, 11), 0)
    # Test data normalization
    tmp = np.reshape(blur, (-1, 3)) / 255

    if k1:
        prob = gaussian(tmp, model.get_model()[0][k1], model.get_model()[1][k1]).reshape((-1))
        prob = np.reshape(prob, (img_shape[0], img_shape[1]))

    else:
        prob = model.get_pdf(tmp)
        prob = np.reshape(prob, (img_shape[0], img_shape[1]))

    _, img_thresh = cv2.threshold(prob, threshold1, 255, cv2.THRESH_BINARY)
    _, img_thresh_slack = cv2.threshold(prob, threshold2, 255, cv2.THRESH_BINARY)
    img_thresh = img_thresh.astype(np.uint8)
    seg = cv2.bitwise_and(img, img, mask=img_thresh)  # segmentaion
    return seg, img_shape, img_thresh, img_thresh_slack
