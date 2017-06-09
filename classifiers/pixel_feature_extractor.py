import numpy as np
from scipy import misc
import pdb
import math
import matplotlib.pyplot as plt
import cv2
from skimage import color
from skimage.feature import hog


class PixelFeatureExtractor:
  NUM_ROWS = 105
  NUM_COLS = 69
  # TODO: Get actual avg aspect ratio

  def __init__(self):
    return

  def preprocess(self, img):
    thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

    # Uncomment to visualize
    # plt.subplot(121),plt.imshow(img),plt.title('Input')
    # plt.subplot(122),plt.imshow(thresh),plt.title('Output')
    # plt.show()
    return thresh

  def get_features(self, imgfile):
    # img = cv2.imread(imgfile)
    gray_img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
    # This does not seem to help, so exluding preprocess step for now
    # processed_img = self.preprocess(gray_img)
    resized = misc.imresize(gray_img, (PixelFeatureExtractor.NUM_ROWS, PixelFeatureExtractor.NUM_COLS))

    # Uncomment to see HOG features
    # hog_features, hog_image = hog(cv2.resize(gray_img, (PixelFeatureExtractor.NUM_COLS * 2, PixelFeatureExtractor.NUM_ROWS * 2)), visualise=True)
    # plt.imshow(hog_image)
    # plt.show()
    # pdb.set_trace()

    hog_features = hog(cv2.resize(gray_img, (PixelFeatureExtractor.NUM_COLS * 2, PixelFeatureExtractor.NUM_ROWS * 2)), block_norm='L2')

    return np.append(np.ndarray.flatten(resized), np.ndarray.flatten(hog_features))
    

# Example usage
# p = PixelFeatureExtractor()
# p.get_features('../set_images/green_one_oval_hollow.jpg')

