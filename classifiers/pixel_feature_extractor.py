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

  # def get_features(self, imgfile, crop_radius=0):
  #   img = misc.imread(imgfile)
  #   resized = misc.imresize(img, (PixelFeatureExtractor.NUM_ROWS, PixelFeatureExtractor.NUM_COLS))
  #   # resized = resized[crop_radius:PixelFeatureExtractor.NUM_ROWS - crop_radius, crop_radius:PixelFeatureExtractor.NUM_COLS - crop_radius ]

  #   return np.ndarray.flatten(resized)

  # def preprocess(self, img):
  #   # thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

  #   # Uncomment to visualize
  #   plt.subplot(121),plt.imshow(img),plt.title('Input')
  #   plt.subplot(122),plt.imshow(thresh),plt.title('Output')
  #   plt.show()
  #   return thresh

  def get_features(self, imgfile):
    gray_img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
    processed_img = self.preprocess(gray_img)
    resized = misc.imresize(processed_img, (PixelFeatureExtractor.NUM_ROWS, PixelFeatureExtractor.NUM_COLS))
    

    # Uncomment to see HOG features
    # hog_features, hog_image = hog(cv2.resize(gray_img, (PixelFeatureExtractor.NUM_COLS * 2, PixelFeatureExtractor.NUM_ROWS * 2)), visualise=True)
    # plt.imshow(hog_image)
    # plt.show()
    # pdb.set_trace()

    hog_features = hog(cv2.resize(gray_img, (PixelFeatureExtractor.NUM_COLS * 2, PixelFeatureExtractor.NUM_ROWS * 2)))

    return np.append(np.ndarray.flatten(resized), np.ndarray.flatten(hog_features))
    

# Example usage
# p = PixelFeatureExtractor()
# p.get_features('../set_images/green_one_oval_hollow.jpg')

