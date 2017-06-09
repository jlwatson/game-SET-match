import numpy as np
from scipy import misc
import pdb
import math
import matplotlib.pyplot as plt
import cv2
from skimage import color
from skimage.feature import hog


class RowPixelFeatureExtractor:
  NUM_ROWS = 105
  NUM_COLS = 69
  ROW_FRACTIONS = [.22, .35, .5, .65, .78]
  COL_FRACTIONS = [.144, .855]
  # TODO: Get actual avg aspect ratio

  def __init__(self):
    return

  def get_features(self, imgfile):
    img = cv2.imread(imgfile)
    gray_img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
    # This does not seem to help, so exluding preprocess step for now
    # processed_img = self.preprocess(gray_img)
    resized = misc.imresize(gray_img, (RowPixelFeatureExtractor.NUM_ROWS, RowPixelFeatureExtractor.NUM_COLS))
    row_indices = np.rint(np.array(RowPixelFeatureExtractor.ROW_FRACTIONS * RowPixelFeatureExtractor.NUM_ROWS)).astype(int)
    rows = resized[row_indices, :]

    
    return np.ndarray.flatten(rows)
    

# Example usage
# p = PixelFeatureExtractor()
# p.get_features('../set_images/green_one_oval_hollow.jpg')

