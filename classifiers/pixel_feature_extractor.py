import numpy as np
from scipy import misc
import pdb
import math


class PixelFeatureExtractor:
	NUM_ROWS = 105
	NUM_COLS = 69
	# TODO: Get actual avg aspect ratio

	def __init__(self):
		return

	def get_features(self, imgfile):
		img = misc.imread(imgfile)
		resized = misc.imresize(img, (PixelFeatureExtractor.NUM_ROWS, PixelFeatureExtractor.NUM_COLS))
		return np.ndarray.flatten(resized)
		

# Example usage
# p = PixelFeatureExtractor()
# p.get_features('../set_images/green_one_oval_hollow.jpg')

