import numpy as np
from scipy import misc
import pdb
import math
# from skimage.feature import daisy
# from skimage import data, color, exposure


class PixelFeatureExtractor:
	NUM_ROWS = 105
	NUM_COLS = 69
	# TODO: Get actual avg aspect ratio

	def __init__(self):
		return

	def get_features(self, imgfile, crop_radius=5):
		img = misc.imread(imgfile)
		resized = misc.imresize(img, (PixelFeatureExtractor.NUM_ROWS, PixelFeatureExtractor.NUM_COLS))
		resized = resized[crop_radius:PixelFeatureExtractor.NUM_ROWS - crop_radius, crop_radius:PixelFeatureExtractor.NUM_COLS - crop_radius ]
		# resized_gray = color.rgb2gray(resized)
		# daisy_features = daisy(resized_gray)
		# pdb.set_trace()

		return np.ndarray.flatten(resized)
		

# Example usage
# p = PixelFeatureExtractor()
# p.get_features('../set_images/green_one_oval_hollow.jpg')

