import numpy as np
from scipy import misc
import pdb
import math
from skimage import io, color, img_as_float

class PixelFeatureExtractor:
	NUM_ROWS = 105
	NUM_COLS = 69
	# TODO: Get actual avg aspect ratio

	def __init__(self):
		return

	def get_features(self, imgfile, crop_radius=0):
		img = misc.imread(imgfile)
		resized = misc.imresize(img, (PixelFeatureExtractor.NUM_ROWS, PixelFeatureExtractor.NUM_COLS))
		# resized = resized[crop_radius:PixelFeatureExtractor.NUM_ROWS - crop_radius, crop_radius:PixelFeatureExtractor.NUM_COLS - crop_radius ]

		return np.ndarray.flatten(resized)

	def get_color_features(self, imgfile, crop_radius=0):
		img = misc.imread(imgfile)
		resized = misc.imresize(img, (PixelFeatureExtractor.NUM_ROWS, PixelFeatureExtractor.NUM_COLS))
		image = img_as_float(resized)
		pixels = image.reshape(image.shape[0] * image.shape[1], 3)
		filtered_pixels = np.array([p for p in pixels if 0.4 <= np.linalg.norm(p) <= 1.1])
		if (filtered_pixels.shape[0] == 0):
			print "No filtered pixels"
			pdb.set_trace()
		mean_color = np.mean(filtered_pixels, axis=0)
		# pdb.set_trace()
		return mean_color.flatten()
		
# Example usage
# p = PixelFeatureExtractor()
# p.get_features('../set_images/green_one_oval_hollow.jpg')

