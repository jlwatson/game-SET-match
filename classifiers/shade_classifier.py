import numpy as np
import os
from pixel_feature_extractor import *
import pdb
from sklearn.svm import LinearSVC
from feature_classifier import *

class ShadeClassifier(FeatureClassifier):
	SHADE_CODES = dict({'stripe': 0, 'solid': 1, 'hollow': 2})

	def __init__(self, train_dir, test_dir, base_dir):
		super(ShadeClassifier, self).__init__(train_dir, test_dir, base_dir, ShadeClassifier.SHADE_CODES, 3, 'shade')

if __name__ == '__main__':
  c = ShadeClassifier('../train_images_1', '../test_images_1', '../set_images')
  c.process_images()
  c.train()

