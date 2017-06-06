import numpy as np
import os
from pixel_feature_extractor import *
import pdb
from sklearn.svm import LinearSVC
from feature_classifier import *

class ColorClassifier(FeatureClassifier):
	COLOR_CODES = dict({'green': 0, 'purple': 1, 'red': 2})

	def __init__(self, train_dir, test_dir, base_dir):
		super(ColorClassifier, self).__init__(train_dir, test_dir, base_dir, ColorClassifier.COLOR_CODES, 0, 'color')


if __name__ == '__main__':
	c = ColorClassifier('../train_images_1', '../test_images_1', '../set_images')
	c.process_images()
	c.train()