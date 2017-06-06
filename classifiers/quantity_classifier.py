import numpy as np
import os
from pixel_feature_extractor import *
import pdb
from sklearn.svm import LinearSVC
from feature_classifier import *

class QuantityClassifier(FeatureClassifier):
	QUANTITY_CODES = dict({'one': 0, 'two': 1, 'three': 2})

	def __init__(self, train_dir, test_dir, base_dir):
		super(QuantityClassifier, self).__init__(train_dir, test_dir, base_dir, QuantityClassifier.QUANTITY_CODES, 1, 'quantity')

if __name__ == '__main__':
	c = QuantityClassifier('../train_images_1', '../test_images_1', '../set_images')
	c.process_images()
	c.train()