import numpy as np
import os
from pixel_feature_extractor import *
import pdb
from sklearn.svm import LinearSVC
from feature_classifier import *


class ShapeClassifier(FeatureClassifier):
	SHAPE_CODES = dict({'oval': 0, 'squiggle': 1, 'rhombus': 2})

	def __init__(self, train_dir, test_dir, base_dir):
		super(ShapeClassifier, self).__init__(train_dir, test_dir, base_dir, ShapeClassifier.SHAPE_CODES, 2, 'shape')

if __name__ == '__main__':
  c = ShapeClassifier('../train_images_1', '../test_images_1', '../set_images')
  c.process_images()
  c.train()