import numpy as np
import os
from pixel_feature_extractor import *
import pdb
from sklearn.svm import LinearSVC
from feature_classifier import *
import sys
import json

class ColorClassifier(FeatureClassifier):
	COLOR_CODES = dict({'green': 0, 'purple': 1, 'red': 2})

	QUANTITY_CODES = dict({'one': 0, 'two': 1, 'three': 2})
	SHADE_CODES = dict({'stripe': 0, 'solid': 1, 'hollow': 2})
	SHAPE_CODES = dict({'oval': 0, 'squiggle': 1, 'rhombus': 2})

	def __init__(self, train_dir, test_dir, base_dir, img_features=None):
		if img_features is None:
			super(ColorClassifier, self).__init__(train_dir, test_dir, base_dir, ColorClassifier.COLOR_CODES, 0)
		else:
			self.train_dir = train_dir
			self.test_dir = test_dir
			self.base_dir = base_dir
			self.train_X = []
			self.train_Y = []
			self.test_X = []
			self.test_Y = []
			self.codes = ColorClassifier.COLOR_CODES
			self.feature_index = 0
			self.img_features = img_features
			# shape, shade, quantity


	def process_images(self):
		if self.img_features is None:
			super(ColorClassifier, self).process_images()
		else:
			p = PixelFeatureExtractor()
			for filename in os.listdir(self.train_dir):
				if filename.endswith(".jpg"):
					trim_name = filename.split('.')[0]
					shape_shade_quant = [
						ColorClassifier.SHAPE_CODES[trim_name.split('_')[2]],
						ColorClassifier.SHADE_CODES[trim_name.split('_')[3]],
						ColorClassifier.QUANTITY_CODES[trim_name.split('_')[1]]
					]
					color_code = self.codes[trim_name.split('_')[self.feature_index]]
					self.train_X.append(np.append(p.get_features(self.train_dir + '/' + filename), np.array(shape_shade_quant)))
					self.train_Y.append(color_code)
			for filename in os.listdir(self.test_dir):
				if filename.endswith(".jpg"):
					trim_name = filename.split('.')[0]
					color_code = self.codes[trim_name.split('_')[self.feature_index]]
					self.test_X.append(np.append(p.get_features(self.base_dir + '/' + filename), np.array(self.img_features[filename])))
					self.test_Y.append(color_code)

	def process_images_split_dir(self, train_names, test_names):
		if self.img_features is None:
			super(ColorClassifier, self).process_images_split_dir(train_names, test_names)
		else:
			p = PixelFeatureExtractor()
			for filename in train_names:
				trim_name = filename.split('.')[0]
				shape_shade_quant = [
					ColorClassifier.SHAPE_CODES[trim_name.split('_')[2]],
					ColorClassifier.SHADE_CODES[trim_name.split('_')[3]],
					ColorClassifier.QUANTITY_CODES[trim_name.split('_')[1]]
				]
				color_code = self.codes[trim_name.split('_')[self.feature_index]]
				self.train_X.append(np.append(p.get_features(self.train_dir + '/' + filename), np.array(shape_shade_quant)))
				self.train_Y.append(color_code)
			for filename in test_names:
				trim_name = filename.split('.')[0]
				color_code = self.codes[trim_name.split('_')[self.feature_index]]
				self.test_X.append(np.append(p.get_features(self.base_dir + '/' + filename), np.array(self.img_features[filename])))
				self.test_Y.append(color_code)


if __name__ == '__main__':
	args = sys.argv[1:]
	if len(args) > 0:
		features_file = open(args[0], 'r')
		img_features = json.load(features_file)
		c = ColorClassifier('../train_images_1', '../output', '../set_images', img_features)
	else:
		c = ColorClassifier('../train_images_1', '../test_images_1', '../set_images')
	c.process_images()
	c.train()