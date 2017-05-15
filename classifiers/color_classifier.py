import numpy as np
import os
from pixel_feature_extractor import *
import pdb
from sklearn.svm import LinearSVC

class ColorClassifier:
	COLOR_CODES = dict({'green': 0, 'purple': 1, 'red': 2})

	def __init__(self, train_dir, test_dir):
		self.train_dir = train_dir
		self.test_dir = test_dir
		self.train_X = []
		self.train_Y = []
		self.test_X = []
		self.test_Y = []


	def process_images(self):
		p = PixelFeatureExtractor()
		for filename in os.listdir(self.train_dir):
			if filename.endswith(".jpg"):
				color_code = ColorClassifier.COLOR_CODES[filename.split('_')[0]]
				self.train_X.append(p.get_features(self.train_dir + '/' + filename))
				self.train_Y.append(color_code)
		for filename in os.listdir(self.test_dir):
			if filename.endswith(".jpg"):
				color_code = ColorClassifier.COLOR_CODES[filename.split('_')[0]]
				self.test_X.append(p.get_features(self.test_dir + '/' + filename))
				self.test_Y.append(color_code)

	def train(self):
		lin_clf = LinearSVC()
		lin_clf.fit(self.train_X, self.train_Y) 
		predictions = lin_clf.predict(self.test_X)
		print "Predicted:"
		print predictions
		score = lin_clf.score(self.test_X, self.test_Y)
		print "Score: %f" % score


c = ColorClassifier('../train_images_1', '../test_images_1')
c.process_images()
c.train()