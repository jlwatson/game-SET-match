import numpy as np
import os
from pixel_feature_extractor import *
import pdb
from sklearn.svm import LinearSVC

class ShapeClassifier:
	SHAPE_CODES = dict({'oval': 0, 'squiggle': 1, 'rhombus': 2})

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
				shape_code = ShapeClassifier.SHAPE_CODES[filename.split('_')[2]]
				self.train_X.append(p.get_features(self.train_dir + '/' + filename))
				self.train_Y.append(shape_code)
		for filename in os.listdir(self.test_dir):
			if filename.endswith(".jpg"):
				shape_code = ShapeClassifier.SHAPE_CODES[filename.split('_')[2]]
				self.test_X.append(p.get_features(self.test_dir + '/' + filename))
				self.test_Y.append(shape_code)

	def train(self):
		lin_clf = LinearSVC()
		lin_clf.fit(self.train_X, self.train_Y) 
		predictions = lin_clf.predict(self.test_X)
		print "Predicted:"
		print predictions
		score = lin_clf.score(self.test_X, self.test_Y)
		print "Score: %f" % score


c = ShapeClassifier('../train_images_1', '../test_images_1')
c.process_images()
c.train()