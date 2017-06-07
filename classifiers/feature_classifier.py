import numpy as np
import os
from pixel_feature_extractor import *
import pdb
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.externals import joblib

class FeatureClassifier(object):

	def __init__(self, train_dir, test_dir, base_dir, codes, feature_index, featureType='unknown'):
		self.train_dir = train_dir
		self.test_dir = test_dir
		self.base_dir = base_dir
		self.train_X = []
		self.train_Y = []
		self.test_X = []
		self.test_Y = []
		self.codes = codes
		self.feature_index = feature_index
		self.type = featureType


	def process_images(self):
		p = PixelFeatureExtractor()
		for filename in os.listdir(self.train_dir):
			if filename.endswith(".jpg"):
				trim_name = filename.split('.')[0]
				shape_code = self.codes[trim_name.split('_')[self.feature_index]]
				self.train_X.append(p.get_features(self.train_dir + '/' + filename))
				self.train_Y.append(shape_code)
		for filename in os.listdir(self.test_dir):
			if filename.endswith(".jpg"):
				trim_name = filename.split('.')[0]
				shape_code = self.codes[trim_name.split('_')[self.feature_index]]
				self.test_X.append(p.get_features(self.test_dir + '/' + filename))
				self.test_Y.append(shape_code)

	def process_images_split_dir(self, train_names, test_names):
		p = PixelFeatureExtractor()
		for filename in train_names:
			trim_name = filename.split('.')[0]
			shape_code = self.codes[trim_name.split('_')[self.feature_index]]
			self.train_X.append(p.get_features(self.base_dir + '/' + filename))
			self.train_Y.append(shape_code)
		for filename in test_names:
			trim_name = filename.split('.')[0]
			shape_code = self.codes[trim_name.split('_')[self.feature_index]]
			self.test_X.append(p.get_features(self.base_dir + '/' + filename))
			self.test_Y.append(shape_code)

	def train(self):
		lin_clf = LinearSVC()
		lin_clf.fit(self.train_X, self.train_Y) 
		predictions = lin_clf.predict(self.test_X)
		print "Predicted:"
		print predictions
		print "Actual:"
		print self.test_Y
		score = lin_clf.score(self.test_X, self.test_Y)
		print "Score: %f" % score
		f1 = f1_score(self.test_Y, predictions, labels=[0, 1, 2], average='micro')
		print "F1 Score: %f" % f1

		joblib.dump(lin_clf, '../pipeline/' + self.type + '_clf.pkl') 
		return (score, f1)
		# label_f1 = f1_score(self.test_Y, predictions, labels=[0, 1, 2], average=None)
		# print str(label_f1)
		# return (score, f1, predictions)
		

	def reset(self):
		self.train_X = []
		self.train_Y = []
		self.test_X = []
		self.test_Y = []
