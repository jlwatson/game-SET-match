import numpy as np
import os
from pixel_feature_extractor import *
import pdb
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

class FeatureClassifier(object):
	CLASSIFIER_NAMES = ['color', 'quantity', 'shape', 'shade']

	CODES = dict({
		'shape': {'oval': 0, 'squiggle': 1, 'rhombus': 2}, 
		'shade': {'stripe': 0, 'solid': 1, 'hollow': 2}, 
		'quantity': {'one': 0, 'two': 1, 'three': 2}, 
		'color': {'green': 0, 'purple': 1, 'red': 2}
	})

	INDICES = dict({
		'color': 0,
		'quantity': 1,
		'shape': 2,
		'shade': 3
	})


	def __init__(self, train_dir, test_dir, base_dir, classifier_name, 
		pre_feature_names=None, pre_features=None):

		self.train_dir = train_dir
		self.test_dir = test_dir
		self.base_dir = base_dir
		self.train_X = []
		self.train_Y = []
		self.test_X = []
		self.test_Y = []
		self.classifier_name = classifier_name
		self.feature_index = FeatureClassifier.INDICES[classifier_name]
		self.codes = FeatureClassifier.CODES[classifier_name]
		self.pre_feature_names = pre_feature_names
		self.pre_features = pre_features
		self.clf = None


	def process_images(self):
		p = PixelFeatureExtractor()
		for filename in os.listdir(self.train_dir):
			if filename.endswith(".jpg"):
				trim_name = filename.split('.')[0]
				trim_name_split = trim_name.split('_')
				gold_code = self.codes[trim_name_split[self.feature_index]]
				to_append = p.get_features(self.train_dir + '/' + filename)
				if self.pre_feature_names is not None:
					additional_features = [FeatureClassifier.CODES[pfn][trim_name_split[FeatureClassifier.INDICES[[pfn]]]] for pfn in pre_feature_names]
					to_append = np.append(to_append, np.array(additional_features))
				self.train_X.append(to_append)
				self.train_Y.append(gold_code)

		for filename in os.listdir(self.test_dir):
			if filename.endswith(".jpg"):
				trim_name = filename.split('.')[0]
				trim_name_split = trim_name.split('_')
				gold_code = self.codes[trim_name_split[self.feature_index]]
				to_append = p.get_features(self.test_dir + '/' + filename)
				if self.pre_feature_names is not None:
					additional_features = [self.pre_features[pfn] for pfn in self.pre_feature_names]
					to_append = np.append(to_append, np.array(additional_features))
				self.test_X.append(to_append)
				self.test_Y.append(gold_code)




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
		self.clf = LinearSVC()
		self.clf.fit(self.train_X, self.train_Y) 

		return self.clf
		

	def test(self):
		predictions = self.clf.predict(self.test_X)
		print "Predicted:"
		print predictions
		print "Actual:"
		print self.test_Y
		score = self.clf.score(self.test_X, self.test_Y)
		print "Score: %f" % score
		f1 = f1_score(self.test_Y, predictions, labels=[0, 1, 2], average='micro')
		print "F1 Score: %f" % f1
		label_f1 = f1_score(self.test_Y, predictions, labels=[0, 1, 2], average=None)
		print str(label_f1)
		return (score, f1, predictions)

	def reset(self):
		self.train_X = []
		self.train_Y = []
		self.test_X = []
		self.test_Y = []
