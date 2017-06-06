import numpy as np
from scipy import misc
from shade_classifier import *
from color_classifier import *
from shape_classifier import *
from quantity_classifier import *
import pdb
import math
import random
import os
import sys

NUM_FOLDS = 9

class CrossFoldValidation:

	def __init__(self):
		return

	def get_folds(self, dataset, num_folds):
		''' Returns folds in the format [[[train], [test]],...]'''
		folds = []
		fold_size = len(dataset) / num_folds
		for i in xrange(num_folds):
			test = dataset[i * fold_size : (i + 1) * fold_size]
			train = dataset[0 : i * fold_size] + dataset[(i + 1) * fold_size : ]
			fold = [train, test]
			folds.append(fold)
		return folds

if __name__ == '__main__':
	args = sys.argv[1:]
	classifier_type = 'shade'
	if len(args) > 0:
		classifier_type = args[0]

	dataset = []
	for filename in os.listdir('../set_images'):
		if filename.endswith('jpg'):
			dataset.append(filename)

	random.seed(10)
	random.shuffle(dataset)



	cfv = CrossFoldValidation()
	folds = cfv.get_folds(dataset, NUM_FOLDS)

	c = ShadeClassifier('../train_images_1', '../test_images_1', '../set_images')
	if classifier_type == 'color':
		img_features = {}
		other_classifiers = [
			ShapeClassifier('../train_images_1', '../test_images_1', '../set_images'),
			ShadeClassifier('../train_images_1', '../test_images_1', '../set_images'),
			QuantityClassifier('../train_images_1', '../test_images_1', '../set_images')
		]
		for fold in folds:
			extracted_features = []
			for oc in other_classifiers:
				oc.process_images_split_dir(fold[0], fold[1])
				extracted_features.append(oc.train()[2])
				oc.reset()
			for i, img in enumerate(fold[1]):
				img_features[img] = [feature_set[i] for feature_set in extracted_features]
		print img_features


		c = ColorClassifier('../train_images_1', '../test_images_1', '../set_images')
	elif classifier_type == 'shape':
		c = ShapeClassifier('../train_images_1', '../test_images_1', '../set_images')
	elif classifier_type == 'quantity':
		c = QuantityClassifier('../train_images_1', '../test_images_1', '../set_images')

	accuracy_sum = 0
	f1_sum = 0

	for fold in folds:
		c.process_images_split_dir(fold[0], fold[1])
		train_res = c.train()
		accuracy_sum += train_res[0]
		f1_sum += train_res[1]
		c.reset()

	print 'Accuracy for %s over %d folds: %f' % (classifier_type, NUM_FOLDS, accuracy_sum / NUM_FOLDS)
	print 'F1 Score for %s over %d folds: %f' % (classifier_type, NUM_FOLDS, f1_sum / NUM_FOLDS)

	# c = ShadeClassifier('../train_images_1', '../test_images_1', '../set_images')
	# c.process_images()
	# c.train()

