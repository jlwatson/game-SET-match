from sklearn.externals import joblib
from sklearn.metrics import f1_score
from classifiers.pixel_feature_extractor import PixelFeatureExtractor
from classifiers.simple_color_classifier import *
from finder.set_card import SetCard
import finder.set_finder as set_finder
import os
import pdb
import numpy as np


class Pipeline:

  CLASSIFIER_NAMES = ['color', 'quantity', 'shape', 'shade']

  CODES = dict({'oval': 0, 'squiggle': 1, 'rhombus': 2, 'stripe': 0, 'solid': 1, 'hollow': 2, 'one': 0, 'two': 1, 'three': 2, 'green': 0, 'purple': 1, 'red': 2})

  def __init__(self, card_dir="detection/output", testing=False):
    self.classifiers = [joblib.load('pipeline/color_clf.pkl'), joblib.load('pipeline/quantity_clf.pkl'), joblib.load('pipeline/shape_clf.pkl'), joblib.load('pipeline/shade_clf.pkl') ]
    self.card_dir = card_dir
    self.testing = testing


  def classify_cards(self):
    p = PixelFeatureExtractor()
    Y = []
    X = []
    for filename in os.listdir(self.card_dir):
      print filename
      if filename.endswith(".jpg"):
        if self.testing: 
          trim_name = filename.split('.')[0]
          labels = trim_name.split('_')
          label_vals = [self.CODES[label] for i, label in enumerate(labels)]
          Y.append(label_vals)
        X.append(p.get_features(self.card_dir + '/' + filename))

    if self.testing:
      Y = np.array(Y)


    # Quantity, shape, shade
    predictions = {}
    for i in xrange(1, 4):
      clf_name = self.CLASSIFIER_NAMES[i]
      clf = self.classifiers[i]

      cur_predictions = clf.predict(X)
      predictions[clf_name] = cur_predictions

      if self.testing:
        f1 = f1_score(Y[:, i], cur_predictions, labels=[0, 1, 2], average='micro')
        print clf_name
        print "Predicted"
        print cur_predictions
        print "Actual"
        print Y[:, i]
        print "F1 Score for %s: %f" % (clf_name, f1)

    # Color
    clf_name = self.CLASSIFIER_NAMES[0]
    cur_predictions = []
    for filename in os.listdir(self.card_dir):
      if filename.endswith(".jpg"):
        cur_predictions.append(get_color(self.card_dir + '/' + filename))

    # for i in xrange(1, 4):
    #   clf_name = self.CLASSIFIER_NAMES[i]
    #   clf = self.classifiers[i]

    #   add_features = predictions[clf_name]

    #   X = np.concatenate((X, np.array(add_features).reshape((12, 1))), axis=1)

    # clf_name = self.CLASSIFIER_NAMES[0]
    # clf = self.classifiers[0]
    # cur_predictions = clf.predict(X)
    predictions[clf_name] = cur_predictions
    
    if self.testing:
        f1 = f1_score(Y[:, 0], cur_predictions, labels=[0, 1, 2], average='micro')
        print clf_name
        print "Predicted"
        print cur_predictions
        print "Actual"
        print Y[:, 0]
        print "F1 Score for %s: %f" % (clf_name, f1)

    cards = []
    for i in xrange(len(predictions['color'])):
      color = predictions['color'][i]
      quantity = predictions['quantity'][i]
      shape = predictions['shape'][i]
      shade = predictions['shade'][i]

      cards.append(SetCard(color, quantity, shape, shade))

    print cards
    return cards

os.system("python detection/corner_detection.py detection/test_input/wb_on_center.jpg detection/output --deterministic --labels detection/test_input/wb_on_center_names.txt")
p = Pipeline(testing=True, card_dir='detection/output')
cards = p.classify_cards()
print set_finder.find(cards)
