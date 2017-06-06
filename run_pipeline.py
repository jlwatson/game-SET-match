from sklearn.externals import joblib
from sklearn.metrics import f1_score
from classifiers.pixel_feature_extractor import PixelFeatureExtractor
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


    predictions = {}
    for i in xrange(4):
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


    return predictions

os.system("python detection/corner_detection.py detection/test_input/on_center.jpg detection/output --labels detection/test_input/on_center_names.txt")
p = Pipeline(testing=True, card_dir='detection/output')
p.classify_cards()