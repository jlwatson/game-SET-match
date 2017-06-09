from sklearn.externals import joblib
from sklearn.metrics import f1_score
from classifiers.pixel_feature_extractor import PixelFeatureExtractor
from classifiers.simple_color_classifier import *
from classifiers.row_pixel_feature_extractor import RowPixelFeatureExtractor
from finder.set_card import SetCard
from detection.card_detector import CardDetector
import finder.set_finder as set_finder
import os
import pdb
import numpy as np


class Pipeline:

  CLASSIFIER_NAMES = ['color', 'quantity', 'shape', 'shade']
  FEATURE_MAPPINGS = ['standard', 'standard', 'standard', 'shade']

  CODES = dict({'oval': 0, 'squiggle': 1, 'rhombus': 2, 'stripe': 0, 'solid': 1, 'hollow': 2, 'one': 0, 'two': 1, 'three': 2, 'green': 0, 'purple': 1, 'red': 2})

  def __init__(self, root_dir="detection/output", testing=False):
    self.classifiers = [joblib.load('pipeline/color_clf.pkl'), joblib.load('pipeline/quantity_clf.pkl'), joblib.load('pipeline/shape_clf.pkl'), joblib.load('pipeline/shade_clf.pkl') ]
    self.root_dir = root_dir
    self.testing = testing

  def classify_cards(self):
    p = PixelFeatureExtractor()
    rp = RowPixelFeatureExtractor()
    Y = []
    features = {}
    color_predictions = []
    for subdir, dirs, files in os.walk(self.root_dir):
      for card_dir in dirs:
        print card_dir
        standard_features = []
        color_features = []
        shade_features = []
        for filename in os.listdir(self.root_dir + '/' + card_dir):
          # print filename
          if filename.endswith(".jpg"):
            if self.testing: 
              trim_name = filename.split('.')[0]
              labels = trim_name.split('_')
              label_vals = [self.CODES[label] for i, label in enumerate(labels)]
              Y.append(label_vals)
            img_filepath = self.root_dir + '/' + card_dir + '/' + filename
            standard_features.append(p.get_features(img_filepath))
            color_predictions.append(get_color(img_filepath))
            shade_features.append(rp.get_features(img_filepath))
    features['shade'] = shade_features
    features['standard'] = standard_features




    if self.testing:
      Y = np.array(Y)

    predictions = {}
    for i in xrange(0, 4):
      clf_name = self.CLASSIFIER_NAMES[i]
      if i == 0:
        cur_predictions = color_predictions
        predictions[clf_name] = color_predictions
      else:
        X = features[self.FEATURE_MAPPINGS[i]]
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

    cards = []
    for i in xrange(len(predictions['color'])):
      color = predictions['color'][i]
      quantity = predictions['quantity'][i]
      shape = predictions['shape'][i]
      shade = predictions['shade'][i]

      cards.append(SetCard(color, quantity, shape, shade))

    print cards
    return cards

  def detect_cards(self, root_dir='better_game_images/'):
    c = CardDetector()
    for filename in os.listdir(root_dir):
      if filename.endswith(".JPG"):
        img_name = filename.split('.')[0]
        label_filename = root_dir + img_name + '_labels.txt'
        if int(os.path.getsize(label_filename)) > 0:
          # labels exist and are complete
          c.getCards(root_dir + filename, 'detection/output/' + img_name, card_name_file=label_filename)
          print "Got cards for " + img_name

# os.system("python detection/corner_detection.py detection/test_input/wb_on_center.jpg detection/output --deterministic --labels detection/test_input/wb_on_center_names.txt")
# c = CardDetector()
# c.getCards('detection/test_input/wb_on_center.jpg', 'detection/output', card_name_file='detection/test_input/wb_on_center_names.txt')
# c.getCards('better_game_images/IMG_0236.jpg', 'detection/output', card_name_file='better_game_images/IMG_0236_labels.txt')



p = Pipeline(testing=True, root_dir='detection/output')
# p.detect_cards()
cards = p.classify_cards()
# print set_finder.find(cards)
