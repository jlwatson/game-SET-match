import numpy as np
import os
from pixel_feature_extractor import *
import pdb
from sklearn.svm import LinearSVC
from feature_classifier import *

class ColorClassifier(FeatureClassifier):
  COLOR_CODES = dict({'green': 0, 'purple': 1, 'red': 2})
  CODES = dict({'oval': 0, 'squiggle': 1, 'rhombus': 2, 'stripe': 0, 'solid': 1, 'hollow': 2, 'one': 0, 'two': 1, 'three': 2, 'green': 0, 'purple': 1, 'red': 2})

  def __init__(self, train_dir, test_dir, base_dir, test=True):
    self.test = test
    super(ColorClassifier, self).__init__(train_dir, test_dir, base_dir, ColorClassifier.COLOR_CODES, 0, 'color')


  def process_images(self):
    if self.test:
      super(ColorClassifier, self).process_images()
    else:
      p = PixelFeatureExtractor()
      for filename in os.listdir(self.train_dir):
        if filename.endswith(".jpg"):
          trim_name = filename.split('.')[0]
          quant_shape_shade = [ColorClassifier.CODES[trim_name.split('_')[i]] for i in xrange(1, 4)]
          color_code = self.codes[trim_name.split('_')[self.feature_index]]
          self.train_X.append(np.append(p.get_color_features(self.train_dir + '/' + filename), np.array(quant_shape_shade)))
          self.train_Y.append(color_code)
     

  def process_images_split_dir(self, train_names, test_names):
    if self.test:
      super(ColorClassifier, self).process_images_split_dir(train_names, test_names)
    else:
      p = PixelFeatureExtractor()
      for filename in train_names:
        trim_name = filename.split('.')[0]
        quant_shape_shade = [ColorClassifier.CODES[trim_name.split('_')[i]] for i in xrange(1, 4)]
        color_code = self.codes[trim_name.split('_')[self.feature_index]]
        self.train_X.append(np.append(p.get_color_features(self.train_dir + '/' + filename), np.array(quant_shape_shade)))
        self.train_Y.append(color_code)

  def train(self):
    if self.test:
      return super(ColorClassifier, self).train()
    else:
      lin_clf = LinearSVC()
      lin_clf.fit(self.train_X, self.train_Y) 
      
      joblib.dump(lin_clf, '../pipeline/' + self.type + '_clf.pkl') 
      return (0, 0)

  def reset(self):
    self.train_X = []
    self.train_Y = []
    self.test_X = []
    self.test_Y = []



if __name__ == '__main__':
	c = ColorClassifier('../train_images_1', '../test_images_1', '../set_images', test=False)
	c.process_images()
	c.train()