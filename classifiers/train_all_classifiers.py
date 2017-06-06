from color_classifier import ColorClassifier
from shape_classifier import ShapeClassifier
from shade_classifier import ShadeClassifier
from quantity_classifier import QuantityClassifier


def train_all(train_dir, test_dir, base_dir):
  print "Training color classifier..."
  c = ColorClassifier(train_dir, test_dir, base_dir, test=False)
  c.process_images()
  c.train()
  print "Color classifier trained"

  print "Training shape classifier..."
  c = ShapeClassifier(train_dir, test_dir, base_dir)
  c.process_images()
  c.train()
  print "Shape classifier trained"

  print "Training shade classifier..."
  c = ShadeClassifier(train_dir, test_dir, base_dir)
  c.process_images()
  c.train()
  print "Shade classifier trained"

  print "Training quantity classifier..."
  c = QuantityClassifier(train_dir, test_dir, base_dir)
  c.process_images()
  c.train()
  print "Quantity classifier trained"


if __name__ == '__main__':
  train_all('../train_images_1', '../test_images_1', '../set_images')
