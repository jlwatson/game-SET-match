from color_classifier import ColorClassifier
from shape_classifier import ShapeClassifier
from shade_classifier import ShadeClassifier
from quantity_classifier import QuantityClassifier


def train_all(train_dir, test_dir, base_dir):
	print "Training color classifier..."
	c = ColorClassifier(train_dir, test_dir, base_dir, test=False)
	c.process_images()
	color_clf = c.train()
	c.test()
	print "Color classifier trained"

	print "Training shape classifier..."
	c = ShapeClassifier(train_dir, test_dir, base_dir)
	c.process_images()
	shape_clf = c.train()
	c.test()
	print "Shape classifier trained"

	print "Training shade classifier..."
	c = ShadeClassifier(train_dir, test_dir, base_dir)
	c.process_images()
	shade_clf = c.train()
	c.test()
	print "Shade classifier trained"

	print "Training quantity classifier..."
	c = QuantityClassifier(train_dir, test_dir, base_dir)
	c.process_images()
	quantity_clf = c.train()
	c.test()
	print "Quantity classifier trained"


if __name__ == '__main__':
	train_all('../train_images_1', '../test_images_1', '../set_images')