class SetCard(object):
	QUANTITY = {0: 1, 1: 2, 2: 3}
	COLOR = {0: 'Green', 1: 'Purple', 2: 'Red'}
	SHAPE = {0: 'Oval', 1: 'Squiggle', 2: 'Rhombus'}
	SHADE = {0: 'Stripe', 1: 'Solid', 2: 'Hollow'}

	color = 0
	quantity = 0
	shape = 0
	shading = 0

	def __init__(self, color, quantity, shape, shading):
		self.color = color
		self.quantity = quantity
		self.shape = shape
		self.shading = shading

	def is_set(self, card_1, card_2):
		quantity_same = (card_1.quantity == card_2.quantity)
		color_same = (card_1.color == card_2.color)
		shading_same = (card_1.shading == card_2.shading)
		shape_same = (card_1.shape == card_2.shape)

		if quantity_same:
			if self.quantity != card_1.quantity:
				return False
		else:
			if self.quantity == card_1.quantity or self.quantity == card_2.quantity:
				return False

		if color_same:
			if self.color != card_1.color:
				return False
		else:
			if self.color == card_1.color or self.color == card_2.color:
				return False

		if shading_same:
			if self.shading != card_1.shading:
				return False
		else:
			if self.shading == card_1.shading or self.shading == card_2.shading:
				return False

		if shape_same:
			if self.shape != card_1.shape:
				return False
		else:
			if self.shape == card_1.shape or self.shape == card_2.shape:
				return False

		return True


	def __str__(self):
		return 'Card: %d %s %s %s' % (
			SetCard.QUANTITY[self.quantity],
			SetCard.COLOR[self.color],
			SetCard.SHADE[self.shading],
			SetCard.SHAPE[self.shape]
		)

	def __repr__(self):
		return 'Card: %d %s %s %s' % (
			SetCard.QUANTITY[self.quantity],
			SetCard.COLOR[self.color],
			SetCard.SHADE[self.shading],
			SetCard.SHAPE[self.shape]
		)