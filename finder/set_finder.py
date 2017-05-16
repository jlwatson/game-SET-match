from set_card import *
from itertools import combinations

def rec_find(cards, sets_found, partial_set):
	if len(cards) == 0:
		return sets_found

	if len(partial_set) == 2:
		# candidate set stub generated, check other cards for uniquely completing
		for card in cards:
			if card.is_set(partial_set[0], partial_set[1]):
				return rec_find([c for c in cards if c != card], sets_found + [(partial_set + [card])], [])

		return rec_find(cards[1:], sets_found, [])
		
	else:
		# no candidate set stub - recurse on all possible candidate pairs
		cur_set_count = -1
		best = []

		for c in combinations(cards, 2):

			result = rec_find([card for card in cards if card not in c], sets_found, [c[0], c[1]])
			if len(result) > cur_set_count:
				cur_set_count = len(result)
				best = result

		return best



def find(cards):
	result = rec_find(cards, [], [])
	return result

def main():
	colors = [0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2]
	shapes = [1, 0, 1, 1, 0, 1, 2, 2, 1, 2, 1, 1]
	quantities = [0, 2, 1, 0, 2, 2, 1, 1, 1, 2, 2, 1]
	shades = [0, 1, 0, 2, 2, 2, 2, 0, 0, 2, 2, 1]

	cards = []
	for i in xrange(len(colors)):
		cards.append(SetCard(colors[i], quantities[i], shapes[i], shades[i]))

	print find(cards)


if __name__ == '__main__':
	main()