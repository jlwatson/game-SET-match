import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread, imresize, imsave
import cv2
import pdb
import string

IMG_WIDTH = 85
IMG_HEIGHT = 132

def extract_cards(img, card_clusters, dirname, card_name_file=None, crop_radius=0):

    card_names = ['image' + str(i) + '.jpg' for i in xrange(12)]
    if card_name_file is not None:
        fo = open(card_name_file, "r")
        card_names = [string.strip(name) for name in fo.readlines()]
    for img_index, corner_orig in enumerate(card_clusters):
        # Sort card corners in same order as corner_transformed
        ind = corner_orig.argsort(axis=0)[:,0]
        corner_orig = corner_orig[ind]
        for i in xrange(2):
            curInd = 2*i
            if corner_orig[curInd, 1] > corner_orig[curInd + 1, 1]:
                temp = corner_orig[curInd, 1]
                corner_orig[curInd, 1] = corner_orig[curInd + 1, 1]
                corner_orig[curInd + 1, 1] = temp
        corner_transformed = np.float32([(0, 0), (0, IMG_WIDTH - 1), (IMG_HEIGHT - 1, 0), (IMG_HEIGHT - 1, IMG_WIDTH - 1)])
        minR, minC = np.amin(corner_orig, 0)
        maxR, maxC = np.amax(corner_orig, 0)
        card_img = img[minR:maxR + 10, minC:maxC + 10]
        card_corners = np.float32(corner_orig - (minR, minC))

        M = cv2.getPerspectiveTransform(card_corners, corner_transformed)
        dst = cv2.warpPerspective(card_img,M,(IMG_WIDTH, IMG_HEIGHT))
        dst = dst[crop_radius:IMG_HEIGHT - crop_radius, crop_radius:IMG_WIDTH - crop_radius ]
        # plt.subplot(121),plt.imshow(card_img),plt.title('Input')
        # plt.scatter(card_corners[:, 1], card_corners[:, 0])
        # plt.subplot(122),plt.imshow(dst),plt.title('Output')
        # plt.show()
        # card_img_tight = img[minR:maxR, minC:maxC]
        imsave(dirname+ '/' + card_names[img_index], dst)
