import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread, imresize, imsave
import cv2
import pdb
import string

# IMG_WIDTH = 85
# IMG_HEIGHT = 132

def extract_cards(img, card_clusters, dirname, ratio, card_name_file=None, crop_radius=0):

    card_names = ['image' + str(i) + '.jpg' for i in xrange(12)]
    if card_name_file is not None:
        fo = open(card_name_file, "r")
        card_names = [string.strip(name) for name in fo.readlines()]
    for img_index, corner_orig in enumerate(card_clusters):
        corner_orig = corner_orig * ratio
        # Sort card corners in tl, bl, tr, br order
        ind = corner_orig.argsort(axis=0)[:,0]
        corner_orig = corner_orig[ind]
        for i in xrange(2):
            curInd = 2*i
            if corner_orig[curInd, 1] > corner_orig[curInd + 1, 1]:
                temp = corner_orig[curInd, 1]
                corner_orig[curInd, 1] = corner_orig[curInd + 1, 1]
                corner_orig[curInd + 1, 1] = temp
        # corner_transformed = np.float32([(0, 0), (0, IMG_WIDTH - 1), (IMG_HEIGHT - 1, 0), (IMG_HEIGHT - 1, IMG_WIDTH - 1)])
        minR, minC = np.amin(corner_orig, 0)
        maxR, maxC = np.amax(corner_orig, 0)
        card_img = img[minR-10:maxR + 10, minC-10:maxC + 10]
        card_corners = np.float32(corner_orig - (minR, minC) + (10,10))
        # pdb.set_trace()

        # now that we have our rectangle of points, let's compute
        # the width of our new image
        (tl, bl, tr, br) = card_corners
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
         
        # ...and now for the height of our new image
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

        # take the maximum of the width and height values to reach
        # our final dimensions
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))

        # construct our destination points which will be used to
        # map the screen to a top-down, "birds eye" view
        dest_corners = np.array([
            [0, 0],
            [0, maxHeight - 1],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1]], dtype = "float32")

        M = cv2.getPerspectiveTransform(card_corners, dest_corners)

        dst = cv2.warpPerspective(card_img, M ,(300, 300))
        # dst = dst[crop_radius:IMG_HEIGHT - crop_radius, crop_radius:IMG_WIDTH - crop_radius ]

        # Uncomment to view transformation
        plt.subplot(121),plt.imshow(card_img),plt.title('Input')
        plt.scatter(card_corners[:, 1], card_corners[:, 0])
        plt.subplot(122),plt.imshow(dst),plt.title('Output')
        plt.show()
        pdb.set_trace()
        imsave(dirname+ '/' + card_names[img_index], dst)

