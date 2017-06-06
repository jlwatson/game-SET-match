'''
corner_detection.py
-------------------
Applies the Shi-Tomasi corner detector to extract corners from a given input
image.
'''

import itertools as it
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread, imresize, imsave
from scipy.signal import fftconvolve
import sys
import time
import pdb
from card_extractor import *
import argparse

'''
Note: borrowed from jlwatson's grayscale implementation in PS0
'''
def convert_to_grayscale(color_image):
    return 0.299 * color_image[:,:,0] + \
           0.587 * color_image[:,:,1] + \
           0.114 * color_image[:,:,2]


WINDOW_RADIUS = 3
THRESHOLD = 8e5
SOBEL_THRESH = 0.76

def calc_gradients(image):

    Ix, Iy = np.zeros(image.shape), np.zeros(image.shape)

    # Add padding around the edges for easy convolution
    padded = np.zeros((image.shape[0]+2, image.shape[1]+2))
    padded[1:-1, 1:-1] = image
    padded[0,0], padded[-1,0], padded[-1,-1], padded[0,-1] = \
        image[0,0], image[-1,0], image[-1,-1], image[0,-1]
    padded[1:-1,0], padded[1:-1,-1], padded[0,1:-1], padded[-1,1:-1] = \
        image[:,0], image[:,-1], image[0,:], image[-1,:]

    for r in range(1, padded.shape[0]-1):
        for c in range(1, padded.shape[1]-1):
            Ix[r-1,c-1] = -padded[r-1,c-1] -2*padded[r,c-1] -padded[r+1,c-1] + \
                padded[r-1,c+1] + 2*padded[r,c+1] + padded[r+1,c+1]
            Iy[r-1,c-1] = -padded[r-1,c-1] -2*padded[r-1,c] -padded[r-1,c+1] + \
                padded[r+1,c-1] + 2*padded[r+1,c] + padded[r+1,c+1]

    return Ix, Iy


def shi_tomasi(image):

    # Calculate X and Y derivative convolutions
    Ix, Iy = calc_gradients(image)

    # Apply Sobel Filter for edge detection
    image = np.sqrt(Ix**2 + Iy**2)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # Threshold
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            image[r,c] = 0.0 if image[r,c] < SOBEL_THRESH else 1.0

    ### For each x, y in the image, get matrix M and score using eigenvals ###
    Ix2, Iy2, Ixy = Ix**2, Iy**2, Ix*Iy
    scores = np.zeros(image.shape)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            minY, maxY = y-WINDOW_RADIUS, y+WINDOW_RADIUS+1
            if minY < 0:
                minY = 0
            if maxY > image.shape[0]:
                maxY = image.shape[0]

            minX, maxX = x-WINDOW_RADIUS, x+WINDOW_RADIUS+1
            if minX < 0:
                minX = 0
            if maxX > image.shape[1]:
                maxX = image.shape[1]

            M = np.array([[np.sum(Ix2[minY:maxY, minX:maxX]),
                           np.sum(Ixy[minY:maxY, minX:maxX])],
                          [np.sum(Ixy[minY:maxY, minX:maxX]),
                           np.sum(Iy2[minY:maxY, minX:maxX])],
                         ])

            eigenvals, _ = np.linalg.eig(M)
            score_val = min(eigenvals[0], eigenvals[1])
            scores[y,x] = 0.0 if score_val < THRESHOLD or image[y,x] <= 0.0 else score_val

    return scores, image


MAX_WINDOW_R = 3
def detect_max(scores):
    top_scores = reversed([np.unravel_index(x,scores.shape) for x in np.argsort(scores, axis=None)])

    points = []
    for r, c in top_scores:
        if scores[r,c] < THRESHOLD:
            continue

        points.append((r,c))
        plt.scatter(x=c, y=r, s=10, c='b')
        scores[r-MAX_WINDOW_R:r+MAX_WINDOW_R+1,c-MAX_WINDOW_R:c+MAX_WINDOW_R+1] = np.zeros((MAX_WINDOW_R*2+1, MAX_WINDOW_R*2+1))

    print len(points)
    return points


def points_before(points, i, axis):
    count = 0
    for p in points:
        if p[axis] < i:
            count += 1

    return count


# points are in (r,c) pairs
def create_segments(image, points):

    points_by_row = sorted(points, key=lambda x: x[0])
    points_by_col = sorted(points, key=lambda x: x[1])
    minr, maxr = points_by_row[0][0], points_by_row[-1][0]
    minc, maxc = points_by_col[0][1], points_by_col[-1][1]

    row_boundaries, col_boundaries = [], [] 

    goal = 0
    temp = []
    for r in range(minr-20, maxr+20):
        before = points_before(points, r, 0)
        if before == goal:
            temp.append(r)
        elif temp:
            row_boundaries.append(int(np.average(temp)))
            goal += 16
            temp = []
    row_boundaries.append(int(maxr))

    goal = 0
    temp = []
    for c in range(minc-20, maxc+20):
        before = points_before(points, c, 1)
        if before == goal:
            temp.append(c)
        elif temp:
            col_boundaries.append(int(np.average(temp)))
            goal += 12
            temp = []
    col_boundaries.append(int(maxc))

    for rb in row_boundaries:
        plt.plot([0, image.shape[1]], [rb, rb])

    for cb in col_boundaries:
        plt.plot([cb, cb], [0, image.shape[0]])

    return row_boundaries, col_boundaries

def temp_card_assignments(row_boundaries, col_boundaries, points):
    row_centers = [float(row_boundaries[i] + row_boundaries[i+1]) / 2 for i in range(len(row_boundaries) - 1)]
    col_centers = [float(col_boundaries[i] + col_boundaries[i+1]) / 2 for i in range(len(col_boundaries) - 1)]
    centroids = []
    for rc in row_centers:
        for cc in col_centers:
            centroids.append((rc, cc))
    card_points = [[] for i in xrange(len(centroids))]
    points = np.array(points)
    centroids = np.array(centroids)
    for p in points:
        card = min([(i, np.linalg.norm(p - c)) for i, c in enumerate(centroids)], key=lambda x:x[1])[0]
        card_points[card].append(p)
    return np.array(card_points)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image_name')
    parser.add_argument('output_dir')
    parser.add_argument('--labels', default=None)
    args = parser.parse_args()

    if not ".jpg" in args.image_name:
        print "Error: expecting JPG image input"
        exit(-1)

    input_image = imresize(imread(args.image_name), 0.15)
    # print "Input image shape"
    # print input_image.shape
    grayscale = convert_to_grayscale(input_image)

    scores, sobel_image = shi_tomasi(grayscale)

    plt.imshow(input_image)
    points = detect_max(scores)
    r_boundaries, c_boundaries = create_segments(sobel_image, points)
    card_clusters = temp_card_assignments(r_boundaries, c_boundaries, points)
    # plt.show()

    extract_cards(input_image, card_clusters, args.output_dir, args.labels, 0)
    # Segment into groups of 4
    # Create a method that given four-point tuples returns rectified card images
    # for i in range(len(r_boundaries)-1):
    #     for j in range(len(c_boundaries)-1):
    #         segment = input_image[r_boundaries[i]:r_boundaries[i+1], c_boundaries[j]:c_boundaries[j+1]]
    #         print "segment shape"
    #         print segment.shape
    #         imsave(sys.argv[2]+"/image"+str(i)+str(j)+".jpg", segment)

    

