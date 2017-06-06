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
from scipy.misc import imread, imresize, imsave, imrotate
from scipy.signal import fftconvolve
from scipy.spatial.distance import cdist
import sys
import time
import pdb
from card_extractor import *

'''
Note: borrowed from jlwatson's grayscale implementation in PS0
'''
def convert_to_grayscale(color_image):
    return 0.299 * color_image[:,:,0] + \
           0.587 * color_image[:,:,1] + \
           0.114 * color_image[:,:,2]


WINDOW_RADIUS = 3
THRESHOLD = 1e6
SOBEL_THRESH = 0.75

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


def sobel_filter(image):

    # Calculate X and Y derivative convolutions
    Ix, Iy = calc_gradients(image)

    # Apply Sobel Filter for edge detection
    image = np.sqrt(Ix**2 + Iy**2)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # Threshold
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            image[r,c] = 0.0 if image[r,c] < SOBEL_THRESH else 1.0

    '''
    plt.imshow(image, cmap='gray');
    plt.show();
    '''
    return image, Ix, Iy


def shi_tomasi(image):

    image, Ix, Iy = sobel_filter(image)
    
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

    '''
    plt.imshow(scores, cmap='gray')
    plt.show()
    '''
    return scores, image


MAX_WINDOW_R = 3
def detect_max(scores):
    top_scores = reversed([np.unravel_index(x,scores.shape) for x in np.argsort(scores, axis=None)])

    points = []
    for r, c in top_scores:
        if scores[r,c] < THRESHOLD:
            continue

        points.append((r,c))
        plt.scatter(x=c, y=r, s=25, c='w')
        row_min, col_min = max(r-MAX_WINDOW_R, 0), max(c-MAX_WINDOW_R, 0)
        row_max, col_max = min(scores.shape[0], r+MAX_WINDOW_R+1), min(scores.shape[1], c+MAX_WINDOW_R+1)

        scores[row_min:row_max,col_min:col_max] = np.zeros((row_max-row_min, col_max-col_min))

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
        plt.plot([0, image.shape[1]], [rb, rb], 'b')

    for cb in col_boundaries:
        plt.plot([cb, cb], [0, image.shape[0]], 'b')

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

def kmeans(features, num_clusters, initial_centroids):

    if initial_centroids is not None:
        centroids = initial_centroids
    else:
        centroids = features[np.random.choice(features.shape[0], size=num_clusters, replace=False)]

    while True:
        distances = cdist(centroids, features)
        closest_centroids = np.argmin(distances, axis=0)

        old_centroids = np.array(centroids)
        for c in range(num_clusters):
            assigned_features = features[np.where(closest_centroids == c)[0]]
            centroids[c] = np.average(assigned_features, axis=0)

        if np.allclose(old_centroids, centroids):
            return closest_centroids, centroids


# returns 1-indexed cluster numbers
def clusters_near_point(cluster_im, pt):
    RAD = 3

    window = cluster_im[pt[0]-RAD:pt[0]+RAD+1, pt[1]-RAD:pt[1]+RAD+1]
    cluster1, cluster2 = list(np.trim_zeros(np.unique(window))[:2])
    return cluster1, cluster2


# distances = distances from all unchosen points to pt index
# angles = angles to point (arctan(pt - all other unchosen points))
def find_closest_pair(cluster_im, centroid_angles, pt, unchosen_indexes, cluster_map, distances, angles, sobel_Ix, sobel_Iy):
    CLUSTER_RAD = min(pt[0], pt[1], cluster_im.shape[0] - pt[0], cluster_im.shape[1] - pt[1], 75)
    RADIAN_THRESH = 0.8

    cluster1, cluster2 = clusters_near_point(cluster_im, pt)
    valid_indices = cluster_map[cluster1].union(cluster_map[cluster2])

    cluster1_angle, cluster2_angle = centroid_angles[cluster1-1], centroid_angles[cluster2-1]
    # print "cluster1, angle:", cluster1, math.degrees(cluster1_angle)
    # print "cluster2, angle:", cluster2, math.degrees(cluster2_angle)

    x, y = np.where(cluster_im[pt[0]-CLUSTER_RAD:pt[0]+CLUSTER_RAD+1, pt[1]-CLUSTER_RAD:pt[1]+CLUSTER_RAD+1] == cluster1)
    x, y = x + pt[0] - CLUSTER_RAD, y + pt[1] - CLUSTER_RAD
    local_angle_1 = np.arctan2(np.mean(sobel_Iy[x,y]), np.mean(sobel_Ix[x,y]))
    # print "local angle cluster 1:", math.degrees(local_angle_1)

    x, y = np.where(cluster_im[pt[0]-CLUSTER_RAD:pt[0]+CLUSTER_RAD+1, pt[1]-CLUSTER_RAD:pt[1]+CLUSTER_RAD+1] == cluster2)
    x, y = x + pt[0] - CLUSTER_RAD, y + pt[1] - CLUSTER_RAD
    local_angle_2 = np.arctan2(np.mean(sobel_Iy[x,y]), np.mean(sobel_Ix[x,y]))
    # print "local angle cluster 2:", math.degrees(local_angle_2)

    angle_diffs1 = local_angle_1 - angles
    angle_diffs1[np.where(angle_diffs1 > math.radians(180.0))[0]] -= math.radians(360)
    angle_diffs1[np.where(angle_diffs1 < math.radians(-180.0))[0]] += math.radians(360)
    angle_diffs1 = np.abs(angle_diffs1)
    for r in range(angle_diffs1.shape[0]):
        if unchosen_indexes[r] not in valid_indices:
            angle_diffs1[r] = math.radians(180.0)

    matching_angles = np.where(angle_diffs1 <= RADIAN_THRESH)[0]
    min_dist_pt = matching_angles[distances[matching_angles].argmin()]

    angle_diffs2 = np.abs(local_angle_2 - angles)
    angle_diffs2[np.where(angle_diffs2 > math.radians(180.0))[0]] -= math.radians(360)
    angle_diffs2[np.where(angle_diffs2 < math.radians(-180.0))[0]] += math.radians(360)
    angle_diffs2 = np.abs(angle_diffs2)
    for r in range(angle_diffs2.shape[0]):
        if unchosen_indexes[r] not in valid_indices:
            angle_diffs2[r] = math.radians(180.0)

    matching_angles = np.where(angle_diffs2 <= RADIAN_THRESH)[0]
    min_dist_pt2 = matching_angles[distances[matching_angles].argmin()]

    return (min_dist_pt, min_dist_pt2)


def group_points(points, cluster_im, cluster_centroids, sobel_Ix, sobel_Iy, out_im):
    # plt.imshow(out_im)
    points = np.array(points)
    if points.shape[0] % 4 != 0:
        print "WARNING: number of corners detected is not a multiple of 4"

    unchosen = np.ones((points.shape[0],1), dtype=bool)
    distances = cdist(points, points)
    centroid_angles = np.arctan2(cluster_centroids[:,1], cluster_centroids[:,0])

    # assuming four card sides
    cluster_map = {1: set(), 2: set(), 3: set(), 4: set()}
    for r in range(points.shape[0]):
        c1, c2 = clusters_near_point(cluster_im, points[r])
        cluster_map[c1].add(r)
        cluster_map[c2].add(r)

    groups = []
    while unchosen.any():
        pt_index = np.random.choice(np.where(unchosen)[0], 1) 
        # print "chosen point:", points[pt_index][0]
        pt = points[pt_index][0]
        unchosen[pt_index] = False

        unchosen_pt_indexes = np.where(unchosen)[0]
        angles = np.arctan2((points[unchosen_pt_indexes] - pt)[:,0], (points[unchosen_pt_indexes] - pt)[:,1])
        # print "unchosen points:", points[unchosen_pt_indexes]
        # print "angles:", np.degrees(angles)
        # print "distances:", distances[unchosen_pt_indexes, pt_index]
        pt1_idx, pt2_idx = find_closest_pair(cluster_im, centroid_angles, pt, unchosen_pt_indexes, 
            cluster_map, distances[unchosen_pt_indexes, pt_index], angles, sobel_Ix, sobel_Iy)
        # pt1_u_idx = pt1_idx
        # pt2_u_idx = pt2_idx
        pt1_idx = np.array([unchosen_pt_indexes[pt1_idx]])
        pt2_idx = np.array([unchosen_pt_indexes[pt2_idx]])

        # print "matching point 1:", pt1_u_idx, points[pt1_idx]
        # print "matching point 2:", pt2_u_idx, points[pt2_idx]
        unchosen[pt_index] = True
        unchosen[pt1_idx] = False

        pt2 = points[pt1_idx][0]

        unchosen_pt_indexes = np.where(unchosen)[0]
        angles = np.arctan2((points[unchosen_pt_indexes] - pt2)[:,0], (points[unchosen_pt_indexes] - pt2)[:,1])

        pt1_idx_2, pt2_idx_2 = find_closest_pair(cluster_im, centroid_angles, pt2, unchosen_pt_indexes,
            cluster_map, distances[unchosen_pt_indexes, pt1_idx], angles, sobel_Ix, sobel_Iy)
        pt1_idx_2 = np.array([unchosen_pt_indexes[pt1_idx_2]])
        pt2_idx_2 = np.array([unchosen_pt_indexes[pt2_idx_2]])

        '''
        plt.scatter(x=points[pt_index,1], y=points[pt_index,0], s=25, c='r')
        plt.scatter(x=points[pt1_idx,1], y=points[pt1_idx,0], s=25, c='r')
        plt.scatter(x=points[pt2_idx,1], y=points[pt2_idx,0], s=25, c='r')
        '''
        if (points[pt1_idx_2][0] == pt).all():
            groups.append([pt, pt2, points[pt2_idx][0], points[pt2_idx_2][0]])
            # plt.scatter(x=points[pt2_idx_2,1], y=points[pt2_idx_2,0], s=25, c='r')
            unchosen[pt2_idx_2] = False
        elif (points[pt2_idx_2][0] == pt).all():
            groups.append([pt, pt2, points[pt2_idx][0], points[pt1_idx_2][0]])
            # plt.scatter(x=points[pt1_idx_2,1], y=points[pt1_idx_2,0], s=25, c='r')
            unchosen[pt1_idx_2] = False
        else:
            print "ERROR: couldn't make group of 4, exiting."
            exit(-1)


        unchosen[pt_index] = False
        unchosen[pt2_idx] = False

    # plt.show()
    return groups, len(groups)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage: python corner_detection.py <input_image.jpg> <output dir>"
        exit(-1)

    image_name = sys.argv[1]
    if not ".jpg" in image_name:
        print "Error: expecting JPG image input"
        exit(-1)

    input_image = imresize(imread(image_name), 0.15)
    grayscale = convert_to_grayscale(input_image)

    sobel_image, Ix, Iy = sobel_filter(grayscale)

    np.set_printoptions(precision=3, suppress=True, linewidth=1000)

    features = np.vstack(sobel_image.nonzero()).T
    # Features for potential X/Y gradient clustering
    actual_features = np.hstack(
        (
            np.array([Ix[features[:,0], features[:,1]]]).T,
            np.array([Iy[features[:,0], features[:,1]]]).T,
        )
    )
    initial_centroids = np.array([
        [np.min(Ix),0.0],
        [np.max(Ix),0.0],
        [0.0,np.min(Iy)],
        [0.0,np.max(Iy)],
    ])

    centroid_assgs, centroids = kmeans(actual_features, 4, initial_centroids)
    centroid_assgs = list(centroid_assgs)
    
    colors = [
        np.array([239,45,45]),  # Turquoise
        np.array([239,45,194]), # Bright Green
        np.array([207,239,45]), # Purplish Blue
        np.array([45,239,191]), # Reddish Pink
    ]

    clustered_image = np.zeros((sobel_image.shape[0], sobel_image.shape[1]), dtype=int)
    for r in range(features.shape[0]):
        clustered_image[int(features[r,0]), int(features[r,1])] = centroid_assgs.pop(0) + 1

    output_image = np.zeros((clustered_image.shape[0], clustered_image.shape[1], 3))
    for r in range(clustered_image.shape[0]):
        for c in range(clustered_image.shape[1]):
            if clustered_image[r,c] == 0: continue
            output_image[r,c] = colors[clustered_image[r,c] - 1]

    scores, sobel_image = shi_tomasi(grayscale)

    points = detect_max(scores)
    grouped_pts, num_cards = group_points(points, clustered_image, centroids, Ix, Iy, output_image)
    card_clusters = np.concatenate(grouped_pts).reshape((num_cards, 4, 2))
    extract_cards(input_image, card_clusters, sys.argv[2])

