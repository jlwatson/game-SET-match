'''
corner_detection.py
-------------------
Applies the Shi-Tomasi corner detector to extract corners from a given input
image.
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread, imresize
from scipy.signal import fftconvolve
import sys

'''
Note: borrowed from jlwatson's grayscale implementation in PS0
'''
def convert_to_grayscale(color_image):
    grayscale_image = np.empty((color_image.shape[0], color_image.shape[1]))
    for r in range(grayscale_image.shape[0]):
        for c in range(grayscale_image.shape[1]):
            grayscale_image[r, c] = 0.299 * color_image[r,c][0] + \
                                    0.587 * color_image[r,c][1] + \
                                    0.114 * color_image[r,c][2]
    return grayscale_image


WINDOW_RADIUS = 6

def shi_tomasi(image):

    ### Calculate X and Y derivative convolutions ###

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

    # Apply Sobel Filter for edge detection
    image = np.sqrt(Ix**2 + Iy**2)

    ### For each x, y in the image, get matrix M and score using eigenvals ###
    scores = np.zeros(image.shape)
    for y0 in range(image.shape[0]):
        for x0 in range(image.shape[1]):
            M = np.zeros((2,2))
            for y in range(y0-WINDOW_RADIUS,y0+WINDOW_RADIUS+1):
                for x in range(x0-WINDOW_RADIUS,x0+WINDOW_RADIUS+1):
                    if y>=image.shape[0] or y<0 or x>=image.shape[1] or x<0:
                        continue # pixel not in image is not in window
                    M += np.array([[Ix[y,x]**2, Ix[y,x]*Iy[y,x]],
                                   [Ix[y,x]*Iy[y,x], Iy[y,x]**2]])
            eigenvals, _ = np.linalg.eig(M)
            scores[y0,x0] = min(eigenvals[0], eigenvals[1])

    plt.imshow(scores, cmap='gray')
    plt.show()

    '''
    # Sobel filter and FFT convolution that mimics the above loop
    Ix_fft = fftconvolve(image, np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]]))
    Iy_fft = fftconvolve(image, np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]]))
    G2 = np.sqrt(Ix_fft**2 + Iy_fft**2)

    plt.imshow(G, cmap='gray')
    plt.imshow(G2, cmap='gray')
    plt.show()
    '''


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Usage: python corner_detection.py <input_image.jpg>"
        exit(-1)

    image_name = sys.argv[1]
    if not ".jpg" in image_name:
        print "Error: expecting JPG image input"
        exit(-1)

    input_image = imresize(imread(image_name), 0.3)
    grayscale = convert_to_grayscale(input_image)

    shi_tomasi(grayscale)

