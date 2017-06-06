from skimage import io, color, img_as_float

import matplotlib.pyplot as plt
import numpy as np
import pdb
import math

COLOR_CODES = dict({'green': 0, 'purple': 1, 'red': 2})

# Could use filtered_pixels as features for classifier
def get_color(img_filename):
  image = img_as_float(io.imread(img_filename))
  pixels = image.reshape(image.shape[0] * image.shape[1], 3)
  filtered_pixels = np.array([p for p in pixels if 0.4 <= np.linalg.norm(p) <= 1.0])
  mean_color = np.mean(filtered_pixels, axis=0)
  distance_red = np.linalg.norm(mean_color - (1, 0, 0))
  distance_green = np.linalg.norm(mean_color - (0, 1, 0))
  distance_purple = np.linalg.norm(mean_color - (102./255, 0, 204./255))
  distance_blue = np.linalg.norm(mean_color - (0, 0, 1))
  distances = [distance_green, distance_purple, distance_red]
  print distances

  width = int(math.sqrt(filtered_pixels.shape[0]))


  test_image = filtered_pixels[:width**2].reshape(width, width, 3) 

  # plt.imshow(test_image)
  # plt.show()
  # pdb.set_trace()

  dist, clr = min((clr, idx) for (idx, clr) in enumerate(distances))
  return clr
  
