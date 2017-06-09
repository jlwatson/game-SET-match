import numpy as np
import cv2
import matplotlib.pyplot as plt
import string
import pdb

class CardDetector:

  def __init__(self, numcards=12):
    self.numcards = numcards
    pass

  def rectify(self, h):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]
     
    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew

  def getCards(self, imfile, output_dir, card_name_file=None):
    card_names = ['image' + str(i) + '.jpg' for i in xrange(12)]
    if card_name_file is not None:
        fo = open(card_name_file, "rU")
        card_names = [string.strip(name) for name in fo.readlines()]
    im = cv2.imread(imfile)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(1,1),1000)
    flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY) 
         
    _, contours, hierarchy = cv2.findContours(thresh,mode=cv2.RETR_TREE,method=cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea,reverse=True)[:self.numcards]  

    for i, card in enumerate(contours):
      peri = cv2.arcLength(card,True)
      approx = self.rectify(cv2.approxPolyDP(card,0.02*peri,True))

      # Keep for drawing contours later

      # box = np.int0(approx)
      # cv2.drawContours(im,[box],0,(255,255,0),6)
      # imx = cv2.resize(im,(1000,600))
      # plt.imshow(imx)
      # plt.title('a')
      # plt.show()
      
      h = np.array([ [0,0],[449,0],[449,449],[0,449] ],np.float32)

      transform = cv2.getPerspectiveTransform(approx,h)
      warp = cv2.warpPerspective(im,transform,(450,450))
      resized = cv2.resize(warp, (138, 210))
      cv2.imwrite(output_dir + '/' + card_names[i], resized)
      # plt.imshow(warp)
      # plt.title('warp')
      # plt.show()
      # pdb.set_trace()

# Usage
# c = CardDetector()
# c.getCards('test_input/wb_on_center.jpg', 'output', 'test_input/wb_on_center_names.txt')
