import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlib

# set up matplot for ipython
plt.ion()
mlib.rcParams['image.cmap'] = 'gray'

#get image
image = cv2.imread('./images/opening.png', cv2.IMREAD_GRAYSCALE)

#set up kernal
kernalSize = 10
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*kernalSize+1, 2*kernalSize+1), (kernalSize, kernalSize))

#perform open operation
imErode = cv2.erode(image, element, iterations=1)
imDilate = cv2.dilate(imErode, element, iterations=1)

#display results
plt.figure()
plt.subplot(131); plt.imshow(image); plt.title('original')
plt.subplot(132); plt.imshow(imErode); plt.title('eroded')
plt.subplot(133); plt.imshow(imDilate); plt.title('dilated')

#perform close operation 
kernalSizeClose = 10
elementClose = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*kernalSizeClose+1, 2*kernalSizeClose+1), (kernalSizeClose, kernalSizeClose))
image2 = cv2.imread('./images/closing.png', cv2.IMREAD_GRAYSCALE)

imageClosed = cv2.morphologyEx(image2, cv2.MORPH_CLOSE, elementClose)
plt.figure()
plt.subplot(121); plt.imshow(image2); plt.title('original')
plt.subplot(122); plt.imshow(imageClosed); plt.title('closed')