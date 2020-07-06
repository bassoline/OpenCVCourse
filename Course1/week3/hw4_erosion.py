import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib

# turn plotting on for ipython
matplotlib.rcParams['image.cmap'] = 'gray'
plt.ion()

#create inital image
im = np.zeros((10, 10), dtype='uint8')
im[0, 1] = 1
im[-1, 0] = 1
im[-2, -1] = 1
im[2, 2] = 1
im[5:8, 5:8] = 1

#create ellipse morph 
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
ksize = element.shape[0]
row, col = im.shape[:2]
border = ksize//2

#create padded image
padded_im = np.ones((row+border*2, col+border*2), dtype='uint8')
padded_im[border:border+row, border:border+col] = im
padded_im = cv2.bitwise_not(padded_im) 
paddedDialatedIm = padded_im.copy()

# create video obj
out = cv2.VideoWriter('erosionScratch.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 
	10, (50, 50))
out.write(paddedDialatedIm)

for r in range(0, row):
	for c in range(0, col):
		if np.any(cv2.bitwise_and(padded_im[r: (r+2*border)+1, c: (c + 2*border) + 1], element)):
			#dialate
			paddedDialatedIm[r+border, c+border] = np.amax(cv2.bitwise_and(padded_im[r: (r+2*border)+1, c: (c + 2*border) + 1], element))
			#resize output to 50x50
			#convert to BGR
			tempImage = paddedDialatedIm * 255
			tempImage = cv2.resize(paddedDialatedIm, (50, 50))
			color = cv2.cvtColor(tempImage, cv2.COLOR_GRAY2BGR)
			#add to video
			out.write(color)

# release video obj
out.release()

#display final image (cropped)
paddedDialatedIm = paddedDialatedIm[border: border+row, border: border+col]
plt.figure()
plt.imshow(paddedDialatedIm)