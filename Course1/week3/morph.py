import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import os 

plt.ion()

# create inital image
im = np.zeros((10,10), dtype='uint8')
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

#    dilation from scratch 
# create padded image (for edge cases)
padded_im = np.zeros((row+border*2, col+border*2))
padded_im = cv2.copyMakeBorder(im, border, border, border, border, 
	cv2.BORDER_CONSTANT, value = 0)

# loop through all pixels and dilate white pixels
for r in range(0, row):
	for c in range(0, col):
		if im[r,c]:
			print('white pixel found @ {}, {}'.format(r, c))

			padded_im[r: (r + 2*border) + 1, c: (c + 2*border) + 1] = \
				cv2.bitwise_or(padded_im[r: (r + 2*border) + 1, c: (c + 2*border) + 1], element)


#crop out original picture
dilatedImage = padded_im[border: border+row, border: border+col]

#using opencv 
dilatedImageOpenCV = cv2.dilate(im, element)
plt.figure()
plt.imshow(dilatedImageOpenCV)

#display results
print(element)
print(im)
plt.figure()
plt.subplot(131)
plt.imshow(im)
plt.subplot(132)
plt.imshow(padded_im)
plt.subplot(133)
plt.imshow(dilatedImage)