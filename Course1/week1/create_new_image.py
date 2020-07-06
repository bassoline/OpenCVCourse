import cv2 
import matplotlib.pyplot as plt
import os 
import matplotlib 
import numpy as np 

#workoncv-4.1.0
#ipython

matplotlib.pyplot.ion()
plt.figure()

img = cv2.imread(os.getcwd() + '/images/boy.jpg')
plt.subplot(131)
plt.imshow(img[:,:,::-1])
imageCopy = img.copy()

#create empty matrix of same size as image
plt.subplot(132)
emptyMatrix = np.zeros_like(img, dtype='uint8')
plt.imshow(emptyMatrix)

plt.subplot(133)
blankMatrix = 255*np.ones_like(img, dtype='uint8')
plt.imshow(blankMatrix)

# crop with slicing 
cpy = img.copy()
crop = img[40:200,170:320]
roiHeight, roiWidth = crop.shape[:2]
cpy[40:40+roiHeight, 10:10+roiWidth] = crop
cpy[40:40+roiHeight, 330:330+roiWidth] = crop

#scale an image 
scaleFactor = 1.5
scaledImage = cv2.resize(img, None, fx=scaleFactor, fy=scaleFactor, interpolation=cv2.INTER_LINEAR)

plt.figure()
plt.subplot(141)
plt.imshow(img[:,:,::-1])
plt.subplot(142)
plt.imshow(crop[:,:,::-1])
plt.subplot(143)
plt.imshow(cpy[:,:,::-1])
plt.subplot(144)
plt.imshow(scaledImage[:,:,::-1])