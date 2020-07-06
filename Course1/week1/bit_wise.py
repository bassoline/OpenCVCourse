import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

plt.ion()

# get images 
musk = cv2.imread(os.getcwd() + '/images/musk.jpg')
glasses = cv2.imread(os.getcwd() + '/images/sunglass.png', -1)
muskCopy = musk.copy()

#resize and seperate sunglasses
sunglassesResized = cv2.resize(glasses, (300, 100))
sunglassesAlpha = sunglassesResized[:,:,3]
sunglassesBGR = sunglassesResized[:,:,0:3]

#get sunglasses ROI
eyeROI = muskCopy[150:250, 140:440]

#make mask 3-D
glassesMask = cv2.merge((sunglassesAlpha, sunglassesAlpha, sunglassesAlpha))
maskedEyeROI = cv2.bitwise_and(eyeROI, cv2.bitwise_not(glassesMask))
maskedFaceROI = cv2.bitwise_and(sunglassesBGR, glassesMask)

#create final image
finalROI = cv2.bitwise_or(maskedEyeROI, maskedFaceROI)
muskCopy[150:250, 140:440] = finalROI

#display image
plt.figure()
plt.subplot(121)
plt.imshow(musk[:,:,::-1])
plt.subplot(122)
plt.imshow(muskCopy[:,:,::-1])

