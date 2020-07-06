import cv2 
import matplotlib.pyplot as plt
import numpy as np
import os 

plt.ion()
#get images
currentDirectory = os.getcwd()
glasses = cv2.imread(currentDirectory + '/images/sunglass.png', -1)
glassesPNG1 = cv2.resize(glasses, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
glassesPNG = cv2.resize(glassesPNG1, (300,100))
elon = cv2.imread(currentDirectory + '/images/musk.jpg')
elon_copy = elon.copy()
hat = cv2.imread(currentDirectory + '/images/hat2.png')
print('hat dim: {}'.format(hat.shape))

#get the alpha channel, and make it a 3-d channel (for RGB)
glassesMask = glassesPNG[:,:,3]
glassesTuple = cv2.merge((glassesMask, glassesMask, glassesMask))

#localize pixels to either 0 or 1 for binary mask 
glassesTuple = np.uint8(glassesTuple/255)

#get eye region
eyeROI = elon[150:250, 140:440]

# mask the eye region w/ sunglasses
maskedEye = cv2.multiply(eyeROI, (1 - glassesTuple))

# mask the face reigion (keep sunglasses)
glassesPNG = glassesPNG[:,:,0:3]
glassesPNG = glassesPNG[:,:,::-1]
maskedFace = cv2.multiply(glassesPNG, glassesTuple)

#eye region with glasses
eyeWithGlasses = cv2.add(maskedEye, maskedFace)
elon_copy[150:250, 140:440] = eyeWithGlasses

#plt images
plt.figure()
plt.subplot(221)
plt.imshow(glassesPNG)
plt.title('glasses')
plt.subplot(222)
plt.imshow(glassesPNG1)
plt.title('clear glasses')
plt.subplot(223)
plt.imshow(elon[:,:,::-1])
plt.title('Elon')
plt.subplot(224)
plt.imshow(elon_copy[:,:,::-1])
plt.title('Elon with Glasses')