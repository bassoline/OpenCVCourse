import cv2
import matplotlib.pyplot as plt
import numpy as np
import os 

plt.ion()

image = cv2.imread(os.getcwd() + '/images/boy.jpg')
contrasted = image*(1.3)

clippedImage = np.clip(contrasted, 0, 255)
contrasted_uint8 = np.uint8(clippedImage)

brightnessOffset = 50
brightnessClipped = np.int32(image) + brightnessOffset
brightnessClipped = np.clip(brightnessClipped, 0, 255)

clearer = cv2.resize(image, None, fx=1.5, fy=1.5)

plt.figure()
plt.subplot(221)
plt.imshow(image[:,:,::-1])
plt.title('og image')

plt.subplot(222)
plt.imshow(contrasted_uint8[:,:,::-1])
plt.title('contrasted image')

plt.subplot(223)
plt.imshow(brightnessClipped[:,:,::-1])
plt.title('brightness increased')

plt.subplot(224)
plt.imshow(clearer[:,:,::-1])
plt.title('clearer image')