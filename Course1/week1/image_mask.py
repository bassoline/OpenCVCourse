import cv2
import matplotlib.pyplot as plt
import os 
import numpy as np 

plt.ion()

# make the image more clear (scale it up)
img = cv2.imread(os.getcwd() + '/images/boy.jpg')
sf = 1.5 
img = cv2.resize(img, None, fx=sf, fy=sf)

# create mask based on pixel location
mask1 = np.zeros_like(img)
mask1[75:300, 300:475] = 255

#create mask based of 'color', pixel intensities
#trying to locate the red pixels 
mask2 = cv2.inRange(img, (0,0,150), (100, 100, 255))

#plot the original and the mask 
plt.figure()
plt.subplot(131)
plt.imshow(img[:,:,::-1])
plt.title('clear image')

plt.subplot(132)
plt.imshow(mask1[:,:,::-1])
plt.title('manual masked image')

plt.subplot(133)
plt.imshow(mask2)
plt.title('masked for red')



