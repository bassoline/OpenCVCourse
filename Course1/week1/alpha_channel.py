import cv2 
import matplotlib.pyplot as plt
import os 
import matplotlib 

matplotlib.pyplot.ion()
imagePath = os.getcwd() + '/images/panther.png'
imgPNG = cv2.imread(imagePath, -1)
print("image Dimension = {}".format(imgPNG.shape))
imgBGR = imgPNG[:, :, 0:3]
imgMask = imgPNG[:, :, 3]

plt.figure()
plt.subplot(121)
plt.imshow(imgBGR[:,:,::-1])
plt.title('color image')

plt.subplot(122)
plt.imshow(imgMask, cmap='gray')
plt.title('Alpha Channel')
