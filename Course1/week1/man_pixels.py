import cv2 
import matplotlib.pyplot as plt
import matplotlib
import os 

matplotlib.pyplot.ion()
imagePath = os.getcwd() + '/images/number_zero.jpg'
testImage = cv2.imread(imagePath,1)
plt.imshow(testImage)

print(testImage[0,0])

# modify individual pixels
testImage[0,0] = (0,255,255)
plt.subplot(141)
plt.imshow(testImage[:, : ,::-1])

testImage[1,1] = (255, 255, 0)
plt.subplot(142)
plt.imshow(testImage[:, :, ::-1])

testImage[2,2] = (255, 0, 255)
plt.subplot(143)
plt.imshow(testImage[:, :, ::-1])

#modify groups of pixels
testImage[0:3, 0:3] = (0, 255, 255)
plt.subplot(144)
plt.imshow(testImage[:,:,::-1])