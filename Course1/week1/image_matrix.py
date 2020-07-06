import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib

# get image and set to gray scale and inline 
matplotlib.pyplot.ion()
imagePath = os.getcwd() + '/images/number_zero.jpg'
testImage = cv2.imread(imagePath, 0)

# print image and meta data
print('\n{}\n'.format(testImage))
print("\nData type = {}".format(testImage.dtype))
print("Object type = {}".format(type(testImage)))
print("Image dimenstions = {}".format(testImage.shape))

#modify pixel
testImage[0,0] = 200
# print(testImage)

#modify region
test_roi = testImage[0:2, 0:4]
print("Original Matrix\n{}\n".format(testImage))
print("Selected Region\n{}\n".format(test_roi))

testImage[0:2, 0:4] = 111
print('Modified Matrix\n{}\n'.format(testImage))

#printing the actual picture itself (using matplotlib)
plt.imshow(testImage)
plt.colorbar()

#printing image using cv2
cv2.imshow('image',testImage)

