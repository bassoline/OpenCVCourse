import cv2 
import matplotlib.pyplot as plt
import matplotlib
import os 

imagePath = os.getcwd() + '/images/musk.jpg'
img = cv2.imread(imagePath)
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# using matplot lib to split
matplotlib.pyplot.ion()
print("image dimension = {}".format(img.shape))
plt.imshow(imgRGB)

plt.figure(figsize=[20,5])
plt.subplot(131)
plt.imshow(img[:,:,0])
plt.title('Blue Channel')

plt.subplot(132)
plt.imshow(img[:,:,1])
plt.title('Green Channel')

plt.subplot(133)
plt.imshow(img[:,:,2])
plt.title('Red Channel')

# using cv2 to split
b,g,r = cv2.split(img)
plt.figure(figsize=[20,5])

plt.subplot(141)
plt.imshow(b)
plt.title('Blue Channel')

plt.subplot(142)
plt.imshow(g)
plt.title('Green Channel')

plt.subplot(143)
plt.imshow(r)
plt.title('Red Channel')

# merge images
imgMerged = cv2.merge((b,g,r))
plt.subplot(144)
plt.imshow(imgMerged[:,:,::-1])
plt.title('Merged output')