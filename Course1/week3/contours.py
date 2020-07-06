import cv2 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

plt.ion()
matplotlib.rcParams['image.cmap'] = 'gray'

image = cv2.imread('./images/Contour.png')
image1 = image.copy()
image2 = image.copy()
image3 = image.copy()

imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
contours, hierarchy = cv2.findContours(imageGray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

print('Number of contours found = {}'.format(len(contours)))
print('\nHierarchy: \n{}'.format(hierarchy))
cv2.drawContours(image, contours, -1, (0, 255, 0), 3);


for i, cnt in enumerate(contours):
	# get area, perimeter, and centroid
	M = cv2.moments(cnt)
	print("area of contour {} is {}, perimeter is {}".format(i+1, M['m00'], cv2.arcLength(cnt, True)))
	x = int(round(M['m10']/M['m00']))
	y = int(round(M['m01']/M['m00']))
	cv2.circle(image, (x,y), 5, (0, 0, 255), -1)

	#plot bounding box
	x, y, w, h = cv2.boundingRect(cnt)
	cv2.rectangle(image1, (x,y), (x+w, y+h), (255, 0, 255), 2)

	#plot optimal bounding box
	box = cv2.minAreaRect(cnt)
	boxPts = np.int0(cv2.boxPoints(box))
	cv2.drawContours(image2, [boxPts], -1, (0, 255, 255), 2)

	# plot circle
	((x, y), radius) = cv2.minEnclosingCircle(cnt)
	cv2.circle(image3, (int(x), int(y)), int(round(radius)), (0, 255, 0), 2)


plt.figure()
plt.subplot(221)
plt.imshow(image[:,:,::-1])
plt.subplot(222)
plt.imshow(image1[:,:,::-1])
plt.subplot(223)
plt.imshow(image2[:,:,::-1])
plt.subplot(224)
plt.imshow(image3[:,:,::-1])
