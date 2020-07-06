import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib
matplotlib.rcParams['image.cmap'] = 'gray'
plt.ion()
#get image
image = cv2.imread('./images/truth.png', cv2.IMREAD_GRAYSCALE)

#threshold image 
th, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

#create padded image
row, col = image.shape[:2]
padded = np.zeros((row+1, col+1), dtype='uint8')
padded[1:, 1:] = image

#store equivelencies
equiv = {}
blobCount = 1

#loop through pixels once for finding blobs
for r in range(row):
	for c in range(col):
		# only care about white pixels
		if image[r, c] == 255:
			if padded[r, c+1] == 0 and padded[r+1, c] == 0:
				padded[r+1, c+1] = blobCount
				blobCount += 1
			elif padded[r, c+1] == 0: 
				padded[r+1, c+1] = padded[r+1, c]
			elif padded[r+1, c] == 0: 
				padded[r+1, c+1] = padded[r, c+1]
			else: 
				mini = padded[r, c+1]
				maxi = padded[r+1, c]
				if mini > maxi:
					temp = mini
					mini = maxi
					maxi = temp
				padded[r+1, c+1] = mini
				
				if mini != maxi:
					if mini not in equiv:
						equiv[mini] = [maxi]
					else: 
						equiv[mini].append(maxi)


#replace all pixels based on equivelence relations
for key, value in equiv.items():
	for v in value:
		padded[padded == v] = key

#remove padded section
padded = padded[1:, 1:]	

#color map the image
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(padded)
padded = 255*(padded-minVal)/(maxVal-minVal)
padded = np.uint8(padded)
imColorMap = cv2.applyColorMap(padded, cv2.COLORMAP_JET)
imColorMap = imColorMap[:,:,::-1]

#display image
plt.figure()
plt.subplot(121)
plt.imshow(image)
plt.subplot(122)		
plt.imshow(imColorMap)

