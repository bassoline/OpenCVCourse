import cv2 

# scale vars 
maxScaleUp = 100
scaleFactor = 1
scaleType = 0
maxType = 1

# string vars
windowName = 'Resize Image'
trackbarValue = 'Scale'
trackbarType = 'Type: \n 0: Scale Up \n 1: Scale Down'

#load image
im = cv2.imread('truth.png')

#create window 
cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)

#scale image function
def scaleImage(scaleFactor, scaleType, im, windowName):
	 
	if scaleType == 0:
		scaleFactor = 1+scaleFactor/100.0
	else: 
		scaleFactor = 1-scaleFactor/100.0

	if scaleFactor <= 0.01: 
	 	scaleFactor = .01

	scaledImage = cv2.resize(im, None, fx=scaleFactor, 
		fy=scaleFactor, interpolation=cv2.INTER_LINEAR)

	cv2.imshow(windowName, scaledImage)


#scale callback 
def scaleImageCallback(*args):
	global scaleFactor, scaleType
	scaleFactor = args[0]
	
	scaleImage(scaleFactor, scaleType, im, windowName)


# scale type callback
def scaleTypeImageCallback(*args):
	global scaleType, scaleFactor
	scaleType = args[0]

	scaleImage(scaleFactor, scaleType, im, windowName)


cv2.createTrackbar(trackbarValue, windowName, scaleFactor, maxScaleUp, scaleImageCallback)
cv2.createTrackbar(trackbarType, windowName, scaleType, maxType, scaleTypeImageCallback)

cv2.imshow(windowName, im)
c = cv2.waitKey(0)
cv2.destroyAllWindows()
