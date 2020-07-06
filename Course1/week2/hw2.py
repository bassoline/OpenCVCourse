import cv2
import math

left = []
right = []

# draw rectangle callback
def drawRectangle(action, x, y, flags, userdata):
	global left, right
	if action == cv2.EVENT_LBUTTONDOWN:
		left = (x, y)
		cv2.circle(source, left, 1, (255, 255, 0), 1, cv2.LINE_AA)
	elif action == cv2.EVENT_LBUTTONUP:
		right = (x, y)
		cv2.rectangle(source, left, right, (255, 255, 0), thickness=1)
		cv2.imshow('Window', source)
		# save image
		crop = source[left[1]:right[1]+1,left[0]:right[0]+1]
		cv2.imwrite('face.png', crop)


# read image and create window
source = cv2.imread('sample.jpg')
dummy = source.copy()
cv2.namedWindow('Window')
cv2.setMouseCallback('Window', drawRectangle)
k = 0

# wait for mouse events
while k != 27: 
	cv2.imshow('Window', source)
	cv2.putText(source, 'Choose top left corner and drag!',
		(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
	k = cv2.waitKey(20) & 0xFF

# clean up 
cv2.destroyAllWindows()