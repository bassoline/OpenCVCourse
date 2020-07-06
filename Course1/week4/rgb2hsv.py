# Import modules
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib
import sys
# set up display properties
plt.ion()
matplotlib.rcParams['figure.figsize'] = (18.0, 12.0)
matplotlib.rcParams['image.interpolation'] = 'bilinear'


def convertBGRtoGray(image):
    shape = image.shape
    b, g, r = cv2.split(image)
    temp = np.ones((shape[0], shape[1]), dtype='float32')
    temp[:,:] = np.uint8(np.round(b*0.114+g*0.587+r*0.299))
    
    return temp


def convertBGRtoHSV(image): 
    #scale and split image
    image = image/255.0
    colors = cv2.split(image)
    b = colors[0]
    g = colors[1]
    r = colors[2]
    
    minArray = np.minimum(np.minimum(b, g), r)
    
    # V = max(b, g, r)
    V = np.maximum(np.maximum(b, g), r)
    
    # S = ( V - min(b, g, r) ) / 2  if  V != 0  else  0 
    S = np.zeros_like(V)
    nonZero = V != 0
    S[nonZero] = (V[nonZero]-minArray[nonZero])/V[nonZero]
    
    # H = (3 different cases reference https://docs.opencv.org/4.1.0/de/d25/imgproc_color_conversions.html)
    H = np.zeros_like(V)
    colors = colors[::-1]
    for i, color in enumerate(colors):
        h_temp = V == color
        H[h_temp] = 120*i+60*(colors[(i+1)%3][h_temp]-colors[(i+2)%3][h_temp])/(V[h_temp]-minArray[h_temp])
        
    # verify H value
    H[H < 0] += 360

    # finalize all values
    V = np.round(V * 255)
    S = np.round(S * 255)
    H = np.round(H/2)
            
    return cv2.merge([H, S, V]).astype(int)

def main():
	img = cv2.imread("../week2/sample.jpg")
	plt.figure(figsize=(18,12))

	if sys.argv[1] == '1':
		# bgr to grey
		gray = convertBGRtoGray(img)
		gray_cv = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

		plt.subplot(1,4,1)
		plt.title("Original")
		plt.imshow(img)
		plt.subplot(1,4,2)
		plt.title("Result from custom function")
		plt.imshow(gray,cmap="gray")
		plt.subplot(1,4,3)
		plt.title("Result from OpenCV function")
		plt.imshow(gray_cv,cmap="gray")
		plt.subplot(1,4,4)
		plt.title("Difference")
		plt.imshow(np.abs(gray-gray_cv),cmap="gray")
		plt.show()
	else: 
		#bgr to hsv
		hsv = convertBGRtoHSV(img)
		hsv_cv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
		plt.subplot(1,3,1)
		plt.title("Result from custom function")
		plt.imshow(hsv[:,:,::-1])
		plt.subplot(1,3,2)
		plt.title("Result from OpenCV function")
		plt.imshow(hsv_cv[:,:,::-1])
		plt.subplot(1,3,3)
		plt.title("Difference")
		plt.imshow(np.abs(hsv-hsv_cv)[:,:,::-1])
		plt.show()

main()
