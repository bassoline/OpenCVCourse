import cv2
import os
import numpy as np 
import matplotlib.pyplot as plt 

plt.ion()

#read image 
img = cv2.imread(os.getcwd() + '/images/IDCard-Satya.png')

#create a qrCodeDetector obj 
qrDecoder = cv2.QRCodeDetector()

#detect qr code in the image
opencvData, bbox, rectifiedImage = qrDecoder.detectAndDecode(img)

#draw bounding box around the detected QR code
n = len(bbox)
print(bbox[0][0])
for i in range(n):
	cv2.line(img, tuple(bbox[i][0]), tuple(bbox[(i+1)%4][0]), (255,0,0), thickness=3, lineType=cv2.LINE_AA)

# #print decoded text
print(opencvData)

#save final image
cv2.imwrite('QRCode-Output.png', img)

#display final image
plt.imshow(img[:,:,::-1])
