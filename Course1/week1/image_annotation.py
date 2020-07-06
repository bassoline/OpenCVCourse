import cv2 
import matplotlib.pyplot as plt
import numpy as np 
import os 

plt.ion()

#add box around face
img = cv2.imread(os.getcwd() + '/images/boy.jpg')
cv2.rectangle(img, (170, 50), (300, 200), (255, 0, 255), 
	thickness=2, lineType=cv2.LINE_8)

#get image size
imgHeight, imgWidth = img.shape[:2]

#define text properties
text = 'asian boi'
fontFace = cv2.FONT_HERSHEY_COMPLEX
fontColor = (250, 10 , 10)
textHeight = 20
thickness = 1
fontScale = cv2.getFontScaleFromHeight(fontFace, textHeight, thickness)
textSize, baseLine = cv2.getTextSize(text, fontFace, fontScale, thickness)
textWidth, textHeight = textSize

#center text (remember (0,0) is top left
x_coord = (imgWidth-textWidth)//2
y_coord = (imgHeight-baseLine-10) 

#draw text box
color = (255, 255, 255)
bottom_left = (x_coord, y_coord+baseLine)
top_right = (x_coord+textWidth, y_coord-textHeight)
cv2.rectangle(img, bottom_left, top_right, color, thickness=-1)

#draw the text 
cv2.putText(img, text, (x_coord, y_coord), fontFace, fontScale, fontColor, thickness, cv2.LINE_AA)

plt.imshow(img[:,:,::-1])

