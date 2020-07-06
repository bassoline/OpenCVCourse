import cv2 
import matplotlib.pyplot as plt
import numpy as np 
import os 

# open video 
video = cv2.VideoCapture(os.getcwd() + '/chaplin.mp4')

#get meta data of video 
width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)


#display video 
while (video.isOpened()):
	#read each frame
	ret, frame = video.read()

	if ret == True:
		cv2.imshow('Video Output', frame)
		cv2.waitKey(1)
	else: 
		break


video.release()