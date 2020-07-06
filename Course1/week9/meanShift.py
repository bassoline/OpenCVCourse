import numpy as np
import cv2,sys,time
from dataPath import DATA_PATH

if __name__ == '__main__':

  filename = DATA_PATH + "videos/face1.mp4"
  cap = cv2.VideoCapture(filename)

  # Read a frame and find the face region using dlib
  ret,frame = cap.read()


  # Detect faces in the image
  faceCascade = cv2.CascadeClassifier(DATA_PATH + 'models/haarcascade_frontalface_default.xml')

  frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  faces = faceCascade.detectMultiScale(frameGray,1.3,5)
  x,y,w,h = faces[0]

  currWindow = (x,y,w,h)
  # get the face region from the frame
  roiObject = frame[y:y+h,x:x+w]
  hsvObject =  cv2.cvtColor(roiObject, cv2.COLOR_BGR2HSV)

  # Get the mask for calculating histogram of the object and also remove noise
  mask = cv2.inRange(hsvObject, np.array((0., 50., 50.)), np.array((180.,255.,255.)))
  cv2.imshow("mask",mask)
  cv2.imshow("Object",roiObject)

  # Find the histogram and normalize it to have values between 0 to 255
  histObject = cv2.calcHist([hsvObject], [0], mask, [180], [0,180])
  cv2.normalize(histObject, histObject, 0, 255, cv2.NORM_MINMAX)

  # Setup the termination criteria, either 10 iterations or move by atleast 1 pt
  term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

  while(1):
    t = time.time()
    ret , frame = cap.read()
    if ret == True:
      # Convert to hsv color space
      hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

      # find the back projected image with the histogram obtained earlier
      backProjectImage = cv2.calcBackProject([hsv], [0], histObject, [0,180], 1)
      cv2.imshow("Back Projected Image", backProjectImage)

      # Compute the new window using mean shift in the present frame
      ret, currWindow = cv2.meanShift(backProjectImage, currWindow, term_crit)

      # Display the frame with the tracked location of face
      x,y,w,h = currWindow
      frameClone = frame.copy()
      cv2.rectangle(frameClone, (x,y), (x+w,y+h), (255,0,0), 2, cv2.LINE_AA)
      cv2.imshow('Mean Shift Object Tracking Demo',frameClone)

      k = cv2.waitKey(10) & 0xff
      if k == 27:
        break
      # print(time.time() - t)
    else:
      break

  cap.release()
  cv2.destroyAllWindows()
