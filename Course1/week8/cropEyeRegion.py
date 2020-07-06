import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('./data/models/haarcascade_frontalface_default.xml')

def getCroppedEyeRegion(targetImage):
    
    targetImageGray = cv2.cvtColor(targetImage, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(targetImageGray,1.3,5)
    x,y,w,h = faces[0]

    face_roi = targetImage[y:y+h,x:x+w]
    face_height,face_width = face_roi.shape[:2]

    # Apply a heuristic formula for getting the eye region from face
    eyeTop = int(1/6.0*face_height)
    eyeBottom = int(3/6.0*face_height)
    print("Eye Height between : {},{}".format(eyeTop,eyeBottom))
    
    eye_roi = face_roi[eyeTop:eyeBottom,:]
    
    # Resize the eye region to a fixed size of 96x32
    cropped = cv2.resize(eye_roi,(96, 32), interpolation = cv2.INTER_CUBIC)
    
    return cropped
