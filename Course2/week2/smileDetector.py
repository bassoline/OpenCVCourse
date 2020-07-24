import cv2
import dlib
import numpy as np

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
# Dlib shape predictor model path
MODEL_PATH = "./data/shape_predictor_68_face_landmarks.dat"

# Load model
shape_predictor = dlib.shape_predictor(MODEL_PATH)

def smile_detector(imDlib):
    # Detect faces
    faces = detector(imDlib, 0)

    if len(faces):
        landmarks = shape_predictor(imDlib, faces[0])
    else:
        return False

    # determine if the person is smiling by checking the lip to jaw ratio
    # based on some analysis, looks like 0.50 is a good cut off point
    landmarks = landmarks.parts()
    left_lip = landmarks[48].x
    right_lip = landmarks[54].x
    left_cheek = (landmarks[3].x + landmarks[4].x)/2
    right_cheek = (landmarks[13].x + landmarks[12].x)/2
    lip_to_cheek_ratio = (right_lip-left_lip)/(right_cheek-left_cheek)
    isSmiling = False if lip_to_cheek_ratio < 0.5 else True
    # Return True if smile is detected
    return isSmiling

# Initializing video capture object.
capture = cv2.VideoCapture("./media/smile.mp4")
if(False == capture.isOpened()):
    print("[ERROR] Video not opened properly")    

# Create a VideoWriter object
smileDetectionOut = cv2.VideoWriter("smileDetectionOutput.avi",
                                   cv2.VideoWriter_fourcc('M','J','P','G'),
                                   15,(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                                       int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
frame_number = 0
smile_frames = []
while (True):
    # grab the next frame
    isGrabbed, frame = capture.read()
    if not isGrabbed:
        break
        
    imDlib = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_has_smile = smile_detector(imDlib)
    if (True == frame_has_smile):
        cv2.putText(frame, "Smiling :)", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        smile_frames.append(frame_number)
    # print("Smile detected in Frame# {}".format(frame_number))
    if frame_number % 50 == 0:
        print('\nProcessed {} frames'.format(frame_number))
        print("Smile detected in Frames: {}".format(smile_frames))
    # Write to VideoWriter
    smileDetectionOut.write(frame)
    
    frame_number += 1

capture.release()
smileDetectionOut.release()
