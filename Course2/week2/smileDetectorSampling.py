import cv2
import dlib 
import numpy as np 
import argparse
import matplotlib.pyplot as plt

# intialize dlib's HOG-based face detector and create landmark predictor 
face_detector = dlib.get_frontal_face_detector()
detector_path = "./data/shape_predictor_68_face_landmarks.dat"
shape_predictor = dlib.shape_predictor(detector_path)
font = cv2.FONT_HERSHEY_SIMPLEX

# take in input from command lines
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# draws the facial landmarks on the face 
def drawLandmarks(im, landmarks, color=(0, 255, 0), radius=3):
  for i,p in enumerate(landmarks.parts()):
    cv2.circle(im, (p.x, p.y), radius, color, -1)
    cv2.putText(im,str(i),(p.x-10, p.y-10), font,
            0.5,(255,255,255),2,cv2.LINE_AA)


def getFaceLandmarks(image):
    faces = face_detector(image, 0)
    # we only care about 1 face in this case
    if len(faces) > 0:
        # return the landmarks 
        return shape_predictor(image, faces[0])
    else: 
        return None


def mouthToJawRatio(landmarks):
    landmarks = landmarks.parts()
    leftLipLandmark = landmarks[48].x
    rightLipLandmark = landmarks[54].x
    leftCheekLandmark = (landmarks[3].x+landmarks[4].x)/2
    rightCheekLandmark = (landmarks[13].x+landmarks[12].x)/2
    print("lip to jaw ratio", (rightLipLandmark-leftLipLandmark)/(rightCheekLandmark-leftCheekLandmark))

# validate with one image
def validateWithSingleImage():
    image = args["image"]
    image = cv2.imread(image)
    landmarks = getFaceLandmarks(image)
    drawLandmarks(image, landmarks)
    mouthToJawRatio(landmarks)
    plt.imshow(image[:,:,::-1])
    plt.show()

# validate with all images
for i in range(1,5):
    smiling_image = cv2.imread("./media/smiling_p{}.png".format(i))
    smiling_landmarks = getFaceLandmarks(smiling_image)
    print('smiling')
    mouthToJawRatio(smiling_landmarks)
    # no image of p2 not smiling
    if i != 2:
        notSmiling_image = cv2.imread("./media/notSmiling_p{}.png".format(i))
        not_smiling_landmarks = getFaceLandmarks(notSmiling_image)
        print('not smiling')
        mouthToJawRatio(not_smiling_landmarks)



