import cv2
import matplotlib.pyplot as plt
import numpy as np

# needed for plotting in cmd with matplotlib
plt.ion()

# landmark constants
L_EYE_LM = [36, 37, 38, 39, 40, 41]
R_EYE_LM = [42, 43, 44, 45, 46, 47]
L_EYELASH_LM = [
    (322, 262),
    (415, 187),
    (618, 182),
    (799, 336),
    (600, 375),
    (400, 362),
]

R_EYELASH_LM = [
    (1, 336),
    (182, 182),
    (385, 187),
    (478, 262),
    (400, 362),
    (200, 375),
]

# eye lash constants
EYE_WIDTH = 477
LEFT_EYE_LASH_PATH = "./eye_lash_left.jpg"
RIGHT_EYE_LASH_PATH = "./eye_lash_right.jpg"

# Dlib shape predictor model path
MODEL_PATH = "../week2/data/shape_predictor_68_face_landmarks.dat"

# create mask for image with no background
def createMask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray.copy()
    foregroundPixels = mask < 100
    backgroundPixels = mask >= 100
    mask[foregroundPixels] = 255
    mask[backgroundPixels] = 0
    return mask


# display image using maplot, need to be running ipython for this work
def plotImage(image):
    plt.figure()
    plt.imshow(image, cmap="gray")


# returns facial landmarks given an image
def getFacialLandmarks(dlib_model_path, image):
    # initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
    print("[INFO] loading facial landmark predictor...")
    face_detector = dlib.get_frontal_face_detector()
    # Load model
    shape_predictor = dlib.shape_predictor(dlib_model_path)
    # captures the face ROI
    faces = face_detector(image, 0)
    # we only care about 1 face in this case
    if len(faces) > 0:
        # return the facial landmarks
        return shape_predictor(image, faces[0])
    else:
        return None


# create image with left & right eyelashes at same height, size and spacing
# as eyes in face picture (orientation doesn't matter, we'll take care of that with the delanuy trianglulation)
def createEyeLashImage(dlib_model_path, image):
    # get facial landmarks
    facial_landmarks = getFacialLandmarks(dlib_model_path, image)
    # pinpoint eye landmark positions in the image
    l_center_x, l_center_y, l_width = getEyeCenterAndWidth(facial_landmarks, L_EYE_LM)
    r_center_x, r_center_y, r_width = getEyeCenterAndWidth(facial_landmarks, R_EYE_LM)
    # reduce eyelashes to be similar size as eye size
    scale = ((l_width + r_width) / 2) / EYE_WIDTH
    # get and scale eye lash images
    l_eye_lash_image = cv2.imread(LEFT_EYE_LASH_PATH)
    r_eye_lash_image = cv2.imread(RIGHT_EYE_LASH_PATH)
    # scale TODO
    # create blank image with the same dimensions as the original image
    eye_lash_image = np.zeros_like(image.shape)
    # determine pixels to replace for r_eye_lash and l_eye_lash
    center_eye_lash = l_eye_lash_image.shape
    # update image to include eye_lashes in approximate locations
    # create mask for eyelashes
    # return eyelash image, and mask of eyelashes


# returns an averaged center of the eye, and the approx width of the eye
def getEyeCenterAndWidth(facial_landmarks, eye_landmarks):
    eye_landmarks = []
    center_x = 0
    center_y = 0
    for i in eye_landmarks:
        lm = facial_landmarks[i]
        eye_landmarks.append((lm.x, lm.y))
        center_x += lm.x
        center_y += lm.y

    center_x /= len(eye_landmarks)
    center_y /= len(eye_landmarks)
    width = eye_landmarks[3][0] - eye_landmarks[0][0]
    return (center_x, center_y, width)


# use delanuy transformation to align eyelashes with face
def delanuyTransformation():
    pass


# alpha blend images
def alphaBlendImages(image, eyeLashes, mask):
    pass


image = cv2.imread("./eyelash_right.jpg")
mask = createMask(image)
plotImage(mask)
