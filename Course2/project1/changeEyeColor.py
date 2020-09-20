import cv2
import dlib
import numpy as np
import argparse
import matplotlib.pyplot as plt
import math

# plt settings for ipython
plt.ion()
plt.figure()

# drawing constants
font = cv2.FONT_HERSHEY_SIMPLEX

# Landmark ROI
# 37 & 38 correspond to top l->r, 41 & 40 bottom l->r
leftEyeLandmarks = [37, 38, 40, 41]
# same pattern as above
rightEyeLandmarks = [43, 44, 47, 46]

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
face_detector = dlib.get_frontal_face_detector()
# Dlib shape predictor model path
MODEL_PATH = "../week2/data/shape_predictor_68_face_landmarks.dat"
# Load model
shape_predictor = dlib.shape_predictor(MODEL_PATH)

# take in input from command lines
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())
image = args["image"]
image = cv2.imread(image)
window_name = "image"

# display image
def showImage(image, win_name=window_name):
    cv2.imshow(win_name, image)
    # (this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0)
    # closing all open windows
    cv2.destroyAllWindows()


# captures the face ROI, then finds facial landmarks on that face
def getFaceLandmarks(image):
    faces = face_detector(image, 0)
    # we only care about 1 face in this case
    if len(faces) > 0:
        # return the landmarks
        return shape_predictor(image, faces[0])
    else:
        return None


# draws the facial landmarks on the face
def drawEyeLandmarks(
    im, landmarks, landmarksToDraw, color=(0, 255, 0), radius=1, test=False
):
    points = landmarks.parts()
    for landmark in landmarksToDraw:
        p = points[landmark]
        cv2.circle(im, (p.x, p.y), radius, color, -1)
        if test:
            cv2.putText(
                im,
                str(landmark),
                (p.x - 10, p.y - 10),
                font,
                0.5,
                color,
                radius,
                cv2.LINE_AA,
            )
    showImage(im)


# based on the eye radius, and distance between the eyes try and find circles that best represent
# the eye region
def houghTransformCircleDetector(image, eye_radius, eye_distance):
    # parameters
    edge_sensitivity_threshold = 450
    edge_accumulator_threshold = 10
    inverse_accumulator_ratio = 1
    min_circle_distance = eye_distance / 2
    # Convert to gray-scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # img = cv2.medianBlur(gray, 5)
    # Apply hough transform on the images
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        inverse_accumulator_ratio,
        min_circle_distance,
        param1=edge_sensitivity_threshold,
        param2=edge_accumulator_threshold,
        minRadius=int(eye_radius - eye_radius / 4),
        maxRadius=int(eye_radius + eye_radius / 2),
    )
    return circles


# given circles draw them on the image
def drawCircles(img, circles):
    color = (0, 255, 0)
    thickness = 1
    # Draw detected circles
    if circles is not None:
        # round the values
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv2.circle(img, center, radius, color, thickness)
        showImage(img)


# eyes are tough to find with the current version of the landmark detector and houghCircles
# so we end up opting to find more circles, and then filtering out the circles to find the one that
# best represents the eye
def filterOutCircles(circles, landmarks):
    circles = np.uint16(np.around(circles))
    # average x & y of eye region landmarks
    left_distances = []
    right_distances = []
    left_x = left_y = right_x = right_y = 0
    for i in leftEyeLandmarks:
        left_x += landmarks[i].x
        left_y += landmarks[i].y
    for i in rightEyeLandmarks:
        right_x += landmarks[i].x
        right_y += landmarks[i].y
    left_x /= 4
    right_x /= 4
    left_y /= 4
    right_y /= 4
    # find the circle with the smallest distance from the left & right eye region
    for circle in circles[0, :]:
        left_distances.append(
            math.sqrt((circle[0] - left_x) ** 2 + (circle[1] - left_y) ** 2)
        )
        right_distances.append(
            math.sqrt((circle[0] - right_x) ** 2 + (circle[1] - right_y) ** 2)
        )
    left_eye = circles[0, :][left_distances.index(min(left_distances))]
    right_eye = circles[0, :][right_distances.index(min(right_distances))]
    return [left_eye, right_eye]


# creates a histogram of the hues in the eye region
def createHistogramOfColorsInEyeRegion(img, circles, eye_radius, test=False):
    # Convert image to gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # convert image to float so to create a mask
    gray = gray / 255.0
    for circle in circles:
        center = (circle[0], circle[1])
        # get all pixels inside eye radius
        cv2.circle(gray, center, int(eye_radius + eye_radius / 2), 2, -1)
    if test:
        showImage(gray)
    eye_pixels = np.where(gray == 2)
    # get histrogram of colors for eye_pixels
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    eye_pixel_hue = h[eye_pixels]
    # h[0]/h[180] = red, h[60] = green, h[120] = blue
    histogram = np.histogram(eye_pixel_hue, np.arange(180))
    return histogram, eye_pixels, eye_pixel_hue, h, s, v


# changes the most dominate hues of the eye region to another color space
def changeHueOfHistogram(
    histogram, eye_pixels, eye_pixel_hue, h, s, v, color, test=False
):
    largestBins = findLargestNConsecutiveBins(histogram, 30)
    if test:
        print(histogram[0])
        print(largestBins)
    if color == "brown":
        destinationHue = np.concatenate((np.arange(170, 180), np.arange(0, 20)))
    elif color == "blue":
        destinationHue = np.arange(100, 131)
    elif color == "green":
        destinationHue = np.arange(40, 71)
    # loop through the larget bins of hue colors, and map them to the destination color
    for i, bin in enumerate(largestBins):
        eye_pixel_hue[eye_pixel_hue == bin] = destinationHue[i]
    # update hue of eye pixel region
    h[eye_pixels] = eye_pixel_hue
    # recreate hsv image with updated hues, covert back to BGR and display
    newImage = cv2.merge([h, s, v])
    newImage = cv2.cvtColor(newImage, cv2.COLOR_HSV2BGR)
    showImage(newImage)


# finds the largets n consecutive numbers in an array when the array is regarded as circular
def findLargestNConsecutiveBins(histogram, n):
    bins = histogram[0]
    maxSum = 0
    maxBins = []
    lengthOfBins = len(bins)
    for i in range(0, lengthOfBins):
        # max numbers can start at the end of the array and wrap to the beggining so we need to account for those
        # ex: [44, 2, 3, 40, 42] -> we need two arrays [0] & [3:4]
        overflow = i + n > lengthOfBins
        maxNum = i + n if not overflow else lengthOfBins
        maxNum2 = 0 if not overflow else (i + n) % lengthOfBins
        binRange = np.concatenate((np.arange(i, maxNum), np.arange(0, maxNum2)))
        newSum = sum(bins[binRange])

        if newSum > maxSum:
            maxSum = newSum
            maxBins = binRange

    return maxBins


def changeEyeColor(img, eye_color, test=False):
    # grab the landmark points
    landmarks = getFaceLandmarks(img)
    points = landmarks.parts()
    # from landmark points get the eye measurements
    eye_radius = np.uint16((points[38].x - points[37].x) / 2)
    eye_distance = np.uint16(points[43].x - points[38].x)
    # if we're in test mode display the landmarks on the image
    if test:
        drawEyeLandmarks(img.copy(), landmarks, leftEyeLandmarks + rightEyeLandmarks)
        print("eye_radius", eye_radius, "eye_distance", eye_distance)

    circles = houghTransformCircleDetector(img, eye_radius, eye_distance)
    # drawCircles(img.copy(), circles, eye_radius)
    circles = filterOutCircles(circles, points)
    histogram, eye_pixels, eye_pixel_hue, h, s, v = createHistogramOfColorsInEyeRegion(
        img, circles, eye_radius, test
    )
    changeHueOfHistogram(histogram, eye_pixels, eye_pixel_hue, h, s, v, eye_color, test)


changeEyeColor(image, "brown", False)
