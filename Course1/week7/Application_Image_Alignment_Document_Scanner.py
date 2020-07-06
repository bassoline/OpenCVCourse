import cv2
import numpy as np
from dataPath import DATA_PATH

# Read reference image
refFilename = DATA_PATH + "images/form.jpg"
print("Reading reference image : ", refFilename)
imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

cv2.imshow("Reference Image",imReference)
cv2.waitKey(0)

# Read image to be aligned
imFilename = DATA_PATH + "images/scanned-form.jpg"
print("Reading image to align : ", imFilename);
im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

cv2.imshow("Image to align", im)
cv2.waitKey(0)

cv2.destroyAllWindows()

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

# Convert images to grayscale
im1Gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im2Gray = cv2.cvtColor(imReference, cv2.COLOR_BGR2GRAY)

# Detect ORB features and compute descriptors.
orb = cv2.ORB_create(MAX_FEATURES)
keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

# Match features.
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
matches = matcher.match(descriptors1, descriptors2, None)

# Sort matches by score
matches.sort(key=lambda x: x.distance, reverse=False)

# Remove not so good matches
numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
matches = matches[:numGoodMatches]

# Draw top matches
imMatches = cv2.drawMatches(im, keypoints1, imReference, keypoints2, matches, None)

cv2.imshow("Matches",imMatches)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Extract location of good matches
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

# Find homography
h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

# Use homography
height, width, channels = imReference.shape
im1Reg = cv2.warpPerspective(im, h, (width, height))

cv2.imshow("Warped Image",im1Reg)
cv2.waitKey(0)

# Print estimated homography
print("Estimated homography : \n",  h)

cv2.destroyAllWindows()
