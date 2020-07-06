import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

plt.ion()

matplotlib.rcParams["figure.figsize"] = (10.0, 10.0)
matplotlib.rcParams["image.cmap"] = "gray"

# Read image as gray-scale
img = cv2.imread("./images/circles.jpg", cv2.IMREAD_COLOR)
# Convert to gray-scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image to reduce noise
img_blur = cv2.medianBlur(gray, 5)
# Apply hough transform on the image
circles = cv2.HoughCircles(
    img_blur,
    cv2.HOUGH_GRADIENT,
    1,
    50,
    param1=450,
    param2=10,
    minRadius=30,
    maxRadius=40,
)
# Draw detected circles
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Draw outer circle
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw inner circle
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

plt.imshow(img[:, :, ::-1])

