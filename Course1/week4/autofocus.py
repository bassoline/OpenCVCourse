# Import modules
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

plt.ion()

matplotlib.rcParams["figure.figsize"] = (10.0, 10.0)
matplotlib.rcParams["image.interpolation"] = "bilinear"


def get_roi():
    # Read input video filename
    filename = "./videos/focus-test.mp4"

    # Create a VideoCapture object
    cap = cv2.VideoCapture(filename)

    # Read first frame from the video
    ret, frame = cap.read()

    # Display total number of frames in the video
    print("Total number of frames : {}".format(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    roi = frame[20:620, 450:1000]
    return roi


def var_abs_laplacian(image):
    # taken from *paper name*
    # perform laplacian + variance
    convolved = cv2.Laplacian(image, cv2.CV_32F, ksize=1, scale=1 / 6, delta=0)
    return np.sum(np.square(convolved))


def sum_modified_laplacian(im):
    # create padded image
    row, col = im.shape
    padded = np.zeros((row + 1, col + 1), dtype="uint8")
    padded[1:, 1:] = blurred

    # perform modified laplacian
    # take from *paper name*
    times2 = 2 * padded
    rowPlus = np.roll(padded, 1, axis=0)
    rowMinus = np.roll(padded, -1, axis=0)
    colPlus = np.roll(padded, 1, axis=1)
    colMinus = np.roll(padded, -1, axis=1)

    output = abs(times2 - rowPlus - rowMinus) + abs(times2 - colPlus - colMinus)
    return np.sum(output)


# Read input video filename
filename = "./videos/focus-test.mp4"

# Create a VideoCapture object
cap = cv2.VideoCapture(filename)

# Read first frame from the video
ret, frame = cap.read()

# Display total number of frames in the video
print("Total number of frames : {}".format(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

maxV1 = 0
maxV2 = 0

# Frame with maximum measure of focus
# Obtained using methods 1 and 2
bestFrame1 = 0
bestFrame2 = 0

# Frame ID of frame with maximum measure
# of focus
# Obtained using methods 1 and 2
bestFrameId1 = 0
bestFrameId2 = 0

# Specify the ROI for flower in the frame
# UPDATE THE VALUES BELOW
top = 450
left = 20
bottom = 1000
right = 620

i = 0
plt.figure()
plt.imshow(frame[left:right, top:bottom])

# Iterate over all the frames present in the video
while ret:
    # Crop the flower region out of the frame
    flower = frame[left:right, top:bottom]
    gray = cv2.cvtColor(flower, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0, 0)

    # Get measures of focus from both methods
    val1 = var_abs_laplacian(blurred)
    val2 = sum_modified_laplacian(blurred)

    # If the current measure of focus is greater
    # than the current maximum
    if val1 > maxV1:
        # Revise the current maximum
        maxV1 = val1
        # Get frame ID of the new best frame
        bestFrameId1 = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # Revise the new best frame
        bestFrame1 = frame.copy()
        print("Frame ID of the best frame [Method 1]: {}".format(bestFrameId1))

    # If the current measure of focus is greater
    # than the current maximum
    if val2 > maxV2:
        # Revise the current maximum
        maxV2 = val2
        # Get frame ID of the new best frame
        bestFrameId2 = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # Revise the new best frame
        bestFrame2 = frame.copy()
        print("Frame ID of the best frame [Method 2]: {}".format(bestFrameId1))

    # Read a new frame
    ret, frame = cap.read()
    i += 1


print("================================================")
# Print the Frame ID of the best frame
print("Frame ID of the best frame [Method 1]: {}".format(bestFrameId1))
print("Frame ID of the best frame [Method 2]: {}".format(bestFrameId2))

# Release the VideoCapture object
cap.release()

# Stack the best frames obtained using both methods
out = np.hstack((bestFrame1, bestFrame2))

# Display the stacked frames
plt.figure()
plt.imshow(out[:, :, ::-1])
plt.axis("off")

