# Standard imports
import cv2
import numpy as np;
from dataPath import DATA_PATH

# Create a black image of size 200x200
im = np.zeros((200,200,3), np.float32)

# Create a blue square in the center
im[50:150,50:150,0] = 1;
im[50:150,50:150,1] = 0.6;
im[50:150,50:150,2] = 0.2

cv2.imshow("Original Image",im)
cv2.waitKey(0)

# Output dimension
outDim = im.shape[0:2]

# Translate by 25,25
warpMat = np.float32(
    [
        [1.0, 0.0, 25],
        [0,   1.0, 25]
    ])

# Warp image
result = cv2.warpAffine(im, warpMat, outDim)

# Display image
cv2.imshow("Translate Image",result)
cv2.waitKey(0)

# Scale along x direction
warpMat = np.float32(
    [
        [2.0, 0.0, 0],
        [0,   1.0, 0]
    ])

# Warp image
result = cv2.warpAffine(im, warpMat, outDim)
cv2.imshow("Scale along x direction", result)
cv2.waitKey(0)

# Scale along x direction
warpMat = np.float32(
    [
        [2.0, 0.0, 0],
        [0,   1.0, 0]
    ])

result = cv2.warpAffine(im, warpMat, (2 * outDim[0], outDim[1]))

cv2.imshow("Scale image width", result)
cv2.waitKey(0)

# Scale along x and y directions
warpMat = np.float32(
    [
        [2.0, 0.0, 0],
        [0,   2.0, 0]
    ])

# Warp image
result = cv2.warpAffine(im, warpMat, (2 * outDim[0], 2 * outDim[1]))
cv2.imshow("Scale along both dimensions", result)
cv2.waitKey(0)

# Rotate image
angleInDegrees = 30
angleInRadians = 30 * np.pi / 180.0

cosTheta = np.cos(angleInRadians)
sinTheta = np.sin(angleInRadians)

# Rotation matrix
# https://en.wikipedia.org/wiki/Rotation_matrix

warpMat = np.float32(
    [
        [ cosTheta, sinTheta, 0],
        [ -sinTheta, cosTheta, 0]
    ])

# Warp image
result = cv2.warpAffine(im, warpMat, outDim)

cv2.imshow("Rotate Image about the origin (0,0)", result)
cv2.waitKey(0)

# Rotate image
angleInDegrees = 30
angleInRadians = 30 * np.pi / 180.0

cosTheta = np.cos(angleInRadians)
sinTheta = np.sin(angleInRadians)

centerX = im.shape[0] / 2
centerY = im.shape[1] / 2

tx = (1-cosTheta) * centerX - sinTheta * centerY
ty =  sinTheta * centerX  + (1-cosTheta) * centerY

# Rotation matrix
# https://en.wikipedia.org/wiki/Rotation_matrix

warpMat = np.float32(
    [
        [ cosTheta, sinTheta, tx],
        [ -sinTheta,  cosTheta, ty]
    ])

# Warp image
result = cv2.warpAffine(im, warpMat, outDim)

cv2.imshow("Rotate image about a specific point (center)",result)
cv2.waitKey(0)

# Get rotation matrix
rotationMatrix = cv2.getRotationMatrix2D((centerX, centerY), angleInDegrees, 1)

# Warp Image
result = cv2.warpAffine(im, rotationMatrix, outDim)

cv2.imshow("Rotate image the easy way",result)
cv2.waitKey(0)

shearAmount = 0.1

warpMat = np.float32(
    [
        [ 1, shearAmount, 0],
        [ 0, 1.0        , 0]
    ])


# Warp image
result = cv2.warpAffine(im, warpMat, outDim, None, flags=cv2.INTER_LINEAR)

cv2.imshow("Shear Transformation", result)
cv2.waitKey(0)

# Scale
scaleAmount = 1.1
scaleMat = np.float32(
    [
        [ scaleAmount, 0.0,       ],
        [ 0,           scaleAmount]
    ])

# Shear
shearAmount = -0.1
shearMat = np.float32(
    [
        [ 1, shearAmount],
        [ 0, 1.0        ]
    ])

# Rotate by 10 degrees about (0,0)

angleInRadians = 10.0 * np.pi / 180.0

cosTheta = np.cos(angleInRadians)
sinTheta = np.sin(angleInRadians)

rotMat = np.float32(
    [
        [ cosTheta, sinTheta],
        [ -sinTheta, cosTheta]
    ])

translateVector = np.float32(
    [
        [10],
        [0]
    ])

# First scale is applied, followed by shear, followed by rotation.
scaleShearRotate = rotMat @ shearMat @ scaleMat

# Add translation
warpMat = np.append(scaleShearRotate, translateVector, 1)
print(warpMat)
outPts = scaleShearRotate @ np.float32([[50, 50],[50, 149],[149, 50], [149, 149]]).T + translateVector
print(outPts)

# Warp image
result = cv2.warpAffine(im, warpMat, outDim)

cv2.imshow("Complex Transformations", result)
cv2.waitKey(0)

srcPoints = np.float32([[50, 50],[50, 149],[149, 50]])
dstPoints = np.float32([[68, 45],[76, 155],[176, 27]])
estimatedMat = cv2.estimateAffine2D(srcPoints, dstPoints)[0]
print("True warp matrix:\n\n", warpMat)
print("\n\nEstimated warp matrix:\n\n", estimatedMat)

srcPoints = np.float32([[50, 50],[50, 149],[149, 149], [149, 50]])
dstPoints = np.float32([[68, 45],[76, 155],[183, 135], [176, 27]])

estimatedMat = cv2.estimateAffine2D(srcPoints, dstPoints)[0]

print("True warp matrix:\n\n", warpMat)
print("\n\nEstimated warp matrix:\n\n", estimatedMat)

# Warp image
result = cv2.warpAffine(im, estimatedMat, outDim)

cv2.imshow("Complex Transformations using 3-Point Correspondences",result)
cv2.waitKey(0)

# Transformed image
imT = np.zeros((200, 200, 3), dtype = np.float32)
dstPoints = np.float32([[75, 50],[50, 149], [149, 149], [124, 50]])
cv2.fillConvexPoly(imT, np.int32(dstPoints), (1.0, 0.6, 0.2), cv2.LINE_AA);

cv2.imshow("Original Image",im)
cv2.waitKey(0)

cv2.imshow("Transformed Image",imT)
cv2.waitKey(0)

estimatedMat = cv2.estimateAffine2D(srcPoints, dstPoints)[0]
print("\n\nEstimated warp matrix:\n\n", estimatedMat)

# Warp image
imA = cv2.warpAffine(im, estimatedMat, outDim)

cv2.imshow("Transformed Image",imT)
cv2.waitKey(0)

cv2.imshow("Image warped using estimated Affine Transform",imA)
cv2.waitKey(0)

# Calculate homography
h, status = cv2.findHomography(srcPoints, dstPoints)
print(h)

# Warp source image to destination based on homography
imH = cv2.warpPerspective(im, h, outDim)

cv2.imshow("Transformed Image",imT)
cv2.waitKey(0)

cv2.imshow("Image warped using estimated Affine Transform",imA)
cv2.waitKey(0)

cv2.imshow("Image warped using estimated Homography",imH)
cv2.waitKey(0)

cv2.destroyAllWindows()
