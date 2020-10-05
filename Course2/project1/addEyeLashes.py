import cv2
import matplotlib.pyplot as plt
import numpy as np
import dlib

# test
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
L_EYELASH_CENTER = (526, 284)

R_EYELASH_LM = [
    (1, 336),
    (182, 182),
    (385, 187),
    (478, 262),
    (400, 362),
    (200, 375),
]
R_EYELASH_CENTER = (274, 284)

# eye lash constants
EYE_WIDTH = 477
LEFT_EYE_LASH_PATH = "./eyelash_left.jpg"
RIGHT_EYE_LASH_PATH = "./eyelash_right.jpg"

# Dlib shape predictor model path
MODEL_PATH = "../week2/data/shape_predictor_68_face_landmarks.dat"

# create mask for image with no background
def createMask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray.copy()
    foregroundPixels = mask < 100
    backgroundPixels = mask >= 100
    mask[foregroundPixels] = 0
    mask[backgroundPixels] = 1
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


# Returns 8 points on the boundary of a rectangle
def getEightBoundaryPoints(h, w):
    boundaryPts = []
    boundaryPts.append((0, 0))
    boundaryPts.append((w / 2, 0))
    boundaryPts.append((w - 1, 0))
    boundaryPts.append((w - 1, h / 2))
    boundaryPts.append((w - 1, h - 1))
    boundaryPts.append((w / 2, h - 1))
    boundaryPts.append((0, h - 1))
    boundaryPts.append((0, h / 2))
    return np.array(boundaryPts, dtype=np.float)


# this scales the predefined landmarks, and then shifts them according to where they
# would be inserted into the image
# left_start_pixels is the top left corner of where the eye lash image would be layed over
# the real image - same goes for right_start_pixels
def getEyeLashFeatures(left_start_pixels, right_start_pixels, scale):
    left_lm_scaled = np.array(
        [
            (lm[0] * scale + left_start_pixels[0], lm[1] * scale + left_start_pixels[1])
            for lm in L_EYELASH_LM
        ]
    )
    right_lm_scaled = np.array(
        [
            (
                lm[0] * scale + right_start_pixels[0],
                lm[1] * scale + right_start_pixels[1],
            )
            for lm in R_EYELASH_LM
        ]
    )
    return np.concatenate((left_lm_scaled, right_lm_scaled))


def getRelevantImageFeatures(image, feature_points):
    feature_indexes = np.concatenate((L_EYE_LM, R_EYE_LM))
    relevant_features = []
    for index in feature_indexes:
        p = feature_points[index]
        relevant_features.append((p.x, p.y))
    return relevant_features


# create image with left & right eyelashes at same height, size and spacing
# as eyes in face picture (orientation doesn't matter, we'll take care of that with the delanuy trianglulation)
def createEyeLashImage(dlib_model_path, image):
    # get facial landmarks
    facial_landmarks = getFacialLandmarks(dlib_model_path, image).parts()
    # pinpoint eye landmark positions in the image
    l_center_x, l_center_y, l_width = getEyeCenterAndWidth(facial_landmarks, L_EYE_LM)
    r_center_x, r_center_y, r_width = getEyeCenterAndWidth(facial_landmarks, R_EYE_LM)
    # reduce eyelashes to be similar size as eye size
    scale = ((l_width + r_width) / 2) / EYE_WIDTH
    # get and scale eye lash images
    l_eye_lash_image = cv2.imread(LEFT_EYE_LASH_PATH)
    r_eye_lash_image = cv2.imread(RIGHT_EYE_LASH_PATH)
    l_eye_lash_image = cv2.resize(
        l_eye_lash_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
    )
    r_eye_lash_image = cv2.resize(
        r_eye_lash_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
    )
    # create blank image with the same dimensions as the original image
    eye_lash_image = np.ones(image.shape, dtype=np.uint8) * 255
    # grab new centers of eye lash and image shape of eyelash
    r_center_eyelash = (R_EYELASH_CENTER[0] * scale, R_EYELASH_CENTER[1] * scale)
    l_center_eyelash = (L_EYELASH_CENTER[0] * scale, L_EYELASH_CENTER[1] * scale)
    eyelash_w, eyelash_h, n = r_eye_lash_image.shape
    # determine start pixels to replace (top left corner)
    left_pixels_to_replace = (
        l_center_x - l_center_eyelash[0],
        l_center_y - l_center_eyelash[1],
    )
    right_pixels_to_replace = (
        r_center_x - r_center_eyelash[0],
        r_center_y - r_center_eyelash[1],
    )
    # update image to include eye_lashes in approximate locations (swapping 0 & 1 order since it goes row, col)
    eye_lash_image[
        int(right_pixels_to_replace[1]) : int(right_pixels_to_replace[1] + eyelash_w),
        int(right_pixels_to_replace[0]) : int(right_pixels_to_replace[0] + eyelash_h),
    ] = r_eye_lash_image
    eye_lash_image[
        int(left_pixels_to_replace[1]) : int(left_pixels_to_replace[1] + eyelash_w),
        int(left_pixels_to_replace[0]) : int(left_pixels_to_replace[0] + eyelash_h),
    ] = l_eye_lash_image

    # create mask for eyelashes
    eye_lash_mask = createMask(eye_lash_image)

    # get the boundary points for the images
    r, c, _ = image.shape
    boundary_points = getEightBoundaryPoints(r, c)

    # grab new eye_lash feature locations
    eye_lash_features = np.concatenate(
        (
            getEyeLashFeatures(left_pixels_to_replace, right_pixels_to_replace, scale),
            boundary_points,
        )
    )
    # create feature array of only the eye features we care about from the image
    image_features = np.concatenate(
        (getRelevantImageFeatures(image, facial_landmarks), boundary_points)
    )

    # return eyelash image, and mask of eyelashes, and feature locations of eye_lash_image, and image
    return eye_lash_image, eye_lash_mask, eye_lash_features, image_features


# returns an averaged center of the eye, and the approx width of the eye
def getEyeCenterAndWidth(facial_landmarks, eye_landmarks):
    landmarks = []
    center_x = 0
    center_y = 0
    for i in eye_landmarks:
        lm = facial_landmarks[i]
        landmarks.append((lm.x, lm.y))
        center_x += lm.x
        center_y += lm.y
    center_x /= len(landmarks)
    center_y /= len(landmarks)
    width = landmarks[3][0] - landmarks[0][0]
    return (center_x, center_y, width)


# Check if a point is inside a rectangle
def rectContains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):

    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(
        src,
        warpMat,
        (size[0], size[1]),
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1] : r1[1] + r1[3], r1[0] : r1[0] + r1[2]]

    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)

    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] = img2[
        r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]
    ] * ((1.0, 1.0, 1.0) - mask)
    img2[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] = (
        img2[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] + img2Rect
    )


# Calculate Delaunay triangles for set of points
# Returns the vector of indices of 3 points for each triangle
def calculateDelaunayTriangles(rect, points):

    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D(rect)

    # Insert points into subdiv
    for p in points:
        subdiv.insert((p[0], p[1]))

    # Get Delaunay triangulation
    triangleList = subdiv.getTriangleList()

    # Find the indices of triangles in the points array
    delaunayTri = []

    for t in triangleList:
        # The triangle returned by getTriangleList is
        # a list of 6 coordinates of the 3 points in
        # x1, y1, x2, y2, x3, y3 format.
        # Store triangle as a list of three points
        pt = []
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if (
            rectContains(rect, pt1)
            and rectContains(rect, pt2)
            and rectContains(rect, pt3)
        ):
            # Variable to store a triangle as indices from list of points
            ind = []
            # Find the index of each vertex in the points list
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if (
                        abs(pt[j][0] - points[k][0]) < 1.0
                        and abs(pt[j][1] - points[k][1]) < 1.0
                    ):
                        ind.append(k)
                # Store triangulation as a list of indices
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))

    return delaunayTri


# use delanuy transformation to align eyelashes with face
def delanuyTransformationEyeLashes(
    image, eyeLash_image, eyeLash_alpha_mask, eye_featurePoints, eyeLash_featurePoints
):
    # Find delanauy traingulation for convex hull points
    sizeImg1 = image.shape
    rect = (0, 0, sizeImg1[1], sizeImg1[0])
    dt = calculateDelaunayTriangles(rect, eye_featurePoints)
    eyeLash_alpha_mask = cv2.merge(
        (eye_lash_mask, eyeLash_alpha_mask, eyeLash_alpha_mask)
    )
    eyeLash_warped = np.zeros(eyeLash_image.shape)
    eyeLash_alpha_warped = np.zeros(eyeLash_image.shape)

    # Apply affine transformation to Delaunay triangles
    for i in range(0, len(dt)):
        t1 = []
        t2 = []

        # get points for img1, img2 corresponding to the triangles
        for j in range(0, 3):
            t1.append(eye_featurePoints[dt[i][j]])
            t2.append(eyeLash_featurePoints[dt[i][j]])

        warpTriangle(eyeLash_image, eyeLash_warped, t1, t2)
        warpTriangle(eyeLash_alpha_mask, eyeLash_alpha_warped, t1, t2)

    return eyeLash_warped, eyeLash_alpha_warped


# alpha blend images
def alphaBlendImages(image, eyeLashes, mask):
    mask_shape = mask.shape
    mask = cv2.merge((mask, mask, mask)) if len(mask_shape) == 2 else mask
    image_mask = cv2.multiply(image / 1.0, mask / 1.0)
    eyeLash_mask = cv2.multiply(eyeLashes / 1.0, (1.0 - mask))
    final_image = image_mask + eyeLash_mask
    return np.uint8(final_image)


image = cv2.imread("./blue_eyes.jpg")
# create the eye lash image and return the feature points of the destination image and the eyelash image
(
    eye_lash_image,
    eye_lash_mask,
    eye_featurePoints,
    eye_lash_featurePoints,
) = createEyeLashImage(MODEL_PATH, image)
# alpha blend the images without delaunay triangulation
simple_blend = alphaBlendImages(image, eye_lash_image, eye_lash_mask)
# use delaunay triangulation to transform the eye_lashes to better fit the face
transformed_eye_lash_image, transformed_eye_lash_mask = delanuyTransformationEyeLashes(
    image, eye_lash_image, eye_lash_mask, eye_featurePoints, eye_lash_featurePoints
)
# alpha blend the transformed image
transformed_image = alphaBlendImages(
    image, transformed_eye_lash_image, transformed_eye_lash_mask
)

plt.figure()
# with delaunay transformation
plt.subplot(141)
plt.imshow(transformed_image[:, :, ::-1])
# naive layering
plt.subplot(142)
plt.imshow(simple_blend[:, :, ::-1])
# original image
plt.subplot(143)
plt.imshow(image[:, :, ::-1])
# just the eye lashes
plt.subplot(144)
plt.imshow(eye_lash_mask)
