import glob
import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib

plt.ion()
matplotlib.rcParams["figure.figsize"] = (6.0, 6.0)
matplotlib.rcParams["image.cmap"] = "gray"
matplotlib.rcParams["image.interpolation"] = "bilinear"

# Path to landmarks and face recognition model files
PREDICTOR_PATH = "../lib/publicdata/models/shape_predictor_68_face_landmarks.dat"
FACE_RECOGNITION_MODEL_PATH = (
    "../lib/publicdata/models/dlib_face_recognition_resnet_model_v1.dat"
)

# Initialize face detector, facial landmarks detector
# and face recognizer
faceDetector = dlib.get_frontal_face_detector()
shapePredictor = dlib.shape_predictor(PREDICTOR_PATH)
faceRecognizer = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)

# Label -> celeb name
labelMap = np.load("./asnlib/publicdata/celeb_mapping.npy", allow_pickle=True).item()

# Testing (have it lower since they're not really the same person)
THRESHOLD = 0.6
# load descriptors and index file generated during enrollment
index = np.load("index.pkl", allow_pickle=True)
faceDescriptorsEnrolled = np.load("descriptors.npy")

# Display results
# read image
testImages = glob.glob("./asnlib/publicdata/test-images/*.jpg")
i = 0
for test in testImages:
    im = cv2.imread(test)
    imDlib = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    t = time.time()

    # detect faces in image
    faces = faceDetector(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

    # Now process each face we found
    # (in this example we are only using test images with a single face)
    for face in faces:

        # Find facial landmarks for each detected face
        shape = shapePredictor(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), face)

        # find coordinates of face rectangle
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        # Compute face descriptor using neural network defined in Dlib
        # using facial landmark shape
        faceDescriptor = faceRecognizer.compute_face_descriptor(im, shape)

        # Convert face descriptor from Dlib's format to list, then a NumPy array
        faceDescriptorList = [m for m in faceDescriptor]
        faceDescriptorNdarray = np.asarray(faceDescriptorList, dtype=np.float64)
        faceDescriptorNdarray = faceDescriptorNdarray[np.newaxis, :]

        # Calculate Euclidean distances between face descriptor calculated on face dectected
        # in current frame with all the face descriptors we calculated while enrolling faces
        distances = np.linalg.norm(
            faceDescriptorsEnrolled - faceDescriptorNdarray, axis=1
        )
        # Calculate minimum distance and index of this face
        argmin = np.argmin(distances)  # index
        minDistance = distances[argmin]  # minimum distance
        print(minDistance)
        # Dlib specifies that in general, if two face descriptor vectors have a Euclidean
        # distance between them less than 0.6 then they are from the same
        # person, otherwise they are from different people.

        # This threshold will vary depending upon number of images enrolled
        # and various variations (illuminaton, camera quality) between
        # enrolled images and query image
        # We are using a threshold of 0.6 since we're not looking for the same person
        # only look alikes

        # If minimum distance if less than threshold
        # find the name of person from index
        # else the person in query image is unknown
        if minDistance <= THRESHOLD:
            # Get full path of each image file
            fullPath = index[argmin]
            folderName = fullPath.split("celeb_mini/")[-1].split("/")[0]
            celeb_name = labelMap[folderName]
            im = cv2.imread(fullPath)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        else:
            celeb_name = "unknown"
            im = np.zeros_like(imDlib.shape)

        print("time taken = {:.3f} seconds".format(time.time() - t))

        plt.figure(i)
        label = "test image {}".format(i)
        plt.subplot(121, label=label)
        plt.imshow(imDlib)
        plt.title("test img")

        label = "celeb image {}".format(i)
        plt.subplot(122, label=label)
        plt.title("celeb doppleganger: {}".format(celeb_name))
        plt.imshow(im)
        i += 1
