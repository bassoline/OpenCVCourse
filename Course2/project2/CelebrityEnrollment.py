import os, random, glob
import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import _pickle as cPickle

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

# Root folder of the dataset
faceDatasetFolder = "./asnlib/publicdata/celeb_mini"

imagePaths = []
# read subfolders in folder "celeb_mini" to list
for x in os.listdir(faceDatasetFolder):
    xpath = os.path.join(faceDatasetFolder, x)
    if os.path.isdir(xpath):
        subfolder = xpath
        for x in os.listdir(subfolder):
            xpath = os.path.join(subfolder, x)
            if x.endswith("JPEG"):
                imagePaths.append(xpath)

# Enroll celebrity
# We will store face descriptors in an ndarray (faceDescriptors)
# and their image paths in a dictionary (index)
index = {}
i = 0
faceDescriptors = None
for imagePath in imagePaths:
    print("processing: {}".format(imagePath))
    # read image and convert it to RGB
    img = cv2.imread(imagePath)

    # detect faces in image
    faces = faceDetector(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    print("{} Face(s) found".format(len(faces)))
    # Now process each face we found
    for k, face in enumerate(faces):

        # Find facial landmarks for each detected face
        shape = shapePredictor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), face)

        # convert landmarks from Dlib's format to list of (x, y) points
        landmarks = [(p.x, p.y) for p in shape.parts()]

        # Compute face descriptor using neural network defined in Dlib.
        # It is a 128D vector that describes the face in img identified by shape.
        faceDescriptor = faceRecognizer.compute_face_descriptor(img, shape)

        # Convert face descriptor from Dlib's format to list, then a NumPy array
        faceDescriptorList = [x for x in faceDescriptor]
        faceDescriptorNdarray = np.asarray(faceDescriptorList, dtype=np.float64)
        faceDescriptorNdarray = faceDescriptorNdarray[np.newaxis, :]

        # Stack face descriptors (1x128) for each face in images, as rows
        if faceDescriptors is None:
            faceDescriptors = faceDescriptorNdarray
        else:
            faceDescriptors = np.concatenate(
                (faceDescriptors, faceDescriptorNdarray), axis=0
            )

        # save the image path for this face to reference later
        index[i] = imagePath
        i += 1

# Write descriptors and index to disk
np.save("descriptors.npy", faceDescriptors)
# index has image paths in same order as descriptors in faceDescriptors
with open("index.pkl", "wb") as f:
    cPickle.dump(index, f)
