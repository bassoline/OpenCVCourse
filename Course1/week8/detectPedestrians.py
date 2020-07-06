import os
import glob
import cv2
import numpy as np
from dataPath import DATA_PATH

# Initialize HOG parameters
winSize = (64, 128)
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9
derivAperture = 1
winSigma = -1
histogramNormType = cv2.HOGDESCRIPTOR_L2HYS
L2HysThreshold = 0.2
gammaCorrection = True
nlevels = 64
signedGradient = False

# Initialize HOG
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride,
                        cellSize, nbins, derivAperture,
                        winSigma, histogramNormType, L2HysThreshold,
                        gammaCorrection, nlevels, signedGradient)

# Load model trained by us
model = cv2.ml.SVM_load(DATA_PATH + 'models/pedestrian.yml')

sv = model.getSupportVectors()
rho, aplha, svidx = model.getDecisionFunction(0)
svmDetectorTrained = np.zeros(sv.shape[1] + 1, dtype=sv.dtype)
svmDetectorTrained[:-1] = -sv[:]
svmDetectorTrained[-1] = rho

# set SVMDetector trained by us in HOG
hog.setSVMDetector(svmDetectorTrained)

# OpenCV's HOG based Pedestrian Detector
hogDefault = cv2.HOGDescriptor(winSize, blockSize, blockStride,
                               cellSize, nbins, derivAperture,
                               winSigma, histogramNormType, L2HysThreshold,
                               gammaCorrection, nlevels, signedGradient)
svmDetectorDefault = cv2.HOGDescriptor_getDefaultPeopleDetector()
hogDefault.setSVMDetector(svmDetectorDefault)

# read images from pedestrians directory
imageDir = DATA_PATH + 'images/pedestrians'
imagePaths = glob.glob(os.path.join(imageDir, '*.jpg'))

# We will run pedestrian detector at an fixed height image
finalHeight = 800.0

for imagePath in imagePaths:
  print('processing: {}'.format(imagePath))

  # read image
  im = cv2.imread(imagePath, cv2.IMREAD_COLOR)

  # resize image to height finalHeight
  scale = finalHeight / im.shape[0]
  im = cv2.resize(im, None, fx=scale, fy=scale)

  # detectMultiScale using detector trained by us
  bboxes, weights = hog.detectMultiScale(im, winStride=(8, 8), padding=(32, 32),
                                         scale=1.05, finalThreshold=2,
                                         hitThreshold=1.0)

  # detectMultiScale using default detector
  bboxes2, weights2 = hogDefault.detectMultiScale(im, winStride=(8, 8), padding=(32, 32),
                                                  scale=1.05, finalThreshold=2,
                                                  hitThreshold=0)

  # print pedestrians detected
  if len(bboxes) > 0:
    print('Trained Detector :: pedestrians detected: {}'.format(bboxes.shape[0]))
  if len(bboxes2) > 0:
    print('Default Detector :: pedestrians detected: {}'.format(bboxes2.shape[0]))

  # Draw detected bouunding boxes over image
  # Red = default detector, Green = Trained Detector
  for bbox in bboxes:
    x1, y1, w, h = bbox
    x2, y2 = x1 + w, y1 + h
    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), thickness=3, lineType=cv2.LINE_AA)

  for bbox in bboxes2:
    x1, y1, w, h = bbox
    x2, y2 = x1 + w, y1 + h
    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), thickness=3, lineType=cv2.LINE_AA)

  # Show final result
  cv2.imshow("image",im)
  cv2.waitKey(0)
  # Write image to disk
  imResultPath = os.path.join('results', os.path.basename(imagePath))
  cv2.imwrite(imResultPath, im)


