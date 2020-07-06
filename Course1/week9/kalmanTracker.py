import os
import sys
import cv2
import random
import numpy as np
from dataPath import DATA_PATH

# To detect max face area in multiple array of faces(x,y,w,h)
def maxRectArea(rects):
  area = 0
  maxRect = rects[0].copy()
  for rect in rects:
    x, y, w, h = rect.ravel()
    if w*h > area:
      area = w*h
      maxRect = rect.copy()
  maxRect = maxRect[:, np.newaxis]
  return maxRect

blue = (255, 0, 0)
red = (0, 0, 255)

if __name__ == '__main__' :

  # Initialize hog descriptor for people detection
  winSize = (64, 128)
  blockSize = (16, 16)
  blockStride = (8, 8)
  cellSize = (8, 8)
  nbins = 9
  derivAperture = 1
  winSigma = -1
  histogramNormType = 0
  L2HysThreshold = 0.2
  gammaCorrection = True
  nlevels = 64
  signedGradient = False

  # Initialize HOG
  hog = cv2.HOGDescriptor(winSize, blockSize, blockStride,
                          cellSize, nbins, derivAperture,
                          winSigma, histogramNormType, L2HysThreshold,
                          gammaCorrection, nlevels, signedGradient)

  svmDetector = cv2.HOGDescriptor_getDefaultPeopleDetector()
  hog.setSVMDetector(svmDetector)

  #  Load video
  filename = DATA_PATH + "videos/boy-walking.mp4"
  cap = cv2.VideoCapture(filename)

  # Confirm video is open
  if not cap.isOpened():
    print("Unable to read video")
    sys.exit(1)

  # Variable for storing frames
  frameDisplay = []
  # Initialize Kalman filter.
  # Internal state has 6 elements (x, y, width, vx, vy, vw)
  # Measurement has 3 elements (x, y, width ).
  # Note: Height = 2 x width, so it is not part of the state
  # or measurement.
  KF = cv2.KalmanFilter(6, 3, 0)

  # Transition matrix is of the form
  # [
  #   1, 0, 0, dt, 0,  0,
  #   0, 1, 0, 0,  dt, 0,
  #   0, 0, 1, 0,  0,  dt,
  #   0, 0, 0, 1,  0,  0,
  #   0, 0, 0, 0,  1,  0,
  #   0, 0, 0, dt, 0,  1
  # ]
  # because
  # x = x + vx * dt
  # y = y + vy * dt
  # w = y + vw * dt

  # vx = vx
  # vy = vy
  # vw = vw
  KF.transitionMatrix = cv2.setIdentity(KF.transitionMatrix)

  # Measurement matrix is of the form
  # [
  #  1, 0, 0, 0, 0,  0,
  #  0, 1, 0, 0, 0,  0,
  #  0, 0, 1, 0, 0,  0,
  # ]
  # because we are detecting only x, y and w.
  # These quantities are updated.
  KF.measurementMatrix = cv2.setIdentity(KF.measurementMatrix)

  # Variable to store detected x, y and w
  measurement = np.zeros((3, 1), dtype=np.float32)
  # Variables to store detected object and tracked object
  objectTracked = np.zeros((4, 1), dtype=np.float32)
  objectDetected = np.zeros((4, 1), dtype=np.float32)

  # Variables to store results of the predict and update (a.k.a correct step).
  updatedMeasurement = np.zeros((3, 1), dtype=np.float32)
  predictedMeasurement = np.zeros((6, 1), dtype=np.float32)

  # Variable to indicate measurement was updated
  measurementWasUpdated = False

  # Timing variable
  ticks = 0
  preTicks = 0

  # Read frames until object is detected for the first time
  success = True
  while success:
    sucess, frame = cap.read()
    objects, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(32, 32),
                                            scale=1.05, hitThreshold=0, finalThreshold=1,
                                            useMeanshiftGrouping=False)

    # Update timer
    ticks = cv2.getTickCount()

    if len(objects) > 0:
      # Copying max face area values to Kalman Filter
      objectDetected = maxRectArea(objects)
      measurement = objectDetected[:3].astype(np.float32)

      # Update state. Note x, y, w are set to measured values.
      # vx = vy = vw because we have no idea about the velocities yet.
      KF.statePost[0:3, 0] = measurement[:, 0]
      KF.statePost[3:6] = 0.0

      # Set diagonal values for covariance matrices.
      # processNoiseCov is Q
      KF.processNoiseCov = cv2.setIdentity(KF.processNoiseCov, (1e-2))
      KF.measurementNoiseCov = cv2.setIdentity(KF.measurementNoiseCov, (1e-2))
      break

  # dt for Transition matrix
  dt = 0.0
  # Random number generator for randomly selecting frames for update
  random.seed(42)

  # Loop over rest of the frames
  while True:
    success, frame = cap.read()
    if not success:
      break

    # Variable for displaying tracking result
    frameDisplay = frame.copy()
    # Variable for displaying detection result
    frameDisplayDetection = frame.copy()

    # Update dt for transition matrix.
    # dt = time elapsed.
    preTicks = ticks;
    ticks = cv2.getTickCount()
    dt = (ticks - preTicks) / cv2.getTickFrequency()

    KF.transitionMatrix[0, 3] = dt
    KF.transitionMatrix[1, 4] = dt
    KF.transitionMatrix[2, 5] = dt

    predictedMeasurement = KF.predict()

    # Detect objects in current frame
    objects, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(32, 32),
                                            scale=1.05, hitThreshold=0, finalThreshold=1,
                                            useMeanshiftGrouping=False)
    if len(objects) > 0:
      # Find largest object
      objectDetected = maxRectArea(objects)

      # Display detected rectangle
      x1, y1, w1, h1 = objectDetected.ravel()
      cv2.rectangle(frameDisplayDetection, (x1, y1), (x1+w1, y1+h1), red, 2, 4)

    # We will update measurements 15% of the time.
    # Frames are randomly chosen.
    update = random.randint(0, 100) < 15

    if update:
      # Kalman filter update step
      if len(objects) > 0:
        # Copy x, y, w from the detected rectangle
        measurement = objectDetected[0:3].astype(np.float32)

        # Perform Kalman update step
        updatedMeasurement = KF.correct(measurement)
        measurementWasUpdated = True
      else:
        # Measurement not updated because no object detected
        measurementWasUpdated = False
    else:
      # Measurement not updated
      measurementWasUpdated = False

    if measurementWasUpdated:
      # Use updated measurement if measurement was updated
      objectTracked[0:3, 0] = updatedMeasurement[0:3, 0].astype(np.int32)
      objectTracked[3, 0] = 2*updatedMeasurement[2, 0].astype(np.int32)
    else:
      # If measurement was not updated, use predicted values.
      objectTracked[0:3, 0] = predictedMeasurement[0:3, 0].astype(np.int32)
      objectTracked[3, 0] = 2*predictedMeasurement[2, 0].astype(np.int32)

    # Draw tracked object
    x2, y2, w2, h2 = objectTracked.ravel()
    cv2.rectangle(frameDisplay, (x2, y2), (x2+w2, y2+h2), blue, 2, 4)

    # Text indicating Tracking or Detection.
    cv2.putText(frameDisplay, "Tracking", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, blue, 2)
    cv2.putText(frameDisplayDetection, "Detection", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, red, 2)

    # Concatenate detected result and tracked result vertically
    output = np.concatenate((frameDisplayDetection, frameDisplay), axis=0)

    # Display result.
    cv2.imshow("object tracker", output)

    # Break if ESC pressed
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()
cv2.destroyAllWindows()
