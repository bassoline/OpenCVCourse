import cv2
import numpy as np
from dataPath import DATA_PATH
from dataPath import MODEL_PATH

protoFile = MODEL_PATH + "bvlc_googlenet.prototxt"
weightFile = MODEL_PATH + "bvlc_googlenet.caffemodel"

frame = cv2.imread(DATA_PATH + "images/panda.jpg")

classFile = MODEL_PATH + "classification_classes_ILSVRC2012.txt"

classes = None
with open(classFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

inHeight = 224
inWidth = 224
swap_rgb = False
mean = [104, 117, 123]
scale = 1.0

# Load a network
net = cv2.dnn.readNetFromCaffe(protoFile, weightFile)
# Create a 4D blob from a frame.
blob = cv2.dnn.blobFromImage(frame, scale, (inWidth, inHeight), mean, swap_rgb, crop=False)

# Run a model
net.setInput(blob)
out = net.forward()

# Get a class with a highest score.
out = out.flatten()
classId = np.argmax(out)
className = classes[classId]
confidence = out[classId]
label = "Predicted Class = {}, Confidence = {:.3f}".format(className, confidence)
print(label)

cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

cv2.imshow("Classification Output", frame)
cv2.imwrite("outCaffe.jpg", frame)
cv2.waitKey(0)
