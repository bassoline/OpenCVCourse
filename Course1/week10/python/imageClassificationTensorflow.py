import cv2
import numpy as np
from dataPath import DATA_PATH
from dataPath import MODEL_PATH

weightFile = MODEL_PATH + "tensorflow_inception_graph.pb"

frame = cv2.imread(DATA_PATH + "images/panda.jpg")

classFile = MODEL_PATH + "imagenet_comp_graph_label_strings.txt"

classes = None
with open(classFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

inHeight = 224
inWidth = 224
swap_rgb = True
mean = [117, 117, 117]
scale = 1.0

# Load a network
net = cv2.dnn.readNetFromTensorflow(weightFile)
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
print(classId)
label = "Predicted Class = {}, Confidence = {:.3f}".format(className, confidence)
print(label)

cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

cv2.imshow("Classification Output", frame)
cv2.imwrite("outTensorflow.jpg", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
