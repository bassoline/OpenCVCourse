import cv2
import sys
from dataPath import DATA_PATH
from dataPath import MODEL_PATH

threshold = 0.5
inWidth = 300
inHeight = 300
mean = (127.5, 127.5, 127.5)

modelFile = MODEL_PATH + "ssd_mobilenet_v2_frozen_inference_graph.pb"
configFile = MODEL_PATH + "ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
classFile = MODEL_PATH + "coco_class_labels.txt"
with open(classFile) as fi:
    labels = fi.read().split('\n')

source = 0
if len(sys.argv) > 1:
    source = sys.argv[1]

cap = cv2.VideoCapture(source)
ret, frame = cap.read()

net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
vid_writer = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame.shape[1],frame.shape[0]))

ii = 0
while(1):
    ret, frame = cap.read()
    rows = frame.shape[0]
    cols = frame.shape[1]
    ii+=1
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0/127.5, (inWidth, inHeight), mean, True, False))
    out = net.forward()

    for i in range(out.shape[2]):
        score = float(out[0, 0, i, 2])
        classId = int(out[0, 0, i, 1])

        x1 = int(out[0, 0, i, 3] * cols)
        y1 = int(out[0, 0, i, 4] * rows)
        x2 = int(out[0, 0, i, 5] * cols)
        y2 = int(out[0, 0, i, 6] * rows)

        if score > threshold:
            cv2.putText(frame, "Object : {}, confidence = {:.3f}".format(labels[classId], score), ( x1, y1 - 10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), cv2.FONT_HERSHEY_DUPLEX, 4)

    cv2.imshow("OpenCV Tensorflow Object Detection Demo", frame)
    vid_writer.write(frame)
    k = cv2.waitKey(10)
    if k == 27 :
        break

cv2.destroyAllWindows()
vid_writer.release()
