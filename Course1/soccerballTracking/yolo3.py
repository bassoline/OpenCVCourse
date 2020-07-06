import numpy as np
import time 
import cv2

# using weights and configs from 
# https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
class YoloV3Detector:
    def __init__(self):
        self.min_confidence = 0.5
        self.threshold = 0.3
        self.scale_factor = 1/255.0
        self.image_size = (416,416)
        self.detection_color = (0,255,0)
        self.tracking_color = (255,0,0)
        self.label_path = 'yolo/coco.names'
        self.weights_path = 'yolo/yolov3.weights'
        self.config_path = 'yolo/yolov3.cfg'
        self.labels = open(self.label_path).read().strip().split("\n")
        self.yolo_detector = cv2.dnn.readNetFromDarknet(self.config_path,
                self.weights_path)
        self.layer_names = self.yolo_detector.getLayerNames()
        # yolo3 has three different output layers - we can use all three 
        self.output_layers = self.yolo_detector.getUnconnectedOutLayers()
        # these are all the yolo_layers (-1 bc i don't think the layers are returned in
        # 0 index fashion)
        self.output_layer_names = [self.layer_names[i[0]-1] for i in
                self.output_layers]
        self.image_shape = None
        self.outputs = None
        self.boxes = []
        self.confidences = []
        self.classIDs = []
        self.COLORS = None
        self.image = None

    def createBlobAndForwardPass(self, image):
        # create blob from image and perform a forward pass 
        # blob performs mean subtraction & scaling for us see here 
        # https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
        self.image = image
        self.image_shape = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, self.scale_factor, self.image_size, swapRB=True,
                crop=False)
        self.yolo_detector.setInput(blob)
        self.outputs = self.yolo_detector.forward(self.output_layer_names)
        
    def createBoundingBoxes(self):
        # get image dimensions
        (h, w) = self.image_shape

        # reset bins
        self.boxes = []
        self.confidences = []
        self.classIDs = []

        for output in self.outputs:
            for detection in output:
                # each row is a candidate detection, the 1st 4 numbers are
                # [center_x, center_y, width, height], followed by (N-4) class
                # probabilities/confidence
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions
                if confidence < self.min_confidence:
                    continue
                
                # for this assignment, only care about sports ball label
                if self.labels[classID] != 'sports ball':
                    continue

                # scale bounding box to size of image
                (center_x, center_y, width, height) = (detection[0:4] * np.array([w, h,
                    w, h])).astype('int')

                # grab the corners of the box
                left = int(center_x - width/2)
                top = int(center_y - height/2)

                # add new values to their arrays
                self.boxes.append([left, top, int(width), int(height)])
                self.confidences.append(float(confidence))
                self.classIDs.append(classID)


    def applyNonMaxSuppression(self):
        # apply non-maxima suppression to overlapping bounding boxes
        self.final_detections = cv2.dnn.NMSBoxes(self.boxes, self.confidences,
                self.min_confidence, self.threshold)

    
    def drawBoundingBoxesWithLabels(self):
        if len(self.final_detections) > 0:
            boxes = self.boxes
            for d in self.final_detections.flatten():
                # draw bounding box
                color = [int(c) for c in self.COLORS[self.classIDs[d]]]
                self.drawBoundingBox(self.image, boxes[d], color)
                text = "{}: {:.4f}".format(self.labels[self.classIDs[d]], self.confidences[d])
                cv2.putText(self.image, text, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                    1)
    

    def drawBoundingBox(self, image, bbox, color=None):
        color = color if color != None else self.detection_color
        print(bbox)
        (x, y) = (int(bbox[0]), int(bbox[1]))
        (w, h) = (int(bbox[2]), int(bbox[3]))
        # draw bounding box
        cv2.rectangle(image, (x,y), (x+w, y+h), color, 2)


    def returnSingleBoundingBox(self):
         if len(self.final_detections) > 0:
            boxes = self.boxes
            # this is based on the assumption that there is only a single sports
            # ball in the image and we care about nothing else 
            singleBox = self.final_detections.flatten()[0]
            print('singleBox', boxes[singleBox])
            return tuple(boxes[singleBox])
         else:
            # could not find the ball
            return None
        
    
    def showImage(self):
        cv2.imshow("image", self.image)
        # cv2.waitKey(0)
        cv2.waitKey(25)
    

    def showImageWithAnnotations(self, image, bbox, color, annotation1,
            annotation2):
        if bbox != None:
            self.drawBoundingBox(image, bbox, color)
            # Display tracker type on frame
            cv2.putText(image, annotation1, (20,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2);

            # Display FPS on frame
            cv2.putText(image, annotation2, (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2);
        else:
            # Tracking failure
            cv2.putText(image, "Tracking failure detected", (20,80),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        
        cv2.imshow("video", image)



    def createColors(self):
        np.random.seed(42) # keeps colors the same everytime
        self.COLORS = np.random.randint(0, 255, size=(len(self.labels), 3), dtype='uint8')


    def detectAndShow(self, image):
       self.createColors()
       self.createBlobAndForwardPass(image)
       self.createBoundingBoxes()
       self.applyNonMaxSuppression()
       self.drawBoundingBoxesWithLabels()
       self.showImage()


    def detectAndReturnBoundingBox(self, image):
       self.createColors()
       self.createBlobAndForwardPass(image)
       self.createBoundingBoxes()
       self.applyNonMaxSuppression()
       return self.returnSingleBoundingBox()
       

# load image for testing
image_path = './images/soccer.jpg'
image = cv2.imread(image_path)

# load video
videoPath = './videos/soccer_game.mp4'
video = cv2.VideoCapture(videoPath)

# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")

# Read first frame.
ok, frame = video.read()
if not ok:
    print('Cannot read video file')

# initalize yolov3 detector 
yolov3 = YoloV3Detector()
bbox = yolov3.detectAndReturnBoundingBox(frame)
        
count = 0
ok = True
fps = 0

while True:
    okay, frame = video.read()
    if not okay:
        break
    
    # Start timer
    timer = cv2.getTickCount()

    if count % 20 == 0 or not ok:       
        method = 'detection'
        color = yolov3.detection_color
        #detection takes some time, so im showing that we're detecting here too
        yolov3.showImageWithAnnotations(frame, bbox, color, method, "FPS : " +
                    str(int(fps))) 
        # detect sports ball
        bbox = yolov3.detectAndReturnBoundingBox(frame)
        # create KCF tracker
        tracker = cv2.TrackerKCF_create()
        tracker.init(frame, bbox)
        ok = bbox != None
    else:
        # Update tracker
        ok, bbox = tracker.update(frame)
        method = 'tracking'
        color = yolov3.tracking_color

    # Calculate processing time and display results.
    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    yolov3.showImageWithAnnotations(frame, bbox, color, method, "FPS : " +
            str(int(fps)))
    
    cv2.waitKey(1)
    count += 1
        
