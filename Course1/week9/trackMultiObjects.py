import sys
import cv2
from random import randint
from dataPath import DATA_PATH

trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

def createTrackerByName(trackerType):
  # Create a tracker based on tracker name
  if trackerType == trackerTypes[0]:
    tracker = cv2.TrackerBoosting_create()
  elif trackerType == trackerTypes[1]:
    tracker = cv2.TrackerMIL_create()
  elif trackerType == trackerTypes[2]:
    tracker = cv2.TrackerKCF_create()
  elif trackerType == trackerTypes[3]:
    tracker = cv2.TrackerTLD_create()
  elif trackerType == trackerTypes[4]:
    tracker = cv2.TrackerMedianFlow_create()
  elif trackerType == trackerTypes[5]:
    tracker = cv2.TrackerGOTURN_create()
  elif trackerType == trackerTypes[6]:
    tracker = cv2.TrackerMOSSE_create()
  elif trackerType == trackerTypes[7]:
    tracker = cv2.TrackerCSRT_create()
  else:
    tracker = None
    print('Incorrect tracker name')
    print('Available trackers are:')
    for t in trackerTypes:
      print(t)

  return tracker

if __name__ == '__main__':

  print("Default tracking algoritm is CSRT \n"
        "Available tracking algorithms are:\n")
  for t in trackerTypes:
      print(t)

  trackerType = "CSRT"

  # Set video to load
  filename = DATA_PATH + "videos/cycle.mp4"

  # Create a video capture object to read videos
  cap = cv2.VideoCapture(filename)
  video_name = filename.split('/')[-1].split('.')[0]
  out = cv2.VideoWriter('{}_{}.mp4'.format(video_name,trackerType),cv2.VideoWriter_fourcc(*'MP4V'), 30, (640,360))

  # Read first frame
  success, frame = cap.read()
  # quit if unable to read the video file
  if not success:
    print('Failed to read video')
    sys.exit(1)

  ## Select boxes
  bboxes = []
  ## Select boxes
  colors = []
  for i in range(3):
    # Select some random colors
    colors.append((randint(64, 255), randint(64, 255),
                randint(64, 255)))
  # Select the bounding boxes
  bboxes = [(471, 250, 66, 159), (349, 232, 69, 102)]
  # print('Selected bounding boxes {}'.format(bboxes))

  # You can also select bounding boxes. Just uncomment the following code snippet
  ## OpenCV's selectROI function doesn't work for selecting multiple objects in Python
  ## So we will call this function in a loop till we are done selecting all objects

  #====================UNCOMMENT THIS FOR SELECTING BOUNDING BOX=============#
  # while True:
  #   # draw bounding boxes over objects
  #   # selectROI's default behaviour is to draw box starting from the center
  #   # when fromCenter is set to false, you can draw box starting from top left corner
  #
  #   bbox = cv2.selectROI('MultiTracker', frame)
  #   bboxes.append(bbox)
  #   colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
  #   print("Press q to quit selecting boxes and start tracking")
  #   print("Press any other key to select next object")
  #   k = cv2.waitKey(0) & 0xFF
  #   if (k == 113):  # q is pressed
  #     break
  #
  # print('Selected bounding boxes {}'.format(bboxes))
  #====================UNCOMMENT THIS FOR SELECTING BOUNDING BOX=============#

  ## Initialize MultiTracker
  # There are two ways you can initialize multitracker
  # 1. tracker = cv2.MultiTracker("CSRT")
  # All the trackers added to this multitracker
  # will use CSRT algorithm as default
  # 2. tracker = cv2.MultiTracker()
  # No default algorithm specified

  # Initialize MultiTracker with tracking algo
  # Specify tracker type

  # Create MultiTracker object
  multiTracker = cv2.MultiTracker_create()

  # Initialize MultiTracker
  for bbox in bboxes:
    multiTracker.add(createTrackerByName(trackerType), frame, bbox)


  # Process video and track objects
  while cap.isOpened():
    success, frame = cap.read()
    if not success:
      break

    # get updated location of objects in subsequent frames
    success, boxes = multiTracker.update(frame)

    # draw tracked objects
    for i, newbox in enumerate(boxes):
      p1 = (int(newbox[0]), int(newbox[1]))
      p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
      cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

    # show frame
    cv2.imshow('MultiTracker', frame)
    outframe = cv2.resize(frame, (640,360))
    out.write(outframe)


    # quit on ESC button
    if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
      break
  out.release()
