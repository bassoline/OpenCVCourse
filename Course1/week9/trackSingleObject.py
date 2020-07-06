import cv2
import sys
from dataPath import DATA_PATH

if __name__ == '__main__' :

    # Set up tracker.
    # Choose one tracker

    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'CSRT', 'MOSSE']
    tracker_type = tracker_types[int(sys.argv[1])]

    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    elif tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    elif tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    elif tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    elif tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    elif tracker_type == "MOSSE":
        tracker = cv2.TrackerMOSSE_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
          print(t)


    # Read video
    filename = DATA_PATH + "videos/cycle.mp4"
    cap = cv2.VideoCapture(filename)
    video_name = filename.split('/')[-1].split('.')[0]
    video = cv2.VideoCapture(filename)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
    # Define an initial bounding box
    # Cycle
    bbox = (477, 254, 55, 152)

    # ship
    # bbox = (751, 146, 51, 78)

    # Hockey
    # bbox = (129, 47, 74, 85)

    # Face2
    # bbox = (237, 145, 74, 88)

    # meeting
    # bbox = (627, 183, 208, 190)     #CSRT
    # bbox = (652, 187, 118, 123)       #KCF

    # surfing
    # bbox = (97, 329, 118, 293)

    # surf
    # bbox = (548, 587, 52, 87)

    # spinning
    # bbox = (232, 218, 377, 377)       #RED
    # bbox = (699, 208, 383, 391)         #BLUE

    # Car
    # bbox = (71, 457, 254, 188)

    # Uncomment the line below to select a different bounding box
    # bbox = cv2.selectROI(frame, False)
    print("Initial bounding box : {}".format(bbox))
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    out = cv2.VideoWriter('{}_{}_{}.mp4'.format(video_name,tracker_type,bbox),cv2.VideoWriter_fourcc(*'MP4V'), 30, (640,360))

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0,255,0), 2, 1)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (20,120), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2, cv2.LINE_AA)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2, cv2.LINE_AA);

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA);


        # Display result
        cv2.imshow("Tracking", frame)

        outframe = cv2.resize(frame, (640,360))
        out.write(outframe)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
    out.release()
