import cv2

cap = cv2.VideoCapture(0)
k = 0

while(True):
	ret, frame = cap.read()
	if k == 27: # esc key
		break
	if k == 101 or k == 69:
		cv2.putText(frame, 'E is pressed, press Z or ESC', 
			(100, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
	if k == 90 or k == 122:
		cv2.putText(frame, 'Z is pressed', (100, 180), 
			cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0 , 0), 1)
	cv2.imshow('Image', frame)
	k = cv2.waitKey(10000) & 0xFF

cap.release()
cv2.destroyAllWindows()