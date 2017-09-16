


from collections import deque 
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt

orangelower = (27,140,250)
orangeupper = (40,255,255)
pts = deque(maxlen = 128)

camera = cv2.VideoCapture("/Users/Rishav/Desktop/IMG_1034.mp4")
#camera = cv2.VideoCapture(0)
plotlist = []
pastRad = 10000000


while True:
	(grabbed,frame) = camera.read()
	if not grabbed: break

	frame = imutils.resize(frame,width = 400)
	#frame = imutils.rotate(frame,270)
	hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(gray,127,255,1)
	contours,h = cv2.findContours(thresh,1,2)

	for cnt in contours:
	    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
	    if len(approx)==4:
	    	print(cnt)
	    	print(cnt[1])
	    	print(cnt[2])
	        cv2.drawContours(frame,[cnt],0,(255,0,255),-1)


	mask1 = cv2.inRange(hsv,orangelower,orangeupper)
	mask = cv2.erode(mask1,None,iterations = 2)
	mask = cv2.dilate(mask1,None,iterations = 2)

	cnts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None
	count = 0

	if len(cnts) > 0 :

		c = max(cnts,key= cv2.contourArea)
		((x,y),radius) = cv2.minEnclosingCircle(c)
		plotlist.append((count,radius))
		print(radius)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		if radius > 3:
			cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)
			pts.appendleft(center)

	for i in xrange(1, len(pts)):
		if pts[i - 1] is None or pts[i] is None:
			continue
		thickness = int(np.sqrt(32 / float(i + 1)) * 3)
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)


	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"): break

camera.release()
cv2.destroyAllWindows()

plt.plot(plotlist)
plt.show()