# libraries to be imported
import cv2
import numpy as np
import math
import subprocess
import pyautogui
import time


def nothing(x):
    pass


# to capture the vedio from webcam,change the 0 to 1 if external camera is attached.
cap = cv2.VideoCapture(0)
cv2.namedWindow("hand", cv2.WINDOW_NORMAL)
cv2.createTrackbar("hl", "hand", 0, 255, nothing)  # 1) Creating trackbar for lower hue value so as to find the desired colored object in frame.
cv2.createTrackbar("hu", "hand", 18, 255, nothing)  # Creating trackbar for upper hue value for same reason as above.
cv2.createTrackbar("sl", "hand", 36, 255, nothing)  # Creating trackbar for lower saturation value for same reason as above.
cv2.createTrackbar("su", "hand", 101, 255, nothing)  # Creating trackbar for upper saturation value for same reason as above.
cv2.createTrackbar("vl", "hand", 77, 255, nothing)    # Creating trackbar for lower value for same reason as above.
cv2.createTrackbar("vu", "hand", 255, 255, nothing)
# cv2.createTrackbar("k1","hand",0,100,nothing)
# cv2.createTrackbar("k2","hand",0,100,nothing)


while(1):

    try:  # an error comes if it does not find anything in window as it cannot find contour of max area
          # therefore this try error statement

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        kernel = np.ones((3, 3), np.uint8)

        # define region of interest
        roi = frame[100:300, 100:300]

        # make a rectangular box in the screen
        cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # get the trackbar values
        hl = cv2.getTrackbarPos("hl", "hand")
        hu = cv2.getTrackbarPos("hu", "hand")
        sl = cv2.getTrackbarPos("sl", "hand")
        su = cv2.getTrackbarPos("su", "hand")
        vl = cv2.getTrackbarPos("vl", "hand")
        vu = cv2.getTrackbarPos("vu", "hand")

    # define range of skin color in HSV
        lower_skin = np.array([hl, sl, vl], dtype=np.uint8)
        upper_skin = np.array([hu, su, vu], dtype=np.uint8)

     # extract skin colur imagw
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # extrapolate the hand to fill dark spots within
        mask = cv2.dilate(mask, kernel, iterations=4)

    # blur the image
        mask = cv2.GaussianBlur(mask, (5, 5), 100)

    # find contours
        _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

   # find contour of max area(hand)
        cnt = max(contours, key=lambda x: cv2.contourArea(x))
        #x, y, w, h = cv2.boundingRect(cnt)
        #cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 0)

    # approx the contour a little
        epsilon = 0.0005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

    # make convex hull around hand
        hull = cv2.convexHull(cnt)

     # define area of hull and area of hand
        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(cnt)

    # find the percentage of area not covered by hand in convex hull
        arearatio = ((areahull - areacnt) / areacnt) * 100

     # find the defects in convex hull with respect to hand
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)

    # l = no. of defects
        l = 0

    # code for finding no. of defects due to fingers
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt = (100, 180)

            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            s = (a + b + c) / 2
            ar = math.sqrt(s * (s - a) * (s - b) * (s - c))

            # distance between point and convex hull
            d = (2 * ar) / a

            # apply cosine rule here
            angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c)) * 57

            # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
            if angle <= 90 and d > 30:
                l += 1
                cv2.circle(roi, far, 3, [255, 0, 0], -1)

            # draw lines around hand
            cv2.line(roi, start, end, [0, 255, 0], 2)

        l += 1

        # print corresponding gestures which are in their ranges
        font = cv2.FONT_HERSHEY_SIMPLEX
        if l == 1:
            if areacnt < 2000:
                cv2.putText(frame, 'WELCOME SOURAV', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                if arearatio < 12:
                    cv2.putText(frame, '0,sourav open fb by showing 5 fingers..', (0, 80), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
                elif arearatio < 17.5:
                    cv2.putText(frame, 'Best of luck', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

                else:
                    cv2.putText(frame, '1', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l == 2:
            cv2.putText(frame, '2', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            # pyautogui.press('right')      # right press of keyboard when fingers are 3
            # time.sleep(0.5)

        elif l == 3:

              # if arearatio<27:
            cv2.putText(frame, '3', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            # pyautogui.press('left')       # left press on keyboard when fingers are 4.
            # time.sleep(0.5)
            # else:
            #cv2.putText(frame,'ok',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

        elif l == 4:
            cv2.putText(frame, '4', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l == 5:
            cv2.putText(frame, '5', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            subprocess.call([r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe', '-new-tab', 'http://www.facebook.com/'])
            # if fingers are 5,open fb
        elif l == 6:
            cv2.putText(frame, 'reposition', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        else:
            cv2.putText(frame, 'reposition', (10, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        # show the windows
        cv2.imshow('mask', mask)
        cv2.imshow('frame', frame)
    except:
        pass

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
