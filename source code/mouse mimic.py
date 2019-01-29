# libraries to be imported  //////////
import pyautogui
import cv2
import numpy as np
import math
import time
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')   # classifier to detect face

# //////////////////////


def nothing(x):
    pass


pyautogui.FAILSAFE = False
SCREEN_X, SCREEN_Y = pyautogui.size()
CLICK = CLICK_MESSAGE = MOVEMENT_START = None


cap = cv2.VideoCapture(0)
cv2.namedWindow("hand", cv2.WINDOW_NORMAL)
cv2.createTrackbar("hue_lower", "hand", 0, 255, nothing)  # 1) Creating trackbar for lower hue value so as to find the desired colored object in frame.
cv2.createTrackbar("hue_upper", "hand", 12, 255, nothing)  # Creating trackbar for upper hue value for same reason as above.
cv2.createTrackbar("saturation_lower", "hand", 57, 255, nothing)  # Creating trackbar for lower saturation value for same reason as above.
cv2.createTrackbar("saturation_upper", "hand", 255, 255, nothing)  # Creating trackbar for upper saturation value for same reason as above.
cv2.createTrackbar("value_lower", "hand", 67, 255, nothing)    # Creating trackbar for lower value for same reason as above.
cv2.createTrackbar("value_upper", "hand", 249, 255, nothing)    # Creating trackbar for upper value for same reason as above.
cv2.createTrackbar("dilate", "hand", 5, 255, nothing)           # Creating trackbar for dilate for same reason as above.
cv2.createTrackbar("erode", "hand", 0, 255, nothing)            # Creating trackbar for erode for same reason as above.
cv2.createTrackbar("blurr", "hand", 5, 255, nothing)            # Creating trackbar for blurr for same reason as above.


while (True):
    ret, img = cap.read()       # read image from camera continiously
    CAMERA_X, CAMERA_Y, channels = img.shape

    img = cv2.flip(img, 1)
    crop_img = img
    # fmask=bs.apply(crop_img)
    #kernel = np.ones((3,3),np.uint8)
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)  # convert image from BGR to HSV
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)  # convert image from BGR to GRAY

    # get the trackbar positions on Screen
    hl = cv2.getTrackbarPos("hue_lower", "hand")
    hu = cv2.getTrackbarPos("hue_upper", "hand")
    sl = cv2.getTrackbarPos("saturation_lower", "hand")
    su = cv2.getTrackbarPos("saturation_upper", "hand")
    vl = cv2.getTrackbarPos("value_lower", "hand")
    vu = cv2.getTrackbarPos("value_upper", "hand")
    dl = cv2.getTrackbarPos("dilate", "hand")
    er = cv2.getTrackbarPos("erode", "hand")
    blu = cv2.getTrackbarPos("blurr", "hand")
    # /////

    kernel = np.ones((dl, dl), np.uint8)
    kernel2 = np.ones((er, er), np.uint8)
    faces = detector.detectMultiScale(gray, 1.3, 5)  # detects faces in the image
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)   # makes a green rectangle around the face
        img[y:y + h + 10, x:x + w + 10] = 0

    value = (35, 35)
    lower_skin = np.array([hl, sl, vl], dtype=np.uint8)  # lower bound of screen color
    upper_skin = np.array([hu, su, vu], dtype=np.uint8)   # upper bound of screen color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    for (x, y, w, h) in faces:
        mask[y:y + h + 90, x:x + w + 40] = 0
    mask = cv2.GaussianBlur(mask, (3, 3), 0)              # blurr the image a bit
    mask = cv2.dilate(mask, kernel, iterations=4)       # dilate the image
    mask = cv2.erode(mask, kernel2, iterations=4)           # erode the image
    cv2.imshow('Blured', mask)

    image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # find contours in the b&w image
    max_area = -1
    for i in range(len(contours)):                   # find contour of max area
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            ci = i
    cnt = contours[ci]
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 0)  # draw rectangle across the largest area contour
    #epsilon = 0.0005*cv2.arcLength(cnt1,True)
    #approx= cv2.approxPolyDP(cnt,epsilon,True)
    hull = cv2.convexHull(cnt)                                       # draw a hull around the
    drawing = np.zeros(crop_img.shape, np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)                       # defects in the convex hull to detect fingers
    count_defects = 0
    cv2.drawContours(mask, contours, -1, (0, 255, 0), 3)

    used_defect = None
    for i in range(defects.shape[0]):                    # geting the defects
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)   # distance of 2 fingers
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)   # distance b/w defect and finger 1
        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)       # distance b/w defect and finger 2
        s = (a + b + c) / 2
        ar = math.sqrt(s * (s - a) * (s - b) * (s - c))
        d = (2 * ar) / a
        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57      # angle b/w two fingers
        if angle <= 90 and d > 50:

            count_defects += 1                                              # count number of defects
            cv2.circle(crop_img, far, 5, [0, 0, 255], -1)
        cv2.line(crop_img, start, end, [0, 255, 0], 2)
        medium_x = (start[0] + end[0]) / 2
        medium_y = (start[1] + end[1]) / 2

        # if count_defects==1 and angle<=90 and d>50:
        # pyautogui.press('right')
        # time.sleep(1)

        if count_defects == 2 and angle <= 90 and d > 50:
            used_defect = {"x": start[0], "y": start[1]}    # locate the second finger location when the number of fingers is 3

    if used_defect is not None:
        best = used_defect
        if count_defects == 2:                               # when number of defects is 2 i.e.. number of fingers is 3
            x = best['x']
            y = best['y']
            display_x = x
            display_y = y

            if MOVEMENT_START is not None:
                M_START = (x, y)
                x = x - MOVEMENT_START[0]
                y = y - MOVEMENT_START[1]
                x = x * (SCREEN_X / CAMERA_X)
                y = y * (SCREEN_Y / CAMERA_Y)
                MOVEMENT_START = M_START
                print("X: " + str(x) + " Y: " + str(y))     # print the location of 2nd finger
                pyautogui.moveRel(x, y)
            else:
                MOVEMENT_START = (x, y)

            cv2.circle(crop_img, (display_x, display_y), 2, [255, 255, 255], 20)   # make a white circle at point(tip of second finger)

        elif count_defects == 3 and CLICK is None:                               # number of defects is 3

            CLICK = time.time()
            pyautogui.click()

            CLICK_MESSAGE = "LEFT MOUSE"  # prints message 'left mouse'
        elif count_defects == 4 and CLICK is None:
            CLICK = time.time()
            pyautogui.rightClick()
            CLICK_MESSAGE = "RIGHT MOUSE"                                         # prints messsage 'right mouse'
    else:
        MOVEMENT_START = None

    if CLICK is not None:
        cv2.putText(img, CLICK_MESSAGE, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, 3)
        if CLICK < time.time():
            CLICK = None

    cv2.putText(img, " Finger count is" + str(count_defects + 1), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)  # put the number of fingers in screen
    cv2.imshow('Gesture', img)  # show the window (img)
    cv2.imshow('Drawing', drawing)  # show the window (drawing)

    k = cv2.waitKey(10) & 0xff
    if k == 27:                             # exit if escape command is executed
        break
