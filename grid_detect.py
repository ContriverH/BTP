from turtle import heading
from xml.etree.ElementTree import tostring
import cv2
import numpy as np
from lane import preprocess
from merged import trafficCounter
# stream processing of traffic

# Create point matrix get coordinates of mouse click on image
cordinates = np.zeros((2, 2), np.int)
counter = 2  # change it to 0 if you want to take input from mouse clicks
width = 20


def mousePoints(event, x, y, flags, params):
    global counter
    if event == cv2.EVENT_LBUTTONDOWN:
        cordinates[counter] = x, y
        counter = counter + 1


# Read image
cap = cv2.VideoCapture('Highway.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
fgbg2 = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
# imshow('subtracted', fgmask2)
carCount = 0
prevCount = 0
cordinates = preprocess("Highway.mp4")

while True:
    ret, frame = cap.read()
    fgmask2 = fgbg2.apply(frame)

    for x in range(0, 2):
        cv2.circle(
            frame, (cordinates[x][0], cordinates[x][1]), 3, (0, 255, 0), cv2.FILLED)

    if counter == 2:
        starting_x = cordinates[0][0]
        starting_y = cordinates[0][1]

        ending_x = cordinates[1][0]
        ending_y = cordinates[1][1]

        cell_count = 20
        cell_width = (ending_x - starting_x) / cell_count

        cropped_cells = []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for i in range(0, cell_count):
            x1 = int(i * cell_width) + starting_x
            x2 = int((i + 1) * cell_width) + starting_x
            # drawing cells
            cv2.rectangle(hsv, (x1, starting_y),
                          (x2, ending_y), (0, 255, 0), 2)

            # cropping cells
            cropped_cells.append(hsv[starting_y:ending_y, x1:x2])
            _h, _s, _v = cv2.split(cropped_cells[-1])
            _h_ = fgbg.apply(_h)
            _s_ = fgbg.apply(_s)
            _v_ = fgbg.apply(_v)

            h_v = cv2.bitwise_and(_h_, _v_)
            num_white = np.sum(h_v == 255)
            num_black = np.sum(h_v == 0)

            # % of white pixels , and black pixels
            percent = (num_white/(num_white + num_black))*100

        # Cropping image
        frame_cropped = hsv[starting_y:ending_y, starting_x:ending_x]

        cv2.imshow("Cropped Area", frame_cropped)
    
    trafficCounter(cordinates)
    cv2.imshow("Original Image ", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()