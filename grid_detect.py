from turtle import heading
from xml.etree.ElementTree import tostring
import cv2
import numpy as np
# stream processing of traffic

# Create point matrix get coordinates of mouse click on image
point_matrix = np.zeros((2, 2), np.int)
counter = 0
width = 20


def mousePoints(event, x, y, flags, params):
    global counter
    if event == cv2.EVENT_LBUTTONDOWN:
        point_matrix[counter] = x, y
        counter = counter + 1


# Read image
cap = cv2.VideoCapture('Highway.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
fgbg2 = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
# imshow('subtracted', fgmask2)
carCount = 0
prevCount = 0

while True:
    ret, frame = cap.read()
    fgmask2 = fgbg2.apply(frame)

    for x in range(0, 2):
        cv2.circle(
            frame, (point_matrix[x][0], point_matrix[x][1]), 3, (0, 255, 0), cv2.FILLED)

    if counter == 2:
        starting_x = point_matrix[0][0]
        starting_y = point_matrix[0][1]

        ending_x = point_matrix[1][0]
        ending_y = point_matrix[1][1]

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

            # print(" HUE values ")
            # print(_h_)

            # print(" Saturation value ")
            # print(_s_)

            h_v = cv2.bitwise_and(_h_, _v_)
            num_white = np.sum(h_v == 255)
            num_black = np.sum(h_v == 0)

            # % of white pixels , and black pixels
            percent = (num_white/(num_white + num_black))*100

            if(percent > 50):
                carCount += 1
                if prevCount != carCount:
                    print("Object Detected = " + str(round(carCount)))
                    prevCount = carCount

            # cv2.imshow("show1" + str(i), _h)
            # cv2.imshow("sho2" + str(i), _s)
            # cv2.imshow("show3" + str(i), _v)

        # Cropping image
        frame_cropped = hsv[starting_y:ending_y, starting_x:ending_x]

        cv2.imshow("Cropped Area", frame_cropped)

    cv2.imshow("Original Image ", frame)
    cv2.setMouseCallback("Original Image ", mousePoints)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
