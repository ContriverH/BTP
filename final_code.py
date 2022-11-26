# BTP
# Grid based Real-time Image processing algorithm for Heterogeneous traffic-system

#       Mentor:
#              Dr. S. Manipriya

# Contributers:
#              Himanshu Pal
#              Nishchay Verma
#              Laukik   Verma

# ---------------------------------------------------------------------------------


from math import ceil
from turtle import Vec2D, heading
from xml.etree.ElementTree import tostring
from cv2 import imshow
import numpy as np
import cv2
import cv2 as cv
import pandas as pd



# ------------------------------------------------------------------------------------------------------------------

# Create point matrix get coordinates of mouse click on image
point_matrix = np.zeros((2, 2), int)  # matrix to store the two coordinates
counter = 0

'''
   mousePoints: capturing the coordinates of the clicked position of the mouse
'''

def mousePoints(event, x, y, flags, params):
    global counter
    if event == cv2.EVENT_LBUTTONDOWN:  # on pressing the left mouse button
        point_matrix[counter] = x, y
        counter = counter + 1


# Capturing the original video
cap = cv2.VideoCapture('Highway.mp4')


frames_count, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(
    cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = int(width)
height = int(height)
print(frames_count, fps, width, height)

# creates a pandas data frame with the number of rows the same length as frame count
df = pd.DataFrame(index=range(int(frames_count)))
df.index.name = "Frames"

framenumber = 0  # keeps track of current frame
carscrossedup = 0  # keeps track of cars that crossed up
carscrosseddown = 0  # keeps track of cars that crossed down
carids = []  # blank list to add car ids
caridscrossed = []  # blank list to add car ids that have crossed
totalcars = 0  # keeps track of total cares

fgbg = cv2.createBackgroundSubtractorMOG2()  # create background subtractor

# information to start saving a video file
ret, frame = cap.read()  # import image
ratio = .5  # resize ratio
# image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # resize image
width2, height2, channels = frame.shape
video = cv2.VideoWriter('traffic_counter.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (height2, width2), 1)  
    

# detecting the object with shadow
# for not having the shadow along with the object, we can set detectShadows=True
# storing the createBackgroundSubtractorMOG2 method in the no_background_method variable
no_background_method = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

# deciding the total cells
# 10 1 2 3 perfect fit till now

cell_count = 24

# deciding the width of the vehicle on the basis of number of the cell-count
# max_width = int(ceil(cell_count / 4))
max_width = 8

# width of each vehicle
# bike_width = max_width / 3
# car_width = 2 * (max_width / 3)
# truck_width = max_width

bike_width = 2
car_width = 6
truck_width = 8

# deciding the white threshold percentage
white_percent_threshold = 30

# counting the vehicle
vehicle_count = 0
prev_vehicle_count = -1
# bike = car = truck = 0
bike, car, truck = 0, 0, 0

# to check what was the status of the h&v value of the grids in previous frame
prev = [0] * cell_count



fgbg = cv2.createBackgroundSubtractorMOG2()  # create background subtractor


# ------------------------------------------------------------------------------------------------------------------

while True:
    # working on each frame of the video, each frame is interpreted as a single image and processing is done on it
    ret, frame = cap.read()

    image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # resize image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converts image to gray

    fgmask = fgbg.apply(gray)  # uses the background subtraction

    cv2.imshow("Intense image", gray)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # kernel to apply to the morphology
    closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    dilation = cv2.dilate(opening, kernel)
    retvalbin, bins = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY)  # removes the shadows
    
    cv2.imshow("Intense image", bins) 


    # getting the image with object and shadow
    # no_background_frame = no_background_method.apply(frame)
    # cv2.imshow('no background frame', no_background_frame)

    # marking the two points on the image as circle
    for x in range(0, 2):
        cv2.circle(
            frame, (point_matrix[x][0], point_matrix[x][1]), 3, (0, 255, 0), cv2.FILLED)

    # if two points are selected, then we can go for the grid formation
    if counter == 2:
        # point 1
        # starting_x = point_matrix[0][0]
        # starting_y = point_matrix[0][1]
        starting_x = 20
        starting_y = 690

        # point 2
        # ending_x = point_matrix[1][0]
        # ending_y = point_matrix[1][1]
        ending_x = 990
        ending_y = 719

        # diving the total grid into 10 cells
        # cell_width will be adjusted accoring to the area of the frame
        cell_width = (ending_x - starting_x) / cell_count

        cropped_cells = []

        # converting the whole frame into hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # -----------------------------------------------------------
        # work on edge detection technique

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # img_gaussian = cv2.GaussianBlur(gray, (3, 3), 0)

        # kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        # kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        # # img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
        # img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)

        # # img_prewitty = cv2.LUT(img_prewitty, [128], [255])
        # _, img_prewitty = cv.threshold(
        #     img_prewitty, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        # # print(img_prewitty)
        # # cv2.imshow("prettyx", img_prewittx)
        # cv2.imshow("prettyy", img_prewitty)

        # ---------------------------------------------------------------------------------
        # To show the hsv frames
        # cv2.imshow("hsv Image ", hsv)

        # h, s, v = cv2.split(frame)

        # h = no_background_method.apply(h)  # hue
        # s = no_background_method.apply(s)  # saturation
        # v = no_background_method.apply(v)  # value

        # # calculating: h & v, this is done so that the whole cell is converted into b&w
        # _h_v = cv2.bitwise_and(h, v)

        # # to show the h&v video for b&w images
        # cv2.imshow("h&v Image ", _h_v)
        # ---------------------------------------------------------------------------------

        # storing the 0 1 values of the current cell
        cur = [0] * cell_count

        for i in range(0, cell_count):
            # getting the coordinate of every cell, (start and end points)
            x1 = int(i * cell_width) + starting_x
            x2 = int((i + 1) * cell_width) + starting_x

            # drawing cells
            cv2.rectangle(hsv, (x1, starting_y),
                          (x2, ending_y), (0, 255, 0), 2)

            # cropping each cell
            cropped_cells.append(hsv[starting_y:ending_y, x1:x2])

            # accessing the h, s, v values of the recently pushed cell
            _h, _s, _v = cv2.split(cropped_cells[-1])

            # print (type(_h), _h.shape)

            _h_ = no_background_method.apply(_h)  # hue
            _s_ = no_background_method.apply(_s)  # saturation
            _v_ = no_background_method.apply(_v)  # value

            # calculating: h & v, this is done so that the whole cell is converted into b&w
            h_v = cv2.bitwise_and(_h_, _v_)

            # checking the type of values in the h_v
            # temp = list(set(h_v.flatten()))
            # print(temp)
            # break

            num_white = np.sum(h_v == 255)
            num_black = np.sum(h_v == 0)

            # % of white pixels
            white_percent = (num_white/(num_white + num_black))*100

            # if the white percent in the cell is more than given threshold, then marking it as vehicle is going through it
            if(white_percent > white_percent_threshold):
                cur[i] = 1

            # cv2.imshow("hue " + str(i), _h)
            # cv2.imshow("saturation " + str(i), _s)
            # cv2.imshow("value " + str(i), _v)

        # vehicle classfication and vehicle count
        i = 0
        while i < cell_count:
            if(cur[i] == 1 and prev[i] != 1):
                vehicle_width = 0
                for j in range(i, min(cell_count, i + max_width)):
                    if(cur[j] == 1 and prev[j] != 1):
                        vehicle_width += 1
                    else:
                        break

                i += vehicle_width  # moving to the cell after the current vehicle

                # classifying the vehicle based on their size (size means the cells they are occupying)
                if vehicle_width > 0 and vehicle_width <= bike_width:
                    bike += 1
                elif vehicle_width > bike_width and vehicle_width <= car_width:
                    car += 1
                elif vehicle_width > car_width and vehicle_width <= truck_width:
                    truck += 1

                if vehicle_width > 0:
                    vehicle_count += 1

            i += 1  # moving to next cell

        # updating the prev array of the++ 0s and 1s with current values of 0s and 1s
        prev = cur

        # Cropping grid to display
        frame_cropped = hsv[starting_y:ending_y, starting_x:ending_x]
        cv2.imshow("Cropped Area", frame_cropped)

    # showing the orignal frame
    cv2.imshow("Original Image ", frame)

    # feeding the original image to capture the mouseclick point
    cv2.setMouseCallback("Original Image ", mousePoints)

    # printing only when the count of vehicle changes
    if prev_vehicle_count != vehicle_count:
        print("Bike: ", bike, ", Car: ", car, "Truck: ", truck)
        print("Vehicles passed by now: ", vehicle_count)
        prev_vehicle_count = vehicle_count  # updating the previous count of vehicle

    # functionality to break the while loop on pressing key q
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()  # release video capture object
# destroying all running videos after the execution ends
cv2.destroyAllWindows()

# tasks
#     : check the video with shadow and without shadow
#     : lane-detection
#     : fixing error at the end of the file
#     : flow of traffic
#     : collecting 4-videos for ppt and name of video dataset
#     : deciding the no. of cells needed to classify the vehicle correctly
#     : deciding the number of cells needed on a fixed width of the road (sigle lane- 3.5m,     double lane - 7m)
#     : deciding the position of points on the lanes detected
#     : preparing ppt