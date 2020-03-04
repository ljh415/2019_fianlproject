from multiprocessing import Process
import time


def filter(a) :
    if a <= 20:
        pass
    elif a <= 40:
        a = a * 0.75
    elif a <= 60:
        a = a * 0.53
    elif a <= 80:
        a = a * 0.42
    else:
        a = a * 0.36

    a = int(a)

    return str(a)

def flow() :
    import numpy as np
    import cv2
    import pandas as pd

    ##### import, 넘파이, cv2, 판다스

    import socket

    cap = cv2.VideoCapture('Default_cut.mp4')
    # frames_count, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS),
    # cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # 영상으로부터 정보를 받아온다.
    frames_count, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    # width = 1400
    # height = 550
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(frames_count, fps, width, height)

    # creates a pandas data frame with the number of rows the same length as frame count
    df_A = pd.DataFrame(index=range(int(frames_count)))
    df_A.index.name = "Frames"
    df_B = pd.DataFrame(index=range(int(frames_count)))
    df_B.index.name = "Frames"
    df_C = pd.DataFrame(index=range(int(frames_count)))
    df_C.index.name = "Frames"
    df_D = pd.DataFrame(index=range(int(frames_count)))
    df_D.index.name = "Frames"

    # 나가는 차량 대수 확인
    df_PA = pd.DataFrame(index=range(int(frames_count)))
    df_PA.index.name = "Frames"
    df_PB = pd.DataFrame(index=range(int(frames_count)))
    df_PB.index.name = "Frames"
    df_PC = pd.DataFrame(index=range(int(frames_count)))
    df_PC.index.name = "Frames"
    df_PD = pd.DataFrame(index=range(int(frames_count)))
    df_PD.index.name = "Frames"

    framenumber = 0  # keeps track of current frame
    carscrossedup = 0  # keeps track of cars that crossed up
    carscrosseddown = 0  # keeps track of cars that crossed down

    # 영역 진입차량
    cars_pass_A_u = 0
    cars_pass_A_d = 0
    carids_A = []
    caridspassed_A = []

    cars_pass_B_u = 0
    cars_pass_B_d = 0
    carids_B = []
    caridspassed_B = []

    cars_pass_C_u = 0
    cars_pass_C_d = 0
    carids_C = []
    caridspassed_C = []

    cars_pass_D_u = 0
    cars_pass_D_d = 0
    carids_D = []
    caridspassed_D = []

    # 영역 나가는 차량
    cars_out_A_u = 0
    cars_out_A_d = 0
    carids_PA = []
    caridspassed_PA = []

    cars_out_B_u = 0
    cars_out_B_d = 0
    carids_PB = []
    caridspassed_PB = []

    cars_out_C_u = 0
    cars_out_C_d = 0
    carids_PC = []
    caridspassed_PC = []

    cars_out_D_u = 0
    cars_out_D_d = 0
    carids_PD = []
    caridspassed_PD = []

    totalcars_A, totalcars_B, totalcars_C, totalcars_D = 0, 0, 0, 0

    totalcars_PA, totalcars_PB, totalcars_PC, totalcars_PD = 0, 0, 0, 0

    fgbg = cv2.createBackgroundSubtractorMOG2()  # create background subtractor
    # 배경을 빼서 움직이는 물체를 검출하는 알고리즘

    # information to start saving a video
    ret, frame = cap.read()  # import image
    ratio = .45

    ratio2 = 1

    image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # resize image
    width2, height2, channels = image.shape
    video = cv2.VideoWriter('traffic_counter.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (height2, width2),
                            1)

    while True:

        ret, frame = cap.read()  # import image

        if ret:  # if there is a frame continue with code

            image = cv2.resize(frame, (0, 0), None, ratio2, ratio2)  # resize image

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converts image to gray

            fgmask = fgbg.apply(gray)  # uses the background subtraction

            # applies different thresholds to fgmask to try and isolate cars
            # just have to keep playing around with settings until cars are easily identifiable
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # kernel to apply to the morphology
            closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
            opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
            dilation = cv2.dilate(opening, kernel)
            # retvalbin, bins = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY)  # removes the shadows
            retvalbin, bins = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY)  # removes the shadows

            # creates contours
            # im2, contours, hierarchy = cv2.findContours(bins, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            (contours, hierarchy) = cv2.findContours(bins, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # use convex hull to create polygon around contours
            hull = [cv2.convexHull(c) for c in contours]
            # convexhull : 외각선 찾기 알고리즘

            # draw contours
            cv2.drawContours(image, hull, -1, (0, 255, 0), 3)

            # Counter_A
            CounterA_in_blue = 10
            CounterA_out_blue = 70
            CounterA_upper = 363
            CounterA_bot = 423
            # CounterA의 check box
            cv2.line(image, (CounterA_in_blue, CounterA_upper), (CounterA_in_blue, CounterA_bot), (255, 0, 0), 2)
            cv2.line(image, (CounterA_out_blue, CounterA_upper), (CounterA_out_blue, CounterA_bot), (255, 0, 0), 2)
            cv2.line(image, (CounterA_in_blue, CounterA_upper), (CounterA_out_blue, CounterA_upper), (255, 0, 0), 2)
            cv2.line(image, (CounterA_in_blue, CounterA_bot), (CounterA_out_blue, CounterA_bot), (255, 0, 0), 2)
            # CounterA의 check Line
            CounterA_check = 50
            cv2.line(image, (CounterA_check, CounterA_upper), (CounterA_check, CounterA_bot), (0, 255, 0), 2)

            # Counter B
            CounterB_in_blue = int(width - CounterA_in_blue)
            CounterB_out_blue = int(width - CounterA_out_blue)
            CounterB_upper = 303
            CounterB_bot = 363
            # CounterB의 check box
            cv2.line(image, (CounterB_in_blue, CounterB_upper), (CounterB_in_blue, CounterB_bot), (255, 0, 0), 2)
            cv2.line(image, (CounterB_out_blue, CounterB_upper), (CounterB_out_blue, CounterB_bot), (255, 0, 0), 2)
            cv2.line(image, (CounterB_in_blue, CounterB_upper), (CounterB_out_blue, CounterB_upper), (255, 0, 0), 2)
            cv2.line(image, (CounterB_in_blue, CounterB_bot), (CounterB_out_blue, CounterB_bot), (255, 0, 0), 2)
            # CounterB의 check Line
            CounterB_check = int(width - CounterA_check)
            cv2.line(image, (CounterB_check, CounterB_upper), (CounterB_check, CounterB_bot), (0, 255, 0), 2)

            # Counter C
            CounterC_in_blue = 25
            CounterC_out_blue = 85
            CounterC_Left = 295
            CounterC_Right = 355
            # CounterC의 check box
            cv2.line(image, (CounterC_Left, CounterC_in_blue), (CounterC_Right, CounterC_in_blue), (255, 0, 0), 2)
            cv2.line(image, (CounterC_Left, CounterC_out_blue), (CounterC_Right, CounterC_out_blue), (255, 0, 0), 2)
            cv2.line(image, (CounterC_Left, CounterC_in_blue), (CounterC_Left, CounterC_out_blue), (255, 0, 0), 2)
            cv2.line(image, (CounterC_Right, CounterC_in_blue), (CounterC_Right, CounterC_out_blue), (255, 0, 0), 2)
            # CounterC의 check Line
            CounterC_check = 65
            cv2.line(image, (CounterC_Left, CounterC_check), (CounterC_Right, CounterC_check), (0, 255, 0), 2)

            # Counter D
            CounterD_in_blue = int(height - 10)
            CounterD_out_blue = CounterD_in_blue - 60
            CounterD_Left = 370
            CounterD_Right = CounterD_Left + 60
            # CounterD의 check box
            cv2.line(image, (CounterD_Left, CounterD_in_blue), (CounterD_Right, CounterD_in_blue), (255, 0, 0), 2)
            cv2.line(image, (CounterD_Left, CounterD_out_blue), (CounterD_Right, CounterD_out_blue), (255, 0, 0), 2)
            cv2.line(image, (CounterD_Left, CounterD_in_blue), (CounterD_Left, CounterD_out_blue), (255, 0, 0), 2)
            cv2.line(image, (CounterD_Right, CounterD_in_blue), (CounterD_Right, CounterD_out_blue), (255, 0, 0), 2)
            # CounterD의 check Line
            CounterD_check = int(height - 50)
            cv2.line(image, (CounterD_Left, CounterD_check), (CounterD_Right, CounterD_check), (0, 255, 0), 2)

            # Counter_passout : Pass
            # Pass_A
            Pass_A_in_blue = 227
            Pass_A_out_blue = Pass_A_in_blue + 60
            Pass_A_upper = 360
            Pass_A_bot = Pass_A_upper + 60
            # PassA의 check box
            cv2.line(image, (Pass_A_in_blue, Pass_A_upper), (Pass_A_in_blue, Pass_A_bot), (255, 0, 0), 2)
            cv2.line(image, (Pass_A_out_blue, Pass_A_upper), (Pass_A_out_blue, Pass_A_bot), (255, 0, 0), 2)
            cv2.line(image, (Pass_A_in_blue, Pass_A_upper), (Pass_A_out_blue, Pass_A_upper), (255, 0, 0), 2)
            cv2.line(image, (Pass_A_in_blue, Pass_A_bot), (Pass_A_out_blue, Pass_A_bot), (255, 0, 0), 2)
            # vPassA의 check Line
            Pass_A_check = 267
            cv2.line(image, (Pass_A_check, Pass_A_upper), (Pass_A_check, Pass_A_bot), (0, 255, 0), 2)

            # Pass_B
            Pass_B_in_blue = int(width - Pass_A_in_blue) + 10
            Pass_B_out_blue = int(width - Pass_A_out_blue) + 10
            Pass_B_upper = 306
            Pass_B_bot = Pass_B_upper + 60
            # Pass_B의 check box
            cv2.line(image, (Pass_B_in_blue, Pass_B_upper), (Pass_B_in_blue, Pass_B_bot), (255, 0, 0), 2)
            cv2.line(image, (Pass_B_out_blue, Pass_B_upper), (Pass_B_out_blue, Pass_B_bot), (255, 0, 0), 2)
            cv2.line(image, (Pass_B_in_blue, Pass_B_upper), (Pass_B_out_blue, Pass_B_upper), (255, 0, 0), 2)
            cv2.line(image, (Pass_B_in_blue, Pass_B_bot), (Pass_B_out_blue, Pass_B_bot), (255, 0, 0), 2)
            # Pass_B의 check Line
            Pass_B_check = int(width - Pass_A_check) + 10
            cv2.line(image, (Pass_B_check, Pass_B_upper), (Pass_B_check, Pass_B_bot), (0, 255, 0), 2)

            # Pass_C
            Pass_C_in_blue = 232
            Pass_C_out_blue = 292
            Pass_C_Left = 298
            Pass_C_Right = Pass_C_Left + 60
            # PassA의 check box
            cv2.line(image, (Pass_C_Left, Pass_C_in_blue), (Pass_C_Right, Pass_C_in_blue), (255, 0, 0), 2)
            cv2.line(image, (Pass_C_Left, Pass_C_out_blue), (Pass_C_Right, Pass_C_out_blue), (255, 0, 0), 2)
            cv2.line(image, (Pass_C_Left, Pass_C_in_blue), (Pass_C_Left, Pass_C_out_blue), (255, 0, 0), 2)
            cv2.line(image, (Pass_C_Right, Pass_C_in_blue), (Pass_C_Right, Pass_C_out_blue), (255, 0, 0), 2)
            # vPassA의 check Line
            Pass_C_check = 272
            cv2.line(image, (Pass_C_Left, Pass_C_check), (Pass_C_Right, Pass_C_check), (0, 255, 0), 2)

            # Pass_D
            Pass_D_in_blue = int(height - 235) + 45
            Pass_D_out_blue = int(height - 295) + 45
            Pass_D_Left = 355
            Pass_D_Right = Pass_D_Left + 60
            # PassA의 check box
            cv2.line(image, (Pass_D_Left, Pass_D_in_blue), (Pass_D_Right, Pass_D_in_blue), (255, 0, 0), 2)
            cv2.line(image, (Pass_D_Left, Pass_D_out_blue), (Pass_D_Right, Pass_D_out_blue), (255, 0, 0), 2)
            cv2.line(image, (Pass_D_Left, Pass_D_in_blue), (Pass_D_Left, Pass_D_out_blue), (255, 0, 0), 2)
            cv2.line(image, (Pass_D_Right, Pass_D_in_blue), (Pass_D_Right, Pass_D_out_blue), (255, 0, 0), 2)
            # vPassA의 check Line
            Pass_D_check = int(height - Pass_C_check) + 43
            cv2.line(image, (Pass_D_Left, Pass_D_check), (Pass_D_Right, Pass_D_check), (0, 255, 0), 2)

            # min area for contours in case a bunch of small noise contours are created
            minarea = 210

            # max area for contours, can be quite large for buses
            maxarea = 50000

            # vectors for the x and y locations of contour centroids in current frame
            cxx_A = np.zeros(len(contours))
            cyy_A = np.zeros(len(contours))
            cxx_B = np.zeros(len(contours))
            cyy_B = np.zeros(len(contours))
            cxx_C = np.zeros(len(contours))
            cyy_C = np.zeros(len(contours))
            cxx_D = np.zeros(len(contours))
            cyy_D = np.zeros(len(contours))

            cxx_PA = np.zeros(len(contours))
            cyy_PA = np.zeros(len(contours))
            cxx_PB = np.zeros(len(contours))
            cyy_PB = np.zeros(len(contours))
            cxx_PC = np.zeros(len(contours))
            cyy_PC = np.zeros(len(contours))
            cxx_PD = np.zeros(len(contours))
            cyy_PD = np.zeros(len(contours))

            # pass

            for i in range(len(contours)):  # cycles through all contours in current frame

                if hierarchy[
                    0, i, 3] == -1:  # using hierarchy to only count parent contours (contours not within others)

                    area = cv2.contourArea(contours[i])  # area of contour
                    # contour의 면적

                    if minarea < area < maxarea:  # area threshold for contour
                        # 면적이 min과 max의 범위 안에 있다면

                        # calculating centroids of contours
                        cnt = contours[i]
                        M = cv2.moments(cnt)
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        # 중심을 저장시킨다

                        # cx로 바꿈 원래cy
                        # Area_A ( 관심영역(ROI)안에 있는 무게중심만 확인 )
                        if cx > CounterA_in_blue and cx < CounterA_out_blue and cy < CounterA_bot and cy > CounterA_upper:
                            # gets bounding points of contour to create rectangle
                            # x,y is top left corner and w,h is width and height
                            x, y, w, h = cv2.boundingRect(cnt)

                            # creates a rectangle around contour
                            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                            # Prints centroid text in order to double check later on
                            cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                        .3, (0, 0, 255), 1)

                            cv2.drawMarker(image, (cx, cy), (0, 0, 255), cv2.MARKER_STAR, markerSize=5, thickness=1,
                                           line_type=cv2.LINE_AA)

                            # adds centroids that passed previous criteria to centroid list
                            cxx_A[i] = cx
                            cyy_A[i] = cy

                        # Area_B
                        elif cx < CounterB_in_blue and cx > CounterB_out_blue and cy < CounterB_bot and cy > CounterB_upper:
                            # gets bounding points of contour to create rectangle
                            # x,y is top left corner and w,h is width and height
                            x, y, w, h = cv2.boundingRect(cnt)

                            # creates a rectangle around contour
                            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                            # Prints centroid text in order to double check later on
                            cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                        .3,
                                        (0, 0, 255), 1)

                            cv2.drawMarker(image, (cx, cy), (0, 0, 255), cv2.MARKER_STAR, markerSize=5, thickness=1,
                                           line_type=cv2.LINE_AA)

                            # adds centroids that passed previous criteria to centroid list
                            cxx_B[i] = cx
                            cyy_B[i] = cy

                        # Area C
                        elif cy > CounterC_in_blue and cy < CounterC_out_blue and cx < CounterC_Right and cx > CounterC_Left:
                            # gets bounding points of contour to create rectangle
                            # x,y is top left corner and w,h is width and height
                            x, y, w, h = cv2.boundingRect(cnt)

                            # creates a rectangle around contour
                            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                            # Prints centroid text in order to double check later on
                            cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                        .3,
                                        (0, 0, 255), 1)

                            cv2.drawMarker(image, (cx, cy), (0, 0, 255), cv2.MARKER_STAR, markerSize=5, thickness=1,
                                           line_type=cv2.LINE_AA)

                            # adds centroids that passed previous criteria to centroid list
                            cxx_C[i] = cx
                            cyy_C[i] = cy

                        # Area D
                        elif cy < CounterD_in_blue and cy > CounterD_out_blue and cx < CounterD_Right and cx > CounterD_Left:
                            # gets bounding points of contour to create rectangle
                            # x,y is top left corner and w,h is width and height
                            x, y, w, h = cv2.boundingRect(cnt)

                            # creates a rectangle around contour
                            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                            # Prints centroid text in order to double check later on
                            cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                        .3,
                                        (0, 0, 255), 1)

                            cv2.drawMarker(image, (cx, cy), (0, 0, 255), cv2.MARKER_STAR, markerSize=5, thickness=1,
                                           line_type=cv2.LINE_AA)

                            # adds centroids that passed previous criteria to centroid list
                            cxx_D[i] = cx
                            cyy_D[i] = cy


                        # Pass _ A
                        elif cx > Pass_A_in_blue and cx < Pass_A_out_blue and cy < Pass_A_bot and cy > Pass_A_upper:
                            # gets bounding points of contour to create rectangle
                            # x,y is top left corner and w,h is width and height
                            x, y, w, h = cv2.boundingRect(cnt)

                            # creates a rectangle around contour
                            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                            # Prints centroid text in order to double check later on
                            cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                        .3,
                                        (0, 0, 255), 1)

                            cv2.drawMarker(image, (cx, cy), (0, 0, 255), cv2.MARKER_STAR, markerSize=5, thickness=1,
                                           line_type=cv2.LINE_AA)

                            # adds centroids that passed previous criteria to centroid list
                            cxx_PA[i] = cx
                            cyy_PA[i] = cy

                        # Pass _ B
                        elif cx < Pass_B_in_blue and cx > Pass_B_out_blue and cy < Pass_B_bot and cy > Pass_B_upper:
                            # gets bounding points of contour to create rectangle
                            # x,y is top left corner and w,h is width and height
                            x, y, w, h = cv2.boundingRect(cnt)

                            # creates a rectangle around contour
                            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                            # Prints centroid text in order to double check later on
                            cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                        .3,
                                        (0, 0, 255), 1)

                            cv2.drawMarker(image, (cx, cy), (0, 0, 255), cv2.MARKER_STAR, markerSize=5, thickness=1,
                                           line_type=cv2.LINE_AA)

                            # adds centroids that passed previous criteria to centroid list
                            cxx_PB[i] = cx
                            cyy_PB[i] = cy

                        # Pass _ C
                        elif cy > Pass_C_in_blue and cy < Pass_C_out_blue and cx < Pass_C_Right and cx > Pass_C_Left:
                            # gets bounding points of contour to create rectangle
                            # x,y is top left corner and w,h is width and height
                            x, y, w, h = cv2.boundingRect(cnt)

                            # creates a rectangle around contour
                            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                            # Prints centroid text in order to double check later on
                            cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                        .3,
                                        (0, 0, 255), 1)

                            cv2.drawMarker(image, (cx, cy), (0, 0, 255), cv2.MARKER_STAR, markerSize=5, thickness=1,
                                           line_type=cv2.LINE_AA)

                            # adds centroids that passed previous criteria to centroid list
                            cxx_PC[i] = cx
                            cyy_PC[i] = cy

                        # Pass _ D
                        elif cy < Pass_D_in_blue and cy > Pass_D_out_blue and cx < Pass_D_Right and cx > Pass_D_Left:
                            # gets bounding points of contour to create rectangle
                            # x,y is top left corner and w,h is width and height
                            x, y, w, h = cv2.boundingRect(cnt)

                            # creates a rectangle around contour
                            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                            # Prints centroid text in order to double check later on
                            cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                        .3,
                                        (0, 0, 255), 1)

                            cv2.drawMarker(image, (cx, cy), (0, 0, 255), cv2.MARKER_STAR, markerSize=5, thickness=1,
                                           line_type=cv2.LINE_AA)

                            # adds centroids that passed previous criteria to centroid list
                            cxx_PD[i] = cx
                            cyy_PD[i] = cy

            # eliminates zero entries (centroids that were not added)
            cxx_A = cxx_A[cxx_A != 0]
            cyy_A = cyy_A[cyy_A != 0]
            cxx_B = cxx_B[cxx_B != 0]
            cyy_B = cyy_B[cyy_B != 0]
            cxx_C = cxx_C[cxx_C != 0]
            cyy_C = cyy_C[cyy_C != 0]
            cxx_D = cxx_D[cxx_D != 0]
            cyy_D = cyy_D[cyy_D != 0]

            cxx_PA = cxx_PA[cxx_PA != 0]
            cyy_PA = cyy_PA[cyy_PA != 0]
            cxx_PB = cxx_PB[cxx_PB != 0]
            cyy_PB = cyy_PB[cyy_PB != 0]
            cxx_PC = cxx_PC[cxx_PC != 0]
            cyy_PC = cyy_PC[cyy_PC != 0]
            cxx_PD = cxx_PD[cxx_PD != 0]
            cyy_PD = cyy_PD[cyy_PD != 0]

            # empty list to later check which centroid indices were added to dataframe
            minx_index2 = []
            miny_index2 = []

            # maximum allowable radius for current frame centroid to be considered the same centroid from previous frame
            maxrad = 25

            # The section below keeps track of the centroids and assigns them to old carids or new carids

            if len(cxx_A):  # if there are centroids in the specified area

                aa = 0

                if not carids_A:  # if carids_A is empty

                    for i in range(len(cxx_A)):  # loops through all centroids

                        carids_A.append(i)  # adds a car id to the empty list carids_A

                        fa = open('fa.txt', 'w')
                        aa = len(carids_A)
                        fa.write(str(aa))
                        fa.close()
                        print("aa saved :", aa)

                        df_A[str(carids_A[i])] = ""  # adds a column to the dataframe corresponding to a carid

                        # assigns the centroid values to the current frame (row) and carid (column)
                        df_A.at[int(framenumber), str(carids_A[i])] = [cxx_A[i], cyy_A[i]]

                        totalcars_A = carids_A[i] + 1  # adds one count to total cars

                else:  # if there are already car ids

                    dx = np.zeros((len(cxx_A), len(carids_A)))  # new arrays to calculate deltas
                    dy = np.zeros((len(cyy_A), len(carids_A)))  # new arrays to calculate deltas

                    for i in range(len(cxx_A)):  # loops through all centroids

                        for j in range(len(carids_A)):  # loops through all recorded car ids

                            # acquires centroid from previous frame for specific carid
                            oldcxcy = df_A.iloc[int(framenumber - 1)][str(carids_A[j])]

                            # acquires current frame centroid that doesn't necessarily line up with previous frame centroid
                            curcxcy = np.array([cxx_A[i], cyy_A[i]])

                            if not oldcxcy:  # checks if old centroid is empty in case car leaves screen and new car shows

                                continue  # continue to next carid

                            else:  # calculate centroid deltas to compare to current frame position later

                                dx[i, j] = oldcxcy[0] - curcxcy[0]
                                dy[i, j] = oldcxcy[1] - curcxcy[1]

                    for j in range(len(carids_A)):  # loops through all current car ids

                        sumsum = np.abs(dx[:, j]) + np.abs(dy[:, j])  # sums the deltas wrt to car ids

                        # finds which index carid had the min difference and this is true index
                        correctindextrue = np.argmin(np.abs(sumsum))
                        minx_index = correctindextrue
                        miny_index = correctindextrue

                        # acquires delta values of the minimum deltas in order to check if it is within radius later on
                        mindx = dx[minx_index, j]
                        mindy = dy[miny_index, j]

                        if mindx == 0 and mindy == 0 and np.all(dx[:, j] == 0) and np.all(dy[:, j] == 0):
                            # checks if minimum value is 0 and checks if all deltas are zero since this is empty set
                            # delta could be zero if centroid didn't move

                            continue  # continue to next carid

                        else:

                            # if delta values are less than maximum radius then add that centroid to that specific carid
                            if np.abs(mindx) < maxrad and np.abs(mindy) < maxrad:
                                # adds centroid to corresponding previously existing carid
                                df_A.at[int(framenumber), str(carids_A[j])] = [cxx_A[minx_index], cyy_A[miny_index]]
                                minx_index2.append(
                                    minx_index)  # appends all the indices that were added to previous carids_A
                                miny_index2.append(miny_index)

                    for i in range(len(cxx_A)):  # loops through all centroids

                        # if centroid is not in the minindex list then another car needs to be added
                        if i not in minx_index2 and miny_index2:

                            df_A[str(totalcars_A)] = ""  # create another column with total cars
                            totalcars_A = totalcars_A + 1  # adds another total car the count
                            t = totalcars_A - 1  # t is a placeholder to total cars
                            carids_A.append(t)  # append to list of car ids

                            fa = open('fa.txt', 'w')
                            aa = len(carids_A)
                            fa.write(str(aa))
                            fa.close()
                            print("aa saved :", aa)

                            df_A.at[int(framenumber), str(t)] = [cxx_A[i], cyy_A[i]]  # add centroid to the new car id

                        elif curcxcy[0] and not oldcxcy and not minx_index2 and not miny_index2:
                            # checks if current centroid exists but previous centroid does not
                            # new car to be added in case minx_index2 is empty

                            df_A[str(totalcars_A)] = ""  # create another column with total cars
                            totalcars_A = totalcars_A + 1  # adds another total car the count
                            t = totalcars_A - 1  # t is a placeholder to total cars
                            carids_A.append(t)  # append to list of car ids

                            fa = open('fa.txt', 'w')
                            aa = len(carids_A)
                            fa.write(str(aa))
                            fa.close()
                            print("aa saved :", aa)

                            df_A.at[int(framenumber), str(t)] = [cxx_A[i], cyy_A[i]]  # add centroid to the new car id

            if len(cxx_B):  # if there are centroids in the specified area

                bb = 0

                if not carids_B:  # if carids_B is empty

                    for i in range(len(cxx_B)):  # loops through all centroids

                        carids_B.append(i)  # adds a car id to the empty list carids

                        fb = open('fb.txt', 'w')
                        bb = len(carids_B)
                        fb.write(str(bb))
                        fb.close()
                        print("bb saved :", bb)

                        df_B[str(carids_B[i])] = ""  # adds a column to the dataframe corresponding to a carid

                        # assigns the centroid values to the current frame (row) and carid (column)
                        df_B.at[int(framenumber), str(carids_B[i])] = [cxx_B[i], cyy_B[i]]

                        totalcars_B = carids_B[i] + 1  # adds one count to total cars

                else:  # if there are already car ids

                    dx = np.zeros((len(cxx_B), len(carids_B)))  # new arrays to calculate deltas
                    dy = np.zeros((len(cyy_B), len(carids_B)))  # new arrays to calculate deltas

                    for i in range(len(cxx_B)):  # loops through all centroids

                        for j in range(len(carids_B)):  # loops through all recorded car ids

                            # acquires centroid from previous frame for specific carid
                            oldcxcy = df_B.iloc[int(framenumber - 1)][str(carids_B[j])]

                            # acquires current frame centroid that doesn't necessarily line up with previous frame centroid
                            curcxcy = np.array([cxx_B[i], cyy_B[i]])

                            if not oldcxcy:  # checks if old centroid is empty in case car leaves screen and new car shows

                                continue  # continue to next carid

                            else:  # calculate centroid deltas to compare to current frame position later

                                dx[i, j] = oldcxcy[0] - curcxcy[0]
                                dy[i, j] = oldcxcy[1] - curcxcy[1]

                    for j in range(len(carids_B)):  # loops through all current car ids

                        sumsum = np.abs(dx[:, j]) + np.abs(dy[:, j])  # sums the deltas wrt to car ids

                        # finds which index carid had the min difference and this is true index
                        correctindextrue = np.argmin(np.abs(sumsum))
                        minx_index = correctindextrue
                        miny_index = correctindextrue

                        # acquires delta values of the minimum deltas in order to check if it is within radius later on
                        mindx = dx[minx_index, j]
                        mindy = dy[miny_index, j]

                        if mindx == 0 and mindy == 0 and np.all(dx[:, j] == 0) and np.all(dy[:, j] == 0):
                            # checks if minimum value is 0 and checks if all deltas are zero since this is empty set
                            # delta could be zero if centroid didn't move

                            continue  # continue to next carid

                        else:

                            # if delta values are less than maximum radius then add that centroid to that specific carid
                            if np.abs(mindx) < maxrad and np.abs(mindy) < maxrad:
                                # adds centroid to corresponding previously existing carid
                                df_B.at[int(framenumber), str(carids_B[j])] = [cxx_B[minx_index], cyy_B[miny_index]]
                                minx_index2.append(
                                    minx_index)  # appends all the indices that were added to previous carids_B
                                miny_index2.append(miny_index)

                    for i in range(len(cxx_B)):  # loops through all centroids

                        # if centroid is not in the minindex list then another car needs to be added
                        if i not in minx_index2 and miny_index2:

                            df_B[str(totalcars_B)] = ""  # create another column with total cars
                            totalcars_B = totalcars_B + 1  # adds another total car the count
                            t = totalcars_B - 1  # t is a placeholder to total cars
                            carids_B.append(t)  # append to list of car ids

                            fb = open('fb.txt', 'w')
                            bb = len(carids_B)
                            fb.write(str(bb))
                            fb.close()
                            print("bb saved :", bb)

                            df_B.at[int(framenumber), str(t)] = [cxx_B[i], cyy_B[i]]  # add centroid to the new car id

                        elif curcxcy[0] and not oldcxcy and not minx_index2 and not miny_index2:
                            # checks if current centroid exists but previous centroid does not
                            # new car to be added in case minx_index2 is empty

                            df_B[str(totalcars_B)] = ""  # create another column with total cars
                            totalcars_B = totalcars_B + 1  # adds another total car the count
                            t = totalcars_B - 1  # t is a placeholder to total cars
                            carids_B.append(t)  # append to list of car ids

                            fb = open('fb.txt', 'w')
                            bb = len(carids_B)
                            fb.write(str(bb))
                            fb.close()
                            print("bb saved :", bb)

                            df_B.at[int(framenumber), str(t)] = [cxx_B[i], cyy_B[i]]  # add centroid to the new car id

            if len(cxx_C):  # if there are centroids in the specified area

                cc = 0

                if not carids_C:  # if carids_C is empty

                    for i in range(len(cxx_C)):  # loops through all centroids

                        carids_C.append(i)  # adds a car id to the empty list carids

                        fc = open('fc.txt', 'w')
                        cc = len(carids_C)
                        fc.write(str(cc))
                        fc.close()
                        print("cc saved :", cc)

                        df_C[str(carids_C[i])] = ""  # adds a column to the dataframe corresponding to a carid

                        # assigns the centroid values to the current frame (row) and carid (column)
                        df_C.at[int(framenumber), str(carids_C[i])] = [cxx_C[i], cyy_C[i]]

                        totalcars_C = carids_C[i] + 1  # adds one count to total cars

                else:  # if there are already car ids

                    dx = np.zeros((len(cxx_C), len(carids_C)))  # new arrays to calculate deltas
                    dy = np.zeros((len(cyy_C), len(carids_C)))  # new arrays to calculate deltas

                    for i in range(len(cxx_C)):  # loops through all centroids

                        for j in range(len(carids_C)):  # loops through all recorded car ids

                            # acquires centroid from previous frame for specific carid
                            oldcxcy = df_C.iloc[int(framenumber - 1)][str(carids_C[j])]

                            # acquires current frame centroid that doesn't necessarily line up with previous frame centroid
                            curcxcy = np.array([cxx_C[i], cyy_C[i]])

                            if not oldcxcy:  # checks if old centroid is empty in case car leaves screen and new car shows

                                continue  # continue to next carid

                            else:  # calculate centroid deltas to compare to current frame position later

                                dx[i, j] = oldcxcy[0] - curcxcy[0]
                                dy[i, j] = oldcxcy[1] - curcxcy[1]

                    for j in range(len(carids_C)):  # loops through all current car ids

                        sumsum = np.abs(dx[:, j]) + np.abs(dy[:, j])  # sums the deltas wrt to car ids

                        # finds which index carid had the min difference and this is true index
                        correctindextrue = np.argmin(np.abs(sumsum))
                        minx_index = correctindextrue
                        miny_index = correctindextrue

                        # acquires delta values of the minimum deltas in order to check if it is within radius later on
                        mindx = dx[minx_index, j]
                        mindy = dy[miny_index, j]

                        if mindx == 0 and mindy == 0 and np.all(dx[:, j] == 0) and np.all(dy[:, j] == 0):
                            # checks if minimum value is 0 and checks if all deltas are zero since this is empty set
                            # delta could be zero if centroid didn't move

                            continue  # continue to next carid

                        else:

                            # if delta values are less than maximum radius then add that centroid to that specific carid
                            if np.abs(mindx) < maxrad and np.abs(mindy) < maxrad:
                                # adds centroid to corresponding previously existing carid
                                df_C.at[int(framenumber), str(carids_C[j])] = [cxx_C[minx_index], cyy_C[miny_index]]
                                minx_index2.append(
                                    minx_index)  # appends all the indices that were added to previous carids_B
                                miny_index2.append(miny_index)

                    for i in range(len(cxx_C)):  # loops through all centroids

                        # if centroid is not in the minindex list then another car needs to be added
                        if i not in minx_index2 and miny_index2:

                            df_C[str(totalcars_C)] = ""  # create another column with total cars
                            totalcars_C = totalcars_C + 1  # adds another total car the count
                            t = totalcars_C - 1  # t is a placeholder to total cars
                            carids_C.append(t)  # append to list of car ids

                            fc = open('fc.txt', 'w')
                            cc = len(carids_C)
                            fc.write(str(cc))
                            fc.close()
                            print("cc saved :", cc)

                            df_C.at[int(framenumber), str(t)] = [cxx_C[i], cyy_C[i]]  # add centroid to the new car id

                        elif curcxcy[0] and not oldcxcy and not minx_index2 and not miny_index2:
                            # checks if current centroid exists but previous centroid does not
                            # new car to be added in case minx_index2 is empty

                            df_C[str(totalcars_C)] = ""  # create another column with total cars
                            totalcars_C = totalcars_C + 1  # adds another total car the count
                            t = totalcars_C - 1  # t is a placeholder to total cars
                            carids_C.append(t)  # append to list of car ids

                            fc = open('fc.txt', 'w')
                            cc = len(carids_C)
                            fc.write(str(cc))
                            fc.close()
                            print("cc saved :", cc)

                            df_C.at[int(framenumber), str(t)] = [cxx_C[i], cyy_C[i]]  # add centroid to the new car id

            if len(cxx_D):  # if there are centroids in the specified area

                dd = 0

                if not carids_D:  # if carids_C is empty

                    for i in range(len(cxx_D)):  # loops through all centroids

                        carids_D.append(i)  # adds a car id to the empty list carids

                        fd = open('fd.txt', 'w')
                        dd = len(carids_D)
                        fd.write(str(dd))
                        fd.close()
                        print("dd saved :", dd)

                        df_D[str(carids_D[i])] = ""  # adds a column to the dataframe corresponding to a carid

                        # assigns the centroid values to the current frame (row) and carid (column)
                        df_D.at[int(framenumber), str(carids_D[i])] = [cxx_D[i], cyy_D[i]]

                        totalcars_D = carids_D[i] + 1  # adds one count to total cars

                else:  # if there are already car ids

                    dx = np.zeros((len(cxx_D), len(carids_D)))  # new arrays to calculate deltas
                    dy = np.zeros((len(cyy_D), len(carids_D)))  # new arrays to calculate deltas

                    for i in range(len(cxx_D)):  # loops through all centroids

                        for j in range(len(carids_D)):  # loops through all recorded car ids

                            # acquires centroid from previous frame for specific carid
                            oldcxcy = df_D.iloc[int(framenumber - 1)][str(carids_D[j])]

                            # acquires current frame centroid that doesn't necessarily line up with previous frame centroid
                            curcxcy = np.array([cxx_D[i], cyy_D[i]])

                            if not oldcxcy:  # checks if old centroid is empty in case car leaves screen and new car shows

                                continue  # continue to next carid

                            else:  # calculate centroid deltas to compare to current frame position later

                                dx[i, j] = oldcxcy[0] - curcxcy[0]
                                dy[i, j] = oldcxcy[1] - curcxcy[1]

                    for j in range(len(carids_D)):  # loops through all current car ids

                        sumsum = np.abs(dx[:, j]) + np.abs(dy[:, j])  # sums the deltas wrt to car ids

                        # finds which index carid had the min difference and this is true index
                        correctindextrue = np.argmin(np.abs(sumsum))
                        minx_index = correctindextrue
                        miny_index = correctindextrue

                        # acquires delta values of the minimum deltas in order to check if it is within radius later on
                        mindx = dx[minx_index, j]
                        mindy = dy[miny_index, j]

                        if mindx == 0 and mindy == 0 and np.all(dx[:, j] == 0) and np.all(dy[:, j] == 0):
                            # checks if minimum value is 0 and checks if all deltas are zero since this is empty set
                            # delta could be zero if centroid didn't move

                            continue  # continue to next carid

                        else:

                            # if delta values are less than maximum radius then add that centroid to that specific carid
                            if np.abs(mindx) < maxrad and np.abs(mindy) < maxrad:
                                # adds centroid to corresponding previously existing carid
                                df_D.at[int(framenumber), str(carids_D[j])] = [cxx_D[minx_index], cyy_D[miny_index]]
                                minx_index2.append(
                                    minx_index)  # appends all the indices that were added to previous carids_B
                                miny_index2.append(miny_index)

                    for i in range(len(cxx_D)):  # loops through all centroids

                        # if centroid is not in the minindex list then another car needs to be added
                        if i not in minx_index2 and miny_index2:

                            df_D[str(totalcars_D)] = ""  # create another column with total cars
                            totalcars_D = totalcars_D + 1  # adds another total car the count
                            t = totalcars_D - 1  # t is a placeholder to total cars
                            carids_D.append(t)  # append to list of car ids

                            fd = open('fd.txt', 'w')
                            dd = len(carids_D)
                            fd.write(str(dd))
                            fd.close()
                            print("dd saved :", dd)

                            df_D.at[int(framenumber), str(t)] = [cxx_D[i], cyy_D[i]]  # add centroid to the new car id

                        elif curcxcy[0] and not oldcxcy and not minx_index2 and not miny_index2:
                            # checks if current centroid exists but previous centroid does not
                            # new car to be added in case minx_index2 is empty

                            df_D[str(totalcars_D)] = ""  # create another column with total cars
                            totalcars_D = totalcars_D + 1  # adds another total car the count
                            t = totalcars_D - 1  # t is a placeholder to total cars
                            carids_D.append(t)  # append to list of car ids

                            fd = open('fd.txt', 'w')
                            dd = len(carids_D)
                            fd.write(str(dd))
                            fd.close()
                            print("dd saved :", dd)

                            df_D.at[int(framenumber), str(t)] = [cxx_D[i], cyy_D[i]]  # add centroid to the new car id

            if len(cxx_PA):  # if there are centroids in the specified area

                if not carids_PA:  # if carids_C is empty

                    for i in range(len(cxx_PA)):  # loops through all centroids

                        carids_PA.append(i)  # adds a car id to the empty list carids

                        fa = open('fa.txt', 'w')
                        aa = len(carids_A) - len(carids_PA)
                        if aa < 0 :
                            aa = 0
                        fa.write(str(aa))
                        fa.close()
                        print("aa saved :", aa)

                        df_PA[str(carids_PA[i])] = ""  # adds a column to the dataframe corresponding to a carid

                        # assigns the centroid values to the current frame (row) and carid (column)
                        df_PA.at[int(framenumber), str(carids_PA[i])] = [cxx_PA[i], cyy_PA[i]]

                        totalcars_PA = carids_PA[i] + 1  # adds one count to total cars

                else:  # if there are already car ids

                    dx = np.zeros((len(cxx_PA), len(carids_PA)))  # new arrays to calculate deltas
                    dy = np.zeros((len(cyy_PA), len(carids_PA)))  # new arrays to calculate deltas

                    for i in range(len(cxx_PA)):  # loops through all centroids

                        for j in range(len(carids_PA)):  # loops through all recorded car ids

                            # acquires centroid from previous frame for specific carid
                            oldcxcy = df_PA.iloc[int(framenumber - 1)][str(carids_PA[j])]

                            # acquires current frame centroid that doesn't necessarily line up with previous frame centroid
                            curcxcy = np.array([cxx_PA[i], cyy_PA[i]])

                            if not oldcxcy:  # checks if old centroid is empty in case car leaves screen and new car shows

                                continue  # continue to next carid

                            else:  # calculate centroid deltas to compare to current frame position later

                                dx[i, j] = oldcxcy[0] - curcxcy[0]
                                dy[i, j] = oldcxcy[1] - curcxcy[1]

                    for j in range(len(carids_PA)):  # loops through all current car ids

                        sumsum = np.abs(dx[:, j]) + np.abs(dy[:, j])  # sums the deltas wrt to car ids

                        # finds which index carid had the min difference and this is true index
                        correctindextrue = np.argmin(np.abs(sumsum))
                        minx_index = correctindextrue
                        miny_index = correctindextrue

                        # acquires delta values of the minimum deltas in order to check if it is within radius later on
                        mindx = dx[minx_index, j]
                        mindy = dy[miny_index, j]

                        if mindx == 0 and mindy == 0 and np.all(dx[:, j] == 0) and np.all(dy[:, j] == 0):
                            # checks if minimum value is 0 and checks if all deltas are zero since this is empty set
                            # delta could be zero if centroid didn't move

                            continue  # continue to next carid

                        else:

                            # if delta values are less than maximum radius then add that centroid to that specific carid
                            if np.abs(mindx) < maxrad and np.abs(mindy) < maxrad:
                                # adds centroid to corresponding previously existing carid
                                df_PA.at[int(framenumber), str(carids_PA[j])] = [cxx_PA[minx_index], cyy_PA[miny_index]]
                                minx_index2.append(
                                    minx_index)  # appends all the indices that were added to previous carids_B
                                miny_index2.append(miny_index)

                    for i in range(len(cxx_PA)):  # loops through all centroids

                        # if centroid is not in the minindex list then another car needs to be added
                        if i not in minx_index2 and miny_index2:

                            df_PA[str(totalcars_PA)] = ""  # create another column with total cars
                            totalcars_PA = totalcars_PA + 1  # adds another total car the count
                            t = totalcars_PA - 1  # t is a placeholder to total cars
                            carids_PA.append(t)  # append to list of car ids

                            fa = open('fa.txt', 'w')
                            aa = len(carids_A) - len(carids_PA)
                            if aa < 0:
                                aa = 0
                            fa.write(str(aa))
                            fa.close()
                            print("aa saved :", aa)

                            df_PA.at[int(framenumber), str(t)] = [cxx_PA[i],
                                                                  cyy_PA[i]]  # add centroid to the new car id

                        elif curcxcy[0] and not oldcxcy and not minx_index2 and not miny_index2:
                            # checks if current centroid exists but previous centroid does not
                            # new car to be added in case minx_index2 is empty

                            df_PA[str(totalcars_PA)] = ""  # create another column with total cars
                            totalcars_PA = totalcars_PA + 1  # adds another total car the count
                            t = totalcars_PA - 1  # t is a placeholder to total cars
                            carids_PA.append(t)  # append to list of car ids

                            fa = open('fa.txt', 'w')
                            aa = len(carids_A) - len(carids_PA)
                            if aa < 0:
                                aa = 0
                            fa.write(str(aa))
                            fa.close()
                            print("aa saved :", aa)

                            df_PA.at[int(framenumber), str(t)] = [cxx_PA[i],
                                                                  cyy_PA[i]]  # add centroid to the new car id

            if len(cxx_PB):
                # if there are centroids in the specified area

                if not carids_PB:  # if carids_C is empty

                    for i in range(len(cxx_PB)):  # loops through all centroids

                        carids_PB.append(i)  # adds a car id to the empty list carids

                        fb = open('fb.txt', 'w')
                        bb = len(carids_B) - len(carids_PB)
                        if bb < 0 :
                            bb = 0
                        fb.write(str(bb))
                        fb.close()
                        print("bb saved :", bb)

                        df_PB[str(carids_PB[i])] = ""  # adds a column to the dataframe corresponding to a carid

                        # assigns the centroid values to the current frame (row) and carid (column)
                        df_PB.at[int(framenumber), str(carids_PB[i])] = [cxx_PB[i], cyy_PB[i]]

                        totalcars_PB = carids_PB[i] + 1  # adds one count to total cars

                else:  # if there are already car ids

                    dx = np.zeros((len(cxx_PB), len(carids_PB)))  # new arrays to calculate deltas
                    dy = np.zeros((len(cyy_PB), len(carids_PB)))  # new arrays to calculate deltas

                    for i in range(len(cxx_PB)):  # loops through all centroids

                        for j in range(len(carids_PB)):  # loops through all recorded car ids

                            # acquires centroid from previous frame for specific carid
                            oldcxcy = df_PB.iloc[int(framenumber - 1)][str(carids_PB[j])]

                            # acquires current frame centroid that doesn't necessarily line up with previous frame centroid
                            curcxcy = np.array([cxx_PB[i], cyy_PB[i]])

                            if not oldcxcy:  # checks if old centroid is empty in case car leaves screen and new car shows

                                continue  # continue to next carid

                            else:  # calculate centroid deltas to compare to current frame position later

                                dx[i, j] = oldcxcy[0] - curcxcy[0]
                                dy[i, j] = oldcxcy[1] - curcxcy[1]

                    for j in range(len(carids_PB)):  # loops through all current car ids

                        sumsum = np.abs(dx[:, j]) + np.abs(dy[:, j])  # sums the deltas wrt to car ids

                        # finds which index carid had the min difference and this is true index
                        correctindextrue = np.argmin(np.abs(sumsum))
                        minx_index = correctindextrue
                        miny_index = correctindextrue

                        # acquires delta values of the minimum deltas in order to check if it is within radius later on
                        mindx = dx[minx_index, j]
                        mindy = dy[miny_index, j]

                        if mindx == 0 and mindy == 0 and np.all(dx[:, j] == 0) and np.all(dy[:, j] == 0):
                            # checks if minimum value is 0 and checks if all deltas are zero since this is empty set
                            # delta could be zero if centroid didn't move

                            continue  # continue to next carid

                        else:

                            # if delta values are less than maximum radius then add that centroid to that specific carid
                            if np.abs(mindx) < maxrad and np.abs(mindy) < maxrad:
                                # adds centroid to corresponding previously existing carid
                                df_PB.at[int(framenumber), str(carids_PB[j])] = [cxx_PB[minx_index], cyy_PB[miny_index]]
                                minx_index2.append(
                                    minx_index)  # appends all the indices that were added to previous carids_B
                                miny_index2.append(miny_index)

                    for i in range(len(cxx_PB)):  # loops through all centroids

                        # if centroid is not in the minindex list then another car needs to be added
                        if i not in minx_index2 and miny_index2:

                            df_PB[str(totalcars_PB)] = ""  # create another column with total cars
                            totalcars_PB = totalcars_PB + 1  # adds another total car the count
                            t = totalcars_PB - 1  # t is a placeholder to total cars
                            carids_PB.append(t)  # append to list of car ids

                            fb = open('fb.txt', 'w')
                            bb = len(carids_B) - len(carids_PB)
                            if bb < 0:
                                bb = 0
                            fb.write(str(bb))
                            fb.close()
                            print("bb saved :", bb)

                            df_PB.at[int(framenumber), str(t)] = [cxx_PB[i],
                                                                  cyy_PB[i]]  # add centroid to the new car id

                        elif curcxcy[0] and not oldcxcy and not minx_index2 and not miny_index2:
                            # checks if current centroid exists but previous centroid does not
                            # new car to be added in case minx_index2 is empty

                            df_PB[str(totalcars_PB)] = ""  # create another column with total cars
                            totalcars_PB = totalcars_PB + 1  # adds another total car the count
                            t = totalcars_PB - 1  # t is a placeholder to total cars
                            carids_PB.append(t)  # append to list of car ids

                            fb = open('fb.txt', 'w')
                            bb = len(carids_B) - len(carids_PB)
                            if bb < 0:
                                bb = 0
                            fb.write(str(bb))
                            fb.close()
                            print("bb saved :", bb)

                            df_PB.at[int(framenumber), str(t)] = [cxx_PB[i],
                                                                  cyy_PB[i]]  # add centroid to the new car id

            if len(cxx_PC):  # if there are centroids in the specified area

                if not carids_PC:  # if carids_C is empty

                    for i in range(len(cxx_PC)):  # loops through all centroids

                        carids_PC.append(i)  # adds a car id to the empty list carids

                        fc = open('fc.txt', 'w')
                        cc = len(carids_C) - len(carids_PC)
                        if cc < 0 :
                            cc = 0
                        fc.write(str(cc))
                        fc.close()
                        print("cc saved :", cc)

                        df_PC[str(carids_PC[i])] = ""  # adds a column to the dataframe corresponding to a carid

                        # assigns the centroid values to the current frame (row) and carid (column)
                        df_PC.at[int(framenumber), str(carids_PC[i])] = [cxx_PC[i], cyy_PC[i]]

                        totalcars_PC = carids_PC[i] + 1  # adds one count to total cars

                else:  # if there are already car ids

                    dx = np.zeros((len(cxx_PC), len(carids_PC)))  # new arrays to calculate deltas
                    dy = np.zeros((len(cyy_PC), len(carids_PC)))  # new arrays to calculate deltas

                    for i in range(len(cxx_PC)):  # loops through all centroids

                        for j in range(len(carids_PC)):  # loops through all recorded car ids

                            # acquires centroid from previous frame for specific carid
                            oldcxcy = df_PC.iloc[int(framenumber - 1)][str(carids_PC[j])]

                            # acquires current frame centroid that doesn't necessarily line up with previous frame centroid
                            curcxcy = np.array([cxx_PC[i], cyy_PC[i]])

                            if not oldcxcy:  # checks if old centroid is empty in case car leaves screen and new car shows

                                continue  # continue to next carid

                            else:  # calculate centroid deltas to compare to current frame position later

                                dx[i, j] = oldcxcy[0] - curcxcy[0]
                                dy[i, j] = oldcxcy[1] - curcxcy[1]

                    for j in range(len(carids_PC)):  # loops through all current car ids

                        sumsum = np.abs(dx[:, j]) + np.abs(dy[:, j])  # sums the deltas wrt to car ids

                        # finds which index carid had the min difference and this is true index
                        correctindextrue = np.argmin(np.abs(sumsum))
                        minx_index = correctindextrue
                        miny_index = correctindextrue

                        # acquires delta values of the minimum deltas in order to check if it is within radius later on
                        mindx = dx[minx_index, j]
                        mindy = dy[miny_index, j]

                        if mindx == 0 and mindy == 0 and np.all(dx[:, j] == 0) and np.all(dy[:, j] == 0):
                            # checks if minimum value is 0 and checks if all deltas are zero since this is empty set
                            # delta could be zero if centroid didn't move

                            continue  # continue to next carid

                        else:

                            # if delta values are less than maximum radius then add that centroid to that specific carid
                            if np.abs(mindx) < maxrad and np.abs(mindy) < maxrad:
                                # adds centroid to corresponding previously existing carid
                                df_PC.at[int(framenumber), str(carids_PC[j])] = [cxx_PC[minx_index], cyy_PC[miny_index]]
                                minx_index2.append(
                                    minx_index)  # appends all the indices that were added to previous carids_C
                                miny_index2.append(miny_index)

                    for i in range(len(cxx_PC)):  # loops through all centroids

                        # if centroid is not in the minindex list then another car needs to be added
                        if i not in minx_index2 and miny_index2:

                            df_PC[str(totalcars_PC)] = ""  # create another column with total cars
                            totalcars_PC = totalcars_PC + 1  # adds another total car the count
                            t = totalcars_PC - 1  # t is a placeholder to total cars
                            carids_PC.append(t)  # append to list of car ids

                            fc = open('fc.txt', 'w')
                            cc = len(carids_C) - len(carids_PC)
                            if cc < 0:
                                cc = 0
                            fc.write(str(cc))
                            fc.close()
                            print("cc saved :", cc)

                            df_PC.at[int(framenumber), str(t)] = [cxx_PC[i],
                                                                  cyy_PC[i]]  # add centroid to the new car id

                        elif curcxcy[0] and not oldcxcy and not minx_index2 and not miny_index2:
                            # checks if current centroid exists but previous centroid does not
                            # new car to be added in case minx_index2 is empty

                            df_PC[str(totalcars_PC)] = ""  # create another column with total cars
                            totalcars_PC = totalcars_PC + 1  # adds another total car the count
                            t = totalcars_PC - 1  # t is a placeholder to total cars
                            carids_PC.append(t)  # append to list of car ids

                            fc = open('fc.txt', 'w')
                            cc = len(carids_C) - len(carids_PC)
                            if cc < 0:
                                cc = 0
                            fc.write(str(cc))
                            fc.close()
                            print("cc saved :", cc)

                            df_PC.at[int(framenumber), str(t)] = [cxx_PC[i],
                                                                  cyy_PC[i]]  # add centroid to the new car id

            if len(cxx_PD):  # if there are centroids in the specified area

                if not carids_PD:  # if carids_D is empty

                    for i in range(len(cxx_PD)):  # loops through all centroids

                        carids_PD.append(i)  # adds a car id to the empty list carids

                        fd = open('fd.txt', 'w')
                        dd = len(carids_D) - len(carids_PD)
                        if dd < 0 :
                            dd = 0
                        fd.write(str(dd))
                        fd.close()
                        print("dd saved :", dd)

                        df_PD[str(carids_PD[i])] = ""  # adds a column to the dataframe corresponding to a carid

                        # assigns the centroid values to the current frame (row) and carid (column)
                        df_PD.at[int(framenumber), str(carids_PD[i])] = [cxx_PD[i], cyy_PD[i]]

                        totalcars_PD = carids_PD[i] + 1  # adds one count to total cars

                else:  # if there are already car ids

                    dx = np.zeros((len(cxx_PD), len(carids_PD)))  # new arrays to calculate deltas
                    dy = np.zeros((len(cyy_PD), len(carids_PD)))  # new arrays to calculate deltas

                    for i in range(len(cxx_PD)):  # loops through all centroids

                        for j in range(len(carids_PD)):  # loops through all recorded car ids

                            # acquires centroid from previous frame for specific carid
                            oldcxcy = df_PD.iloc[int(framenumber - 1)][str(carids_PD[j])]

                            # acquires current frame centroid that doesn't necessarily line up with previous frame centroid
                            curcxcy = np.array([cxx_PD[i], cyy_PD[i]])

                            if not oldcxcy:  # checks if old centroid is empty in case car leaves screen and new car shows

                                continue  # continue to next carid

                            else:  # calculate centroid deltas to compare to current frame position later

                                dx[i, j] = oldcxcy[0] - curcxcy[0]
                                dy[i, j] = oldcxcy[1] - curcxcy[1]

                    for j in range(len(carids_PD)):  # loops through all current car ids

                        sumsum = np.abs(dx[:, j]) + np.abs(dy[:, j])  # sums the deltas wrt to car ids

                        # finds which index carid had the min difference and this is true index
                        correctindextrue = np.argmin(np.abs(sumsum))
                        minx_index = correctindextrue
                        miny_index = correctindextrue

                        # acquires delta values of the minimum deltas in order to check if it is within radius later on
                        mindx = dx[minx_index, j]
                        mindy = dy[miny_index, j]

                        if mindx == 0 and mindy == 0 and np.all(dx[:, j] == 0) and np.all(dy[:, j] == 0):
                            # checks if minimum value is 0 and checks if all deltas are zero since this is empty set
                            # delta could be zero if centroid didn't move

                            continue  # continue to next carid

                        else:

                            # if delta values are less than maximum radius then add that centroid to that specific carid
                            if np.abs(mindx) < maxrad and np.abs(mindy) < maxrad:
                                # adds centroid to corresponding previously existing carid
                                df_PD.at[int(framenumber), str(carids_PD[j])] = [cxx_PD[minx_index], cyy_PD[miny_index]]
                                minx_index2.append(
                                    minx_index)  # appends all the indices that were added to previous carids_D
                                miny_index2.append(miny_index)

                    for i in range(len(cxx_PD)):  # loops through all centroids

                        # if centroid is not in the minindex list then another car needs to be added
                        if i not in minx_index2 and miny_index2:

                            df_PD[str(totalcars_PD)] = ""  # create another column with total cars
                            totalcars_PD = totalcars_PD + 1  # adds another total car the count
                            t = totalcars_PD - 1  # t is a placeholder to total cars
                            carids_PD.append(t)  # append to list of car ids

                            fd = open('fd.txt', 'w')
                            dd = len(carids_D) - len(carids_PD)
                            if dd < 0:
                                dd = 0
                            fd.write(str(dd))
                            fd.close()
                            print("dd saved :", dd)

                            df_PD.at[int(framenumber), str(t)] = [cxx_PD[i],
                                                                  cyy_PD[i]]  # add centroid to the new car id

                        elif curcxcy[0] and not oldcxcy and not minx_index2 and not miny_index2:
                            # checks if current centroid exists but previous centroid does not
                            # new car to be added in case minx_index2 is empty

                            df_PD[str(totalcars_PD)] = ""  # create another column with total cars
                            totalcars_PD = totalcars_PD + 1  # adds another total car the count
                            t = totalcars_PD - 1  # t is a placeholder to total cars
                            carids_PD.append(t)  # append to list of car ids

                            fd = open('fd.txt', 'w')
                            dd = len(carids_D) - len(carids_PD)
                            if dd < 0:
                                dd = 0
                            fd.write(str(dd))
                            fd.close()
                            print("dd saved :", dd)

                            df_PD.at[int(framenumber), str(t)] = [cxx_PD[i],
                                                                  cyy_PD[i]]  # add centroid to the new car id

            ########################### The section below labels the centroids on screen
            ######################AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
            currentcars_A = 0  # current cars on screen
            currentcarsindex_A = []  # current cars on screen carid index

            for i in range(len(carids_A)):  # loops through all carids_B

                if df_A.at[int(framenumber), str(carids_A[i])] != '':
                    # checks the current frame to see which car ids are active
                    # by checking in centroid exists on current frame for certain car id

                    currentcars_A = currentcars_A + 1  # adds another to current cars on screen
                    currentcarsindex_A.append(i)  # adds car ids to current cars on screen

            for i in range(currentcars_A):  # loops through all current car ids on screen

                # grabs centroid of certain carid for current frame
                curcent_A = df_A.iloc[int(framenumber)][str(carids_A[currentcarsindex_A[i]])]

                # grabs centroid of certain carid for previous frame
                oldcent_A = df_A.iloc[int(framenumber - 1)][str(carids_A[currentcarsindex_A[i]])]

                if curcent_A:  # if there is a current centroid

                    # On-screen text for current centroid
                    cv2.putText(image, "Centroid" + str(curcent_A[0]) + "," + str(curcent_A[1]),
                                (int(curcent_A[0]), int(curcent_A[1])), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)

                    cv2.putText(image, "ID:" + str(carids_A[currentcarsindex_A[i]]),
                                (int(curcent_A[0]), int(curcent_A[1] - 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)

                    cv2.drawMarker(image, (int(curcent_A[0]), int(curcent_A[1])), (0, 0, 255), cv2.MARKER_STAR,
                                   markerSize=5,
                                   thickness=1, line_type=cv2.LINE_AA)

                    if oldcent_A:  # checks if old centroid exists
                        # adds radius box from previous centroid to current centroid for visualization
                        xstart = oldcent_A[0] - maxrad
                        ystart = oldcent_A[1] - maxrad
                        xwidth = oldcent_A[0] + maxrad
                        yheight = oldcent_A[1] + maxrad
                        cv2.rectangle(image, (int(xstart), int(ystart)), (int(xwidth), int(yheight)), (0, 125, 0), 1)

                        # checks if old centroid is on or below line and curcent is on or above line
                        # to count cars and that car hasn't been counted yet
                        if oldcent_A[0] >= CounterA_check and curcent_A[0] <= CounterA_check and \
                                carids_A[currentcarsindex_A[i]] not in caridspassed_A:

                            cars_pass_A_u = cars_pass_A_u + 1

                            cv2.line(image, (CounterA_check, 0), (CounterA_check, height), (0, 0, 255), 5)
                            caridspassed_A.append(currentcarsindex_A[i])
                            # adds car id to list of count cars to prevent double counting

                        # checks if old centroid is on or above line and curcent is on or below line
                        # to count cars and that car hasn't been counted yet

                        elif oldcent_A[0] <= CounterA_check and curcent_A[0] >= CounterA_check and \
                                carids_A[currentcarsindex_A[i]] not in caridspassed_A:

                            cars_pass_A_d = cars_pass_A_d + 1

                            cv2.line(image, (CounterA_check, 0), (CounterA_check, height), (0, 0, 125), 5)
                            caridspassed_A.append(currentcarsindex_A[i])

            ######################BBBBBBBBBBBBBBBBBBBBBBBB
            currentcars_B = 0
            currentcarsindex_B = []

            for i in range(len(carids_B)):

                if df_B.at[int(framenumber), str(carids_B[i])] != '':
                    currentcars_B = currentcars_B + 1  # adds another to current cars on screen
                    currentcarsindex_B.append(i)  # adds car ids to current cars on screen

            for i in range(currentcars_B):

                curcent_B = df_B.iloc[int(framenumber)][str(carids_B[currentcarsindex_B[i]])]
                oldcent_B = df_B.iloc[int(framenumber - 1)][str(carids_B[currentcarsindex_B[i]])]

                if curcent_B:

                    cv2.putText(image, "Centroid" + str(curcent_B[0]) + "," + str(curcent_B[1]),
                                (int(curcent_B[0]), int(curcent_B[1])), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)

                    cv2.putText(image, "ID:" + str(carids_B[currentcarsindex_B[i]]),
                                (int(curcent_B[0]), int(curcent_B[1] - 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)

                    cv2.drawMarker(image, (int(curcent_B[0]), int(curcent_B[1])), (0, 0, 255), cv2.MARKER_STAR,
                                   markerSize=5,
                                   thickness=1, line_type=cv2.LINE_AA)

                    if oldcent_B:

                        xstart = oldcent_B[0] - maxrad
                        ystart = oldcent_B[1] - maxrad
                        xwidth = oldcent_B[0] + maxrad
                        yheight = oldcent_B[1] + maxrad
                        cv2.rectangle(image, (int(xstart), int(ystart)), (int(xwidth), int(yheight)), (0, 125, 0),
                                      1)

                        if oldcent_B[0] >= CounterB_check and curcent_B[0] <= CounterB_check and carids_B[
                            currentcarsindex_B[i]] not in caridspassed_B:

                            cars_pass_B_u = cars_pass_B_u + 1

                            cv2.line(image, (CounterB_check, 0), (CounterB_check, height), (0, 0, 255), 5)
                            caridspassed_B.append(currentcarsindex_B[i])

                        elif oldcent_B[0] <= CounterB_check and curcent_B[0] >= CounterB_check and carids_B[
                            currentcarsindex_B[i]] not in caridspassed_B:

                            cars_pass_B_d = cars_pass_B_d + 1

                            cv2.line(image, (CounterB_check, 0), (CounterB_check, height), (0, 0, 125), 5)
                            caridspassed_B.append(currentcarsindex_B[i])

            ######################ccccccccc
            currentcars_C = 0
            currentcarsindex_C = []

            for i in range(len(carids_C)):

                if df_C.at[int(framenumber), str(carids_C[i])] != '':
                    currentcars_C = currentcars_C + 1  # adds another to current cars on screen
                    currentcarsindex_C.append(i)  # adds car ids to current cars on screen

            for i in range(currentcars_C):

                curcent_C = df_C.iloc[int(framenumber)][str(carids_C[currentcarsindex_C[i]])]
                oldcent_C = df_C.iloc[int(framenumber - 1)][str(carids_C[currentcarsindex_C[i]])]

                if curcent_C:

                    cv2.putText(image, "Centroid" + str(curcent_C[0]) + "," + str(curcent_C[1]),
                                (int(curcent_C[0]), int(curcent_C[1])), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)

                    cv2.putText(image, "ID:" + str(carids_C[currentcarsindex_C[i]]),
                                (int(curcent_C[0]), int(curcent_C[1] - 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)

                    cv2.drawMarker(image, (int(curcent_C[0]), int(curcent_C[1])), (0, 0, 255), cv2.MARKER_STAR,
                                   markerSize=5,
                                   thickness=1, line_type=cv2.LINE_AA)

                    if oldcent_C:

                        xstart = oldcent_C[0] - maxrad
                        ystart = oldcent_C[1] - maxrad
                        xwidth = oldcent_C[0] + maxrad
                        yheight = oldcent_C[1] + maxrad
                        cv2.rectangle(image, (int(xstart), int(ystart)), (int(xwidth), int(yheight)), (0, 125, 0),
                                      1)

                        if oldcent_C[1] >= CounterC_check and curcent_C[1] <= CounterC_check and carids_C[
                            currentcarsindex_C[i]] not in caridspassed_C:

                            cars_pass_C_u = cars_pass_C_u + 1

                            cv2.line(image, (0, CounterC_check), (height, CounterC_check), (0, 0, 255), 5)
                            caridspassed_C.append(currentcarsindex_C[i])

                        elif oldcent_C[1] <= CounterC_check and curcent_C[1] >= CounterC_check and carids_C[
                            currentcarsindex_C[i]] not in caridspassed_C:

                            cars_pass_C_d = cars_pass_C_d + 1

                            cv2.line(image, (0, CounterC_check), (height, CounterC_check), (0, 0, 125), 5)
                            caridspassed_C.append(currentcarsindex_C[i])

            ######################DDDDDDDDDDDDDDDDD
            currentcars_D = 0
            currentcarsindex_D = []

            for i in range(len(carids_D)):

                if df_D.at[int(framenumber), str(carids_D[i])] != '':
                    currentcars_D = currentcars_D + 1  # adds another to current cars on screen
                    currentcarsindex_D.append(i)  # adds car ids to current cars on screen

            for i in range(currentcars_D):

                curcent_D = df_D.iloc[int(framenumber)][str(carids_D[currentcarsindex_D[i]])]
                oldcent_D = df_D.iloc[int(framenumber - 1)][str(carids_D[currentcarsindex_D[i]])]

                if curcent_D:

                    cv2.putText(image, "Centroid" + str(curcent_D[0]) + "," + str(curcent_D[1]),
                                (int(curcent_D[0]), int(curcent_D[1])), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255),
                                2)

                    cv2.putText(image, "ID:" + str(carids_D[currentcarsindex_D[i]]),
                                (int(curcent_D[0]), int(curcent_D[1] - 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)

                    cv2.drawMarker(image, (int(curcent_D[0]), int(curcent_D[1])), (0, 0, 255), cv2.MARKER_STAR,
                                   markerSize=5,
                                   thickness=1, line_type=cv2.LINE_AA)

                    if oldcent_D:

                        xstart = oldcent_D[0] - maxrad
                        ystart = oldcent_D[1] - maxrad
                        xwidth = oldcent_D[0] + maxrad
                        yheight = oldcent_D[1] + maxrad
                        cv2.rectangle(image, (int(xstart), int(ystart)), (int(xwidth), int(yheight)), (0, 125, 0),
                                      1)

                        if oldcent_D[1] >= CounterD_check and curcent_D[1] <= CounterD_check and carids_D[
                            currentcarsindex_D[i]] not in caridspassed_D:

                            cars_pass_D_u = cars_pass_D_u + 1

                            cv2.line(image, (0, CounterD_check), (height, CounterD_check), (0, 0, 255), 5)
                            caridspassed_D.append(currentcarsindex_D[i])

                        elif oldcent_D[1] <= CounterD_check and curcent_D[1] >= CounterD_check and carids_D[
                            currentcarsindex_D[i]] not in caridspassed_D:

                            cars_pass_D_d = cars_pass_D_d + 1

                            cv2.line(image, (0, CounterD_check), (height, CounterD_check), (0, 0, 125), 5)
                            caridspassed_D.append(currentcarsindex_D[i])

            ######################PA
            currentcars_PA = 0  # current cars on screen
            currentcarsindex_PA = []  # current cars on screen carid index

            for i in range(len(carids_PA)):  # loops through all carids_B

                if df_PA.at[int(framenumber), str(carids_PA[i])] != '':
                    # checks the current frame to see which car ids are active
                    # by checking in centroid exists on current frame for certain car id

                    currentcars_PA = currentcars_PA + 1  # adds another to current cars on screen
                    currentcarsindex_PA.append(i)  # adds car ids to current cars on screen

            for i in range(currentcars_PA):  # loops through all current car ids on screen

                # grabs centroid of certain carid for current frame
                curcent_PA = df_PA.iloc[int(framenumber)][str(carids_PA[currentcarsindex_PA[i]])]

                # grabs centroid of certain carid for previous frame
                oldcent_PA = df_PA.iloc[int(framenumber - 1)][str(carids_PA[currentcarsindex_PA[i]])]

                if curcent_PA:  # if there is a current centroid

                    # On-screen text for current centroid
                    cv2.putText(image, "Centroid" + str(curcent_PA[0]) + "," + str(curcent_PA[1]),
                                (int(curcent_PA[0]), int(curcent_PA[1])), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255),
                                2)

                    cv2.putText(image, "ID:" + str(carids_PA[currentcarsindex_PA[i]]),
                                (int(curcent_PA[0]), int(curcent_PA[1] - 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)

                    cv2.drawMarker(image, (int(curcent_PA[0]), int(curcent_PA[1])), (0, 0, 255), cv2.MARKER_STAR,
                                   markerSize=5,
                                   thickness=1, line_type=cv2.LINE_AA)

                    if oldcent_PA:  # checks if old centroid exists
                        # adds radius box from previous centroid to current centroid for visualization
                        xstart = oldcent_PA[0] - maxrad
                        ystart = oldcent_PA[1] - maxrad
                        xwidth = oldcent_PA[0] + maxrad
                        yheight = oldcent_PA[1] + maxrad
                        cv2.rectangle(image, (int(xstart), int(ystart)), (int(xwidth), int(yheight)), (0, 125, 0), 1)

                        # checks if old centroid is on or below line and curcent is on or above line
                        # to count cars and that car hasn't been counted yet
                        if oldcent_PA[0] >= Pass_A_check and curcent_PA[0] <= Pass_A_check and carids_PA[
                            currentcarsindex_PA[i]] not in caridspassed_PA:

                            cars_out_A_u = cars_out_A_u + 1

                            cv2.line(image, (Pass_A_check, 0), (Pass_A_check, height), (0, 0, 255), 5)
                            caridspassed_PA.append(currentcarsindex_PA[i])
                            # adds car id to list of count cars to prevent double counting

                        # checks if old centroid is on or above line and curcent is on or below line
                        # to count cars and that car hasn't been counted yet

                        elif oldcent_PA[0] <= Pass_A_check and curcent_PA[0] >= Pass_A_check and carids_PA[
                            currentcarsindex_PA[i]] not in caridspassed_PA:

                            cars_out_A_d = cars_out_A_d + 1

                            cv2.line(image, (Pass_A_check, 0), (Pass_A_check, height), (0, 0, 125), 5)
                            caridspassed_PA.append(currentcarsindex_PA[i])

            ######################PB
            currentcars_PB = 0  # current cars on screen
            currentcarsindex_PB = []  # current cars on screen carid index

            for i in range(len(carids_PB)):  # loops through all carids_B

                if df_PB.at[int(framenumber), str(carids_PB[i])] != '':
                    # checks the current frame to see which car ids are active
                    # by checking in centroid exists on current frame for certain car id

                    currentcars_PB = currentcars_PB + 1  # adds another to current cars on screen
                    currentcarsindex_PB.append(i)  # adds car ids to current cars on screen

            for i in range(currentcars_PB):  # loops through all current car ids on screen

                # grabs centroid of certain carid for current frame
                curcent_PB = df_PB.iloc[int(framenumber)][str(carids_PB[currentcarsindex_PB[i]])]

                # grabs centroid of certain carid for previous frame
                oldcent_PB = df_PB.iloc[int(framenumber - 1)][str(carids_PB[currentcarsindex_PB[i]])]

                if curcent_PB:  # if there is a current centroid

                    # On-screen text for current centroid
                    cv2.putText(image, "Centroid" + str(curcent_PB[0]) + "," + str(curcent_PB[1]),
                                (int(curcent_PB[0]), int(curcent_PB[1])), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255),
                                2)

                    cv2.putText(image, "ID:" + str(carids_PB[currentcarsindex_PB[i]]),
                                (int(curcent_PB[0]), int(curcent_PB[1] - 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)

                    cv2.drawMarker(image, (int(curcent_PB[0]), int(curcent_PB[1])), (0, 0, 255), cv2.MARKER_STAR,
                                   markerSize=5,
                                   thickness=1, line_type=cv2.LINE_AA)

                    if oldcent_PB:  # checks if old centroid exists
                        # adds radius box from previous centroid to current centroid for visualization
                        xstart = oldcent_PB[0] - maxrad
                        ystart = oldcent_PB[1] - maxrad
                        xwidth = oldcent_PB[0] + maxrad
                        yheight = oldcent_PB[1] + maxrad
                        cv2.rectangle(image, (int(xstart), int(ystart)), (int(xwidth), int(yheight)), (0, 125, 0), 1)

                        # checks if old centroid is on or below line and curcent is on or above line
                        # to count cars and that car hasn't been counted yet
                        if oldcent_PB[0] >= Pass_B_check and curcent_PB[0] <= Pass_B_check and carids_PB[
                            currentcarsindex_PB[i]] not in caridspassed_PB:

                            cars_out_B_u = cars_out_B_u + 1

                            cv2.line(image, (Pass_B_check, 0), (Pass_B_check, height), (0, 0, 255), 5)
                            caridspassed_PB.append(currentcarsindex_PB[i])
                            # adds car id to list of count cars to prevent double counting

                        # checks if old centroid is on or above line and curcent is on or below line
                        # to count cars and that car hasn't been counted yet

                        elif oldcent_PB[0] <= Pass_B_check and curcent_PB[0] >= Pass_B_check and carids_PB[
                            currentcarsindex_PB[i]] not in caridspassed_PB:

                            cars_out_B_d = cars_out_B_d + 1

                            cv2.line(image, (Pass_B_check, 0), (Pass_B_check, height), (0, 0, 125), 5)
                            caridspassed_PB.append(currentcarsindex_PB[i])

            ######################PC
            currentcars_PC = 0  # current cars on screen
            currentcarsindex_PC = []  # current cars on screen carid index

            for i in range(len(carids_PC)):  # loops through all carids_C

                if df_PC.at[int(framenumber), str(carids_PC[i])] != '':
                    # checks the current frame to see which car ids are active
                    # by checking in centroid exists on current frame for certain car id

                    currentcars_PC = currentcars_PC + 1  # adds another to current cars on screen
                    currentcarsindex_PC.append(i)  # adds car ids to current cars on screen

            for i in range(currentcars_PC):  # loops through all current car ids on screen

                # grabs centroid of certain carid for current frame
                curcent_PC = df_PC.iloc[int(framenumber)][str(carids_PC[currentcarsindex_PC[i]])]

                # grabs centroid of certain carid for previous frame
                oldcent_PC = df_PC.iloc[int(framenumber - 1)][str(carids_PC[currentcarsindex_PC[i]])]

                if curcent_PC:  # if there is a current centroid

                    # On-screen text for current centroid
                    cv2.putText(image, "Centroid" + str(curcent_PC[0]) + "," + str(curcent_PC[1]),
                                (int(curcent_PC[0]), int(curcent_PC[1])), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255),
                                2)

                    cv2.putText(image, "ID:" + str(carids_PC[currentcarsindex_PC[i]]),
                                (int(curcent_PC[0]), int(curcent_PC[1] - 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)

                    cv2.drawMarker(image, (int(curcent_PC[0]), int(curcent_PC[1])), (0, 0, 255), cv2.MARKER_STAR,
                                   markerSize=5,
                                   thickness=1, line_type=cv2.LINE_AA)

                    if oldcent_PC:  # checks if old centroid exists
                        # adds radius box from previous centroid to current centroid for visualization
                        xstart = oldcent_PC[0] - maxrad
                        ystart = oldcent_PC[1] - maxrad
                        xwidth = oldcent_PC[0] + maxrad
                        yheight = oldcent_PC[1] + maxrad
                        cv2.rectangle(image, (int(xstart), int(ystart)), (int(xwidth), int(yheight)), (0, 125, 0), 1)

                        # checks if old centroid is on or below line and curcent is on or above line
                        # to count cars and that car hasn't been counted yet
                        if oldcent_PC[0] >= Pass_C_check and curcent_PC[0] <= Pass_C_check and carids_PC[
                            currentcarsindex_PC[i]] not in caridspassed_PC:

                            cars_out_C_u = cars_out_C_u + 1

                            cv2.line(image, (Pass_C_check, 0), (Pass_C_check, height), (0, 0, 255), 5)
                            caridspassed_PC.append(currentcarsindex_PC[i])
                            # adds car id to list of count cars to prevent double counting

                        # checks if old centroid is on or above line and curcent is on or below line
                        # to count cars and that car hasn't been counted yet

                        elif oldcent_PC[0] <= Pass_C_check and curcent_PC[0] >= Pass_C_check and carids_PC[
                            currentcarsindex_PC[i]] not in caridspassed_PC:

                            cars_out_C_d = cars_out_C_d + 1

                            cv2.line(image, (Pass_C_check, 0), (Pass_C_check, height), (0, 0, 125), 5)
                            caridspassed_PC.append(currentcarsindex_PC[i])

            ######################PD
            currentcars_PD = 0  # current cars on screen
            currentcarsindex_PD = []  # current cars on screen carid index

            for i in range(len(carids_PD)):  # loops through all carids_D

                if df_PD.at[int(framenumber), str(carids_PD[i])] != '':
                    # checks the current frame to see which car ids are active
                    # by checking in centroid exists on current frame for certain car id

                    currentcars_PD = currentcars_PD + 1  # adds another to current cars on screen
                    currentcarsindex_PD.append(i)  # adds car ids to current cars on screen

            for i in range(currentcars_PD):  # loops through all current car ids on screen

                # grabs centroid of certain carid for current frame
                curcent_PD = df_PD.iloc[int(framenumber)][str(carids_PD[currentcarsindex_PD[i]])]

                # grabs centroid of certain carid for previous frame
                oldcent_PD = df_PD.iloc[int(framenumber - 1)][str(carids_PD[currentcarsindex_PD[i]])]

                if curcent_PD:  # if there is a current centroid

                    # On-screen text for current centroid
                    cv2.putText(image, "Centroid" + str(curcent_PD[0]) + "," + str(curcent_PD[1]),
                                (int(curcent_PD[0]), int(curcent_PD[1])), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255),
                                2)

                    cv2.putText(image, "ID:" + str(carids_PD[currentcarsindex_PD[i]]),
                                (int(curcent_PD[0]), int(curcent_PD[1] - 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)

                    cv2.drawMarker(image, (int(curcent_PD[0]), int(curcent_PD[1])), (0, 0, 255), cv2.MARKER_STAR,
                                   markerSize=5,
                                   thickness=1, line_type=cv2.LINE_AA)

                    if oldcent_PD:  # checks if old centroid exists
                        # adds radius box from previous centroid to current centroid for visualization
                        xstart = oldcent_PD[0] - maxrad
                        ystart = oldcent_PD[1] - maxrad
                        xwidth = oldcent_PD[0] + maxrad
                        yheight = oldcent_PD[1] + maxrad
                        cv2.rectangle(image, (int(xstart), int(ystart)), (int(xwidth), int(yheight)), (0, 125, 0), 1)

                        # checks if old centroid is on or below line and curcent is on or above line
                        # to count cars and that car hasn't been counted yet
                        if oldcent_PD[0] >= Pass_D_check and curcent_PD[0] <= Pass_D_check and carids_PD[
                            currentcarsindex_PD[i]] not in caridspassed_PD:

                            cars_out_D_u = cars_out_D_u + 1

                            cv2.line(image, (Pass_D_check, 0), (Pass_D_check, height), (0, 0, 255), 5)
                            caridspassed_PD.append(currentcarsindex_PD[i])
                            # adds car id to list of count cars to prevent double counting

                        # checks if old centroid is on or above line and curcent is on or below line
                        # to count cars and that car hasn't been counted yet

                        elif oldcent_PD[0] <= Pass_D_check and curcent_PD[0] >= Pass_D_check and carids_PD[
                            currentcarsindex_PD[i]] not in caridspassed_PD:

                            cars_out_D_d = cars_out_D_d + 1

                            cv2.line(image, (Pass_D_check, 0), (Pass_D_check, height), (0, 0, 125), 5)
                            caridspassed_PD.append(currentcarsindex_PD[i])

            # Top left hand corner on-screen text
            # CounterA
            cv2.rectangle(image, (0, 0), (130, 55), (255, 0, 0), -1)  # background rectangle for on-screen text
            cv2.putText(image, "In Area A", (0, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 170, 0), 1)
            cv2.putText(image, "Cars: " + str(len(carids_A)), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 170, 0), 1)
            cv2.putText(image, "Cars Crossed: " + str(cars_pass_A_d), (0, 45), cv2.FONT_HERSHEY_SIMPLEX, .5,
                        (0, 170, 0),
                        1)

            # CounterB
            cv2.rectangle(image, (0, 55), (130, 110), (255, 0, 0), -1)  # background rectangle for on-screen text
            cv2.putText(image, "In Area B", (0, 70), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 170, 0), 1)
            cv2.putText(image, "Cars: " + str(len(carids_B)), (0, 85), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 170, 0), 1)
            cv2.putText(image, "Cars Crossed: " + str(cars_pass_B_u), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, .5,
                        (0, 170, 0),
                        1)
            # CounterC
            cv2.rectangle(image, (0, 110), (130, 165), (255, 0, 0), -1)  # background rectangle for on-screen text
            cv2.putText(image, "In Area C", (0, 125), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 170, 0), 1)
            cv2.putText(image, "Cars: " + str(len(carids_C)), (0, 140), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 170, 0), 1)
            cv2.putText(image, "Cars Crossed: " + str(cars_pass_C_d), (0, 155), cv2.FONT_HERSHEY_SIMPLEX, .5,
                        (0, 170, 0),
                        1)
            # CounterD
            cv2.rectangle(image, (0, 165), (130, 220), (255, 0, 0), -1)  # background rectangle for on-screen text
            cv2.putText(image, "In Area D", (0, 180), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 170, 0), 1)
            cv2.putText(image, "Cars: " + str(len(carids_D)), (0, 195), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 170, 0), 1)
            cv2.putText(image, "Cars Crossed: " + str(cars_pass_D_u), (0, 210), cv2.FONT_HERSHEY_SIMPLEX, .5,
                        (0, 170, 0),
                        1)

            # Pass_A, B, C, D??
            # NONO

            # Total
            cv2.rectangle(image, (490, 0), (int(width), 70), (255, 255, 255),
                          -1)  # background rectangle for on-screen text
            cv2.putText(image, "Cars in Area: " + str(currentcars_A + currentcars_B + currentcars_C + currentcars_D),
                        (490, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
            cv2.putText(image,
                        "Total Cars Detected: " + str(len(carids_A) + len(carids_B) + len(carids_C) + len(carids_D)),
                        (490, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
            cv2.putText(image, "Frame: " + str(framenumber) + ' of ' + str(frames_count), (490, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
            cv2.putText(image, 'Time: ' + str(round(framenumber / fps, 1)) + ' sec of ' + str(
                round(frames_count / fps, 2)) + ' sec', (490, 60), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)

            # origin
            '''
            cv2.rectangle(image, (0, 0), (width, 100), (255, 0, 0), -1)  # background rectangle for on-screen text

            cv2.putText(image, "Cars in Area: " + str(currentcars), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 170, 0), 1)

            cv2.putText(image, "Cars Crossed Up: " + str(carscrossedup), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 170, 0),
                        1)

            cv2.putText(image, "Cars Crossed Down: " + str(carscrosseddown), (0, 45), cv2.FONT_HERSHEY_SIMPLEX, .5,
                        (0, 170, 0), 1)

            cv2.putText(image, "Total Cars Detected: " + str(len(carids)), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, .5,
                        (0, 170, 0), 1)

            cv2.putText(image, "Frame: " + str(framenumber) + ' of ' + str(frames_count), (0, 75), cv2.FONT_HERSHEY_SIMPLEX,
                        .5, (0, 170, 0), 1)

            cv2.putText(image, 'Time: ' + str(round(framenumber / fps, 2)) + ' sec of ' + str(
                round(frames_count / fps, 2)) + ' sec', (0, 90), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 170, 0), 1)
            '''

            # displays images and transformations
            cv2.imshow("countours", image)
            cv2.moveWindow("countours", 0, 0)

            cv2.imshow("fgmask", fgmask)
            cv2.moveWindow("fgmask", int(width * ratio), 0)

            cv2.imshow("closing", closing)
            cv2.moveWindow("closing", int(width * 0.9), 0)

            cv2.imshow("opening", opening)
            cv2.moveWindow("opening", 0, int(height * ratio * 2))

            cv2.imshow("dilation", dilation)
            cv2.moveWindow("dilation", int(width * ratio), int(height * ratio * 2))

            cv2.imshow("binary", bins)
            cv2.moveWindow("binary", int(width * 0.9), int(height * ratio * 2))

            video.write(image)  # save the current image to video file from earlier

            # adds to framecount
            framenumber = framenumber + 1

            k = cv2.waitKey(int(1000 / fps)) & 0xff  # int(1000/fps) is normal speed since waitkey is in ms
            if k == 27:
                break

        else:  # if video is finished then break loop

            break

    cap.release()
    cv2.destroyAllWindows()

    # saves dataframe to csv file for later analysis
    df_A.to_csv('traffic_A.csv', sep=',')
    df_B.to_csv('traffic_B.csv', sep=',')
    df_C.to_csv('traffic_C.csv', sep=',')
    df_D.to_csv('traffic_D.csv', sep=',')



def socket() :
    import socket

    s = socket.socket()
    host = ""
    port = 9093

    s.bind((host, port))
    print("Open Server")
    s.listen(1024)

    while True :
        c, addr = s.accept()
        print('Got connection from', addr)

        print('Making data...')

        fa = open('fa.txt', 'r')
        la = fa.read(1024)
        aaa = int(la)
        la = filter(aaa)

        fb = open('fb.txt', 'r')
        lb = fb.read(1024)
        bbb = int(lb)
        lb = filter(bbb)

        fc = open('fc.txt', 'r')
        lc = fc.read(1024)
        ccc = int(lc)
        lc = filter(ccc)

        fd = open('fd.txt', 'r')
        ld = fd.read(1024)
        ddd = int(ld)
        ld = filter(ddd)

        c.send((la+','+lb+','+lc+','+ld).encode())
        fa.close()
        fb.close()
        fc.close()
        fd.close()
        print("Done Sending")

        c.close()


if __name__ == '__main__' :

    p1 = Process(target=flow)
    p2 = Process(target=socket)

    p1.start()
    p2.start()

    p1.join()
    p2.join()
