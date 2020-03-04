# -*- coding: utf-8-*-


import RPi.GPIO as GPIO

import time

import socket

server_ip = "IP_ADDRESS"
server_port = #Port_Num
server_addr = (server_ip, server_port)

GPIO.setmode(GPIO.BCM)

# A<->B 상황 신호등
GPIO.setup(13, GPIO.OUT)  # Green
GPIO.setup(19, GPIO.OUT)  # Yellow
GPIO.setup(26, GPIO.OUT)  # Red
GPIO.setup(6, GPIO.OUT)  # 좌회전

# C<->D 상황 신호등
GPIO.setup(16, GPIO.OUT)  # Green
GPIO.setup(20, GPIO.OUT)  # Yellow
GPIO.setup(21, GPIO.OUT)  # Red
GPIO.setup(12, GPIO.OUT)  # 여기에 좌회전 추가

# 처음 defalut 신호 시간 설정 total: 70sec
Time_AB = 35
Time_CD = 35
Time_ABL = 15  # int(Time_AB * (7 / 17))
Time_AB = Time_AB - Time_ABL
Time_CDL = 15  # int(Time_CD * (7 / 17))
Time_CD = Time_CD - Time_CDL

while True:
    GPIO.cleanup()
    GPIO.setmode(GPIO.BCM)

    # A<->B 상황 신호등
    GPIO.setup(13, GPIO.OUT)  # Green
    GPIO.setup(19, GPIO.OUT)  # Yellow
    GPIO.setup(26, GPIO.OUT)  # Red
    GPIO.setup(6, GPIO.OUT)  # 여기에 좌회전 GPIO추가

    # C<->D 상황 신호등
    GPIO.setup(16, GPIO.OUT)  # Green
    GPIO.setup(20, GPIO.OUT)  # Yellow
    GPIO.setup(21, GPIO.OUT)  # Red
    GPIO.setup(12, GPIO.OUT)  # 여기에 좌회전 추가

    # 신호등 제어
    #### C, D 제어 상황 ###
    # 싸이클: C,D 좌회전 -> C,D초록불 -> A,B 좌회전 -> A,B 초록불 #
    # C,D 상황을 제어 할때 A, B는 항상 빨간 불
    GPIO.output(13, False)  # Green
    GPIO.output(19, False)  # Yellow
    GPIO.output(26, True)  # Red
    GPIO.output(6, False)  # 좌회전

    # C,D 좌회전 / 좌회전 시 빨간불 + 좌회전 켜짐
    GPIO.output(16, False)
    GPIO.output(20, False)
    GPIO.output(21, True)
    GPIO.output(12, True)

    time.sleep(int(Time_CDL - 5))  # 황색 시간 5초 빼줌

    # C,D 좌회전 노란불
    GPIO.output(16, False)
    GPIO.output(20, True)
    GPIO.output(21, True)
    GPIO.output(12, False)

    time.sleep(5)

    # C<->D 초록불
    GPIO.output(16, True)
    GPIO.output(20, False)
    GPIO.output(21, False)
    GPIO.output(12, False)

    time.sleep(int(Time_CD - 5))

    # C<->D 노란불
    GPIO.output(16, False)
    GPIO.output(20, True)
    GPIO.output(21, False)
    GPIO.output(12, False)

    time.sleep(5)

    #### A, B 제어 상황 ####

    # C, D 항상 빨간불
    GPIO.output(16, False)
    GPIO.output(20, False)
    GPIO.output(21, True)
    GPIO.output(6, False)

    # A,B 좌회전
    GPIO.output(13, False)
    GPIO.output(19, False)
    GPIO.output(26, True)
    GPIO.output(6, True)

    time.sleep(int(Time_ABL - 5))

    # A,B 좌회전 노란불
    GPIO.output(13, False)
    GPIO.output(19, True)
    GPIO.output(26, True)
    GPIO.output(6, False)

    time.sleep(5)

    # A<->B 초록불 코드
    GPIO.output(13, True)
    GPIO.output(19, False)
    GPIO.output(26, False)
    GPIO.output(6, False)

    time.sleep(int(Time_AB - 5))

    # 소켓 통신 및 연산 ( 마지막 노란신호전에 )
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(server_addr)
    l = client_socket.recv(1024)

    f = open('num of cars.txt', 'wb')
    print("Receiving data ... ")
    f.write(l)
    f.close()
    print("Receiving Done")
    client_socket.close()
    f = open('num of cars.txt', 'r')

    # 파일에 내용을 읽어 들임
    lines = f.read(1024)
    f.close()

    # 내용을 , 기준으로 짤라 time_str에 저장 (str형 list)
    time_str = lines.split(',')

    # str형을 int형으로 변화e_CD = Time_CD - Time_CDL
    time_int = []
    for t in time_str:
        time_int.append(int(t))

    # 신호 시간 계산
    total = time_int[0] + time_int[1] + time_int[2] + time_int[3]  # 총 이동량
    count_AB = time_int[0] + time_int[1]  # A+B의 이동량
    count_CD = time_int[2] + time_int[3]  # C+D의 이동량
    weight = 0.0

    if (total <= 64):  # 0~64
        weight = 1.00
    elif (total <= 79):  # 65~79
        weight = 1.32
    elif (total <= 94):  # 80~94
        weight = 1.64
    elif (total <= 109):  # 95~109
        weight = 1.96
    elif (total <= 124):  # 110~124
        weight = 2.28
    else:  # 124~
        weight = 2.6

    if (count_AB > count_CD):
        # round(실수, n): n 자리까지만 표시
        Time_AB = round(70 * weight / 2 + 70 * (weight - 1) * (count_AB / total), 2)
        Time_CD = round(70 * weight - Time_AB, 2)
    else:
        Time_CD = round(70 * weight / 2 + 70 * (weight - 1) * (count_CD / total), 2)
        Time_AB = round(70 * weight - Time_CD, 2)

    #print("{} {}".format(count_AB, count_CD))
    #print("{} {}".format(Time_AB, Time_CD))

    # 좌회전 시간계산
    Time_ABL = Time_AB * 7 / 17
    Time_AB = Time_AB - Time_ABL
    Time_CDL = Time_CD * 7 / 17
    Time_CD = Time_CD - Time_CDL

    #print("{}, {}, {}, {}".format(Time_AB, Time_CD, Time_ABL, Time_CDL))

    # A<->B 노란불 코드
    GPIO.output(13, False)
    GPIO.output(19, True)
    GPIO.output(26, False)
    GPIO.output(6, False)

    time.sleep(5)