# 2019 졸업 프로젝트

### 드론을 이용한 교통혼잡제어 시스템 설계 및 구현

------

### 구성

- multi_processing.py
  - 소켓통신(Server), 영상처리, 멀티프로세싱
- Traffic_Light.py
  - Raspberry Pi 3B+ 에서 구동, 소켓통신(Client)
- Default_cut.mp4
  - 사거리 데모 영상

------

### 개요

#### 1-1) Car Detection and Socket Communication

![](C:\github\사진\2019_finalproject_02.png)

- Car Detection
  - 배경제거, 움직이는 물체는 contour로 표시
  - ROI(파란 박스)안에서만 ID부여, Centroid를 통해 추적
  - 이후 차량수는 다음 1-2에서 간단하게 설명
- Socket Communication (Server)
  - 영상 시작과 함께 Multi Processing을 통해 Car Detection과 Socket Communication을 함께 실행
  - 시작과 함께 연결을 대기
  - Car Detection을 수행하면서 중간 중간 Client(신호등)의 요청이 있으면 현재 Counting한 값들을 소켓으로 전송
  - 각 교차로별 txt파일들(Area A, B, C, D)을 읽어들여 한 번에 Client(신호등)에 전송



#### 1-2) multi_processing.py 실행 화면

![](C:\github\사진\2019_finalproject_01.png)

- 각 거리마다 들어오는 차량 Counter(Input Counter), 나가는 차량 Counter(Output Counter)
  - 파란박스에 들어오면 ID, centroid를 부여, 박스안에서 tracking을 한다. 연두색 라인을 지날때 counting을 해서 차량수를 하나씩 늘려간다
  - 오른쪽 : Area A, 왼쪽 : Area B, 위 : Area C, 아래 : Area D
  - 각 차선에 있는 신호 대기 차량 수 = Input Counter - Output Counter
  - 실시간으로 각 교차로별 차량수를 담은 txt파일로 저장



#### 2) Traffic Light Control System

![](C:\github\사진\2019_finalproject_03.png)



- Socket Communication (Client)
  - 신호 길이를 설정, 한 Cycle경과 후 Server에 교통량 Data를 요청
  - 받은 소켓을 저장후에 Area A, B, C, D 별로 교통량을 확인해 신호길이를 유동적으로 조정
  - 수신 받은 교통량으로 새로운 신호등 Cycle을 계산, GPIO를 통해 신호등 LED에 빛을 출력