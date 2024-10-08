1. open_camera.py
    1. 카메라가 열리는지 체크

2. lidar_cam_cali.py
    1. 카메라 내부 파라미터 구함
        1. 체커보드의 한 변의 길이 수정해줘야함!
        2. Lidar랑 카메라의 매칭 포인트 수정해줘야함!
            1. 카메라랑 Lidar 고정된 위치에 두기
            2. Lidar 실행하고 Rivz켜서 월드 좌표계 확인 (점 크기 크게 해서 잘 보기)
            * lidar의 point의 경우 “rostopic echo /clicked_point”를 이용하여 수집
3. lidar.py
    1. Lidar에서 scan data가져오는 거

4. main_cali.py
    1. Lidar랑 카메라 data overlay 되는 거 확인

5. main_cam.py
    1. 카메라 내부 파라미터를 이용한 calibration으로 거리 측정

6. main.py
    1. Lidar와 카메라 calibration을 통해, 객체 인식 후 거리 측정 후 경고 메시지 표시
    2. troubleshooting에 나온 것 처럼 조건들 추가
