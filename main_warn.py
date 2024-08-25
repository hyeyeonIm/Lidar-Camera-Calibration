import cv2
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from lidar import LaserToPointCloud
import sensor_msgs.point_cloud2 as pc2
from ultralytics import YOLO


# YOLOv8 모델 로드 (사전 학습된 COCO 가중치 사용)
model = YOLO('yolov8n.pt')  

class CameraCali(object):
    def __init__(self):
        # 카메라 캘리브레이션 데이터 로드
        data = np.load("camera_calibration_data.npz")
        self.cameraMatrix = data['mtx']  # 카메라 매트릭스 (내부 파라미터)
        self.dist_coeffs = data['dist'].astype(np.float32)  # 렌즈 왜곡 계수

        # 왜곡 계수가 1차원 배열일 경우 2차원 배열로 변환
        if len(self.dist_coeffs.shape) == 1:
            self.dist_coeffs = self.dist_coeffs.reshape(-1, 1)

        # LiDAR와 카메라 간의 회전 및 변환 벡터
        self.rvec = data['rvec']  # 회전 벡터
        self.tvec = data['tvec'].flatten()  # 변환 벡터

        # 초점 거리 추출
        self.focal_length = self.cameraMatrix[0, 0]  # f_x

def calculate_distance(focal_length, real_height, image_height):
    """
    카메라 매트릭스와 객체의 실제 크기를 기반으로 거리를 계산
    """
    return (focal_length * real_height) / image_height

def lidar_callback(msg):
    # LiDAR의 각도 및 거리 데이터 추출
    angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
    ranges = np.array(msg.ranges)

    # 유효한 거리 데이터만 필터링
    valid_indices = np.where((ranges > msg.range_min) & (ranges < msg.range_max))
    ranges = ranges[valid_indices]
    angles = angles[valid_indices]

    # 극좌표 데이터를 카르테시안 좌표로 변환 (x, y)
    x_coords = ranges * np.cos(angles)
    y_coords = ranges * np.sin(angles)

    # (x, y) 좌표 리스트로 반환
    return list(zip(x_coords, y_coords))

def process_lidar_and_camera(lidar_point, distance_camera, distance_lidar):
    """
    LiDAR와 Camera의 거리를 처리하는 함수
    """
    if distance_camera < 0.2:
        return distance_camera  # 0.2m 이하일 경우 카메라 거리만 사용
    elif distance_lidar < 0.5:
        error = abs(distance_camera - distance_lidar)
        if error < 0.05:
            return (distance_camera + distance_lidar) / 2
        else:
            return distance_lidar  # 오차가 큰 경우 LiDAR 거리 사용
    elif 0.5 <= distance_lidar < 10:
        return distance_lidar  # LiDAR의 첫 번째 포인트 사용
    else:
        return None  # 범위 외

def display_warnings(annotated_frame, final_distance):
    """
    객체의 거리와 경고 메시지를 화면에 표시
    """
    if final_distance < 0.50:
        text = "WARNING! Immediate Action Required!"
        color = (0, 0, 255)  # 빨간색
    elif 0.50 <= final_distance <= 0.70:
        text = "CAUTION! Object Approaching!"
        color = (0, 0, 255)  # 빨간색
    else:
        return

    # 텍스트 크기를 계산
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    text_w, text_h = text_size

    # 경고 메시지 표시 위치 조정 (이미지 경계를 넘지 않도록)
    position_x = min(annotated_frame.shape[1] - text_w - 10, 10)  # 오른쪽 경계를 넘지 않도록 조정
    position_y = max(annotated_frame.shape[0] - 20, text_h + 10)  # 하단 경계를 넘지 않도록 조정

    # 텍스트를 이미지에 그리기
    cv2.putText(annotated_frame, text, (position_x, position_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)



if __name__ == '__main__':
    # ROS 노드 초기화
    rospy.init_node('main_node', anonymous=True)
    cam_cali = CameraCali()  # 카메라 캘리브레이션 데이터 로드
    cap = cv2.VideoCapture(0)  # 카메라 비디오 스트림 시작

    lidar_points = []  # LiDAR 포인트를 저장할 리스트

    def scan_callback(msg):
        """LiDAR 데이터 수신 콜백 함수"""
        global lidar_points
        lidar_points = lidar_callback(msg)  # LiDAR 데이터를 처리하여 좌표 리스트로 저장

    # LiDAR 데이터 수신을 위한 ROS Subscriber 설정
    rospy.Subscriber("/scan", LaserScan, scan_callback)

    while cap.isOpened():
        success, frame = cap.read()  # 카메라로부터 프레임 캡처

        if success:
            results = model(frame)  # YOLOv8 모델을 사용해 객체 인식
            annotated_frame = results[0].plot()  # 인식된 객체를 포함한 프레임 생성

            if len(results[0].boxes) > 0:
                # 인식된 객체가 있는 경우
                for box in results[0].boxes:
                    # 객체의 경계 상자, 신뢰도, 클래스 정보 추출
                    xyxy = box.xyxy.cpu().numpy()[0]  # 경계 상자 좌표 (왼쪽, 위, 오른쪽, 아래)
                    image_height = xyxy[3] - xyxy[1]  # 이미지에서 객체의 높이 계산
                    distance_camera = calculate_distance(cam_cali.focal_length, 0.1, image_height)  # 카메라 기반 거리 계산

                    # 경계 상자 내부의 LiDAR 포인트 찾기
                    points_in_box = []
                    for (x, y) in lidar_points:
                        # LiDAR 포인트를 3D 포인트로 변환 (z=0 가정)
                        lidar_point_3d = np.array([x, y, 0.0], dtype=np.float32).reshape(-1, 3)

                        # LiDAR 포인트를 카메라 이미지에 프로젝션
                        img_points, _ = cv2.projectPoints(lidar_point_3d, cam_cali.rvec, cam_cali.tvec, cam_cali.cameraMatrix, cam_cali.dist_coeffs)
                        img_points = img_points.squeeze()

                        # LiDAR 포인트가 경계 상자 내부에 있는지 확인
                        if (xyxy[0] <= img_points[0] <= xyxy[2]) and (xyxy[1] <= img_points[1] <= xyxy[3]):
                            points_in_box.append((x, y))

                    if points_in_box:
                        # 경계 상자 내부에 있는 LiDAR 포인트 중 첫 번째 포인트 사용
                        lidar_point = np.array(points_in_box[0])
                        distance_lidar = np.linalg.norm(lidar_point)  # LiDAR 포인트로부터의 거리 계산

                        # LiDAR와 카메라의 거리를 처리하여 최종 거리 계산
                        final_distance = process_lidar_and_camera(lidar_point, distance_camera, distance_lidar)

                        # 거리 정보를 프레임에 표시
                        text = f'Distance: {final_distance:.2f}m'
                        cv2.putText(annotated_frame, text, 
                                    (int(xyxy[0]), int(xyxy[1]) + 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # 경고 메시지 표시
                        display_warnings(annotated_frame, final_distance)

            # # LiDAR 포인트를 카메라 이미지에 프로젝션
            # if lidar_points:
            #     for (x, y) in lidar_points:
            #         # LiDAR 포인트를 3D 포인트로 변환 (z=0 가정)
            #         lidar_point_3d = np.array([x, y, 0.0], dtype=np.float32).reshape(-1, 3)

            #         # 3D 포인트를 카메라 이미지에 프로젝션
            #         img_points, _ = cv2.projectPoints(
            #             lidar_point_3d, cam_cali.rvec, cam_cali.tvec, cam_cali.cameraMatrix, cam_cali.dist_coeffs)
            #         img_points = img_points.squeeze()
            #         cv2.circle(annotated_frame, (int(img_points[0]), int(img_points[1])), 3, (0, 0, 255), 1)

            # 최종 프레임 출력
            cv2.imshow("YOLOv8 and LiDAR Projection", annotated_frame)

            # 'q' 키를 누르면 루프 종료
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()
