#!/usr/bin/env python
# -- coding: utf-8 --

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
                    conf = box.conf.cpu().numpy()[0]  # 신뢰도 점수
                    cls = box.cls.cpu().numpy()[0]  # 클래스 인덱스

                    # 경계 상자와 레이블을 프레임에 표시
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    cv2.rectangle(annotated_frame, (int(xyxy[0]), int(xyxy[1])), 
                                  (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                    cv2.putText(annotated_frame, label, 
                                (int(xyxy[0]), int(xyxy[1]) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

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
                        distance_camera = np.linalg.norm(np.dot(cam_cali.rvec, np.append(lidar_point, 0)) + cam_cali.tvec)  # 카메라로부터의 거리 계산
                        
                        # 거리 정보를 프레임에 표시
                        text = f'Lidar Dist: {distance_lidar:.2f}m, Camera Dist: {distance_camera:.2f}m'
                        cv2.putText(annotated_frame, text, 
                                    (int(xyxy[0]), int(xyxy[1]) + 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
