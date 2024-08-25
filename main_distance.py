#!/usr/bin/env python
# -- coding: utf-8 --
import cv2
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from lidar import LaserToPointCloud
import sensor_msgs.point_cloud2 as pc2
import torch
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO('./model/best_hand.pt')

class CameraCali(object):
    def __init__(self):
        # 저장된 캘리브레이션 데이터 불러오기
        data = np.load("camera_calibration_data.npz")
        self.cameraMatrix = data['mtx']
        self.dist_coeffs = data['dist'].astype(np.float32)

        if len(self.dist_coeffs.shape) == 1:
            self.dist_coeffs = self.dist_coeffs.reshape(-1, 1)

        self.rvec = data['rvec']
        self.tvec = data['tvec'].flatten()

def lidar_callback(msg):
    # LiDAR의 각도 및 거리 데이터 추출
    angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
    ranges = np.array(msg.ranges)

    # 유효한 거리 데이터만 사용 (range_min과 range_max 사이)
    valid_indices = np.where((ranges > msg.range_min) & (ranges < msg.range_max))
    ranges = ranges[valid_indices]
    angles = angles[valid_indices]

    # 각도와 거리로 x, y 좌표 계산
    x_coords = ranges * np.cos(angles)
    y_coords = ranges * np.sin(angles)

    # (x, y) 좌표 리스트로 반환
    return list(zip(x_coords, y_coords))

if __name__ == '__main__':
    rospy.init_node('main_node', anonymous=True)
    cam_cali = CameraCali()
    cap = cv2.VideoCapture(0)

    lidar_points = []

    def scan_callback(msg):
        global lidar_points
        lidar_points = lidar_callback(msg)

    # LiDAR 데이터 수신을 위한 ROS Subscriber 설정
    rospy.Subscriber("/scan", LaserScan, scan_callback)

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model(frame)
            annotated_frame = results[0].plot()

            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    xyxy = box.xyxy.cpu().numpy()[0]
                    conf = box.conf.cpu().numpy()[0]
                    cls = box.cls.cpu().numpy()[0]

                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    cv2.rectangle(annotated_frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                    cv2.putText(annotated_frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    if lidar_points:
                        # 첫 번째 LiDAR 포인트를 사용 (2D 포인트로 가정)
                        lidar_point = np.array(lidar_points[0])
                        
                        # LiDAR로부터의 거리
                        distance_lidar = np.linalg.norm(lidar_point)
                        
                        # 카메라로부터의 거리
                        distance_camera = np.linalg.norm(np.dot(cam_cali.rvec, np.append(lidar_point, 0)) + cam_cali.tvec)
                        
                        # 거리 정보 표시
                        text = f'Lidar Dist: {distance_lidar:.2f}m, Camera Dist: {distance_camera:.2f}m'
                        cv2.putText(annotated_frame, text, (int(xyxy[0]), int(xyxy[1]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("YOLOv8 Inference", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
