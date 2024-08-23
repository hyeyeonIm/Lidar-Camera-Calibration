#!/usr/bin/env python
# -- coding: utf-8 --
import cv2
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
from lidar import LaserToPointCloud
import sensor_msgs.point_cloud2 as pc2
import torch
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO('./best_hand.pt')

class CameraCali(object):
    def __init__(self):
        # 저장된 캘리브레이션 데이터 불러오기
        data = np.load("camera_calibration_data.npz")
        self.cameraMatrix = data['mtx']
        self.dist_coeffs = data['dist'].astype(np.float32)

        # dist_coeffs가 1차원 배열이면 reshape을 통해 (n, 1) 형식으로 변경
        if len(self.dist_coeffs.shape) == 1:
            self.dist_coeffs = self.dist_coeffs.reshape(-1, 1)

        self.rvec = data['rvec']
        self.tvec = data['tvec']

def calculate_world_position(lidar_point, cam_cali):
    # LiDAR 포인트를 동차 좌표로 변환
    objPoints = np.array(lidar_point, dtype=np.float32).reshape(-1, 3)

    # 3D 포인트를 이미지 좌표로 변환
    img_points, _ = cv2.projectPoints(
        objPoints, cam_cali.rvec, cam_cali.tvec, cam_cali.cameraMatrix, cam_cali.dist_coeffs)
    
    # 이미지 좌표 반환
    return img_points.squeeze()

if __name__ == '__main__':
    rospy.init_node('main_node', anonymous=True)

    # Laser to pointcloud
    ltp = LaserToPointCloud()

    # 캘리브레이션 데이터 로드
    cam_cali = CameraCali()

    # 비디오 열기
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # YOLOv8 객체 감지를 프레임에서 실행
            results = model(frame)
            annotated_frame = results[0].plot()

            # 결과가 있을 때만 처리
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    xyxy = box.xyxy.cpu().numpy()[0]
                    conf = box.conf.cpu().numpy()[0]
                    cls = box.cls.cpu().numpy()[0]

                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    cv2.rectangle(annotated_frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                    cv2.putText(annotated_frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # LiDAR 데이터를 사용해 객체의 이미지를 계산
                    points = pc2.read_points(ltp.cloud)
                    points_list = list(points)
                    points_list = [(x, y, z) for x, y, z, _, _ in points_list]
                    
                    if points_list:
                        # 첫 번째 LiDAR 포인트를 예시로 사용
                        image_point = calculate_world_position(points_list[0], cam_cali)

                        # 거리와 위치 정보 표시
                        text = f'Position: {image_point[:2]}'
                        cv2.putText(annotated_frame, text, (int(xyxy[0]), int(xyxy[1]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("YOLOv8 Inference", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
