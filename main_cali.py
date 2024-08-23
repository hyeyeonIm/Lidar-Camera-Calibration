#!/usr/bin/env python
# -- coding: utf-8 --
import cv2
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
from lidar import LaserToPointCloud
import sensor_msgs.point_cloud2 as pc2

class CameraCali(object):
    def __init__(self):
        # 저장된 캘리브레이션 데이터 불러오기
        data = np.load("camera_calibration_data.npz")
        self.cameraMatrix = data['mtx']
        self.dist_coeffs = data['dist'].astype(np.float32)  # float32로 변환

        # dist_coeffs가 1차원 배열이면 reshape을 통해 (n, 1) 형식으로 변경
        if len(self.dist_coeffs.shape) == 1:
            self.dist_coeffs = self.dist_coeffs.reshape(-1, 1)

        self.rvec = data['rvec']
        self.tvec = data['tvec']

if __name__ == '__main__':
    rospy.init_node('main_node', anonymous=True)

    # Laser to pointcloud
    ltp = LaserToPointCloud()

    # Load calibration data
    cam_cali = CameraCali()

    # Open the video
    cap = cv2.VideoCapture(0)

    # Loop through the video frames
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Read points from the cloud
            points = pc2.read_points(ltp.cloud)
            points_list = list(points)
            # Points consist of world-coordinate x, y, z, intensity and ring
            # We only use (x, y, z)
            points_list = [(x, y, z) for x, y, z, _, _ in points_list]

            print("Length of points: ", len(points_list))
            print("First few elements of points: ", points_list[:10])

            # Convert to numpy array and reshape
            objPoints = np.array(points_list, dtype=np.float32).reshape(-1, 3)

            # Display the annotated frame
            img_points, jacobian = cv2.projectPoints(
                objPoints, cam_cali.rvec, cam_cali.tvec, cam_cali.cameraMatrix, cam_cali.dist_coeffs)

            # Flatten img_points to remove the extra dimension
            img_points = img_points.squeeze()

            for i in range(len(img_points)):
                # Express Lidar points to image using circle
                cv2.circle(frame, (int(img_points[i][0]), int(img_points[i][1])), 3, (0, 0, 255), 1)

            cv2.imshow("image", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

        # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
