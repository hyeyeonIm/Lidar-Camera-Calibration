#!/usr/bin/env python
# -- coding: utf-8 --

import numpy as np
import cv2
import os
import glob
from tqdm import tqdm  # tqdm import

class CameraCali(object):
    def __init__(self):
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        wc = 8  # the number of horizontal checkerboard patterns - 1
        hc = 6  # the number of vertical checkerboard patterns - 1
        square_size = 2.5  # 한 변의 길이(cm)

        # 각 코너의 실제 3D 좌표를 정의할 때, 한 변의 길이를 고려
        objp = np.zeros((wc * hc, 3), np.float32)
        objp[:, :2] = np.mgrid[0:wc, 0:hc].T.reshape(-1, 2)
        objp *= square_size  # 실제 세계의 크기를 반영

        objpoints = []
        imgpoints = []

        # import images in directory
        images = glob.glob('/home/haley/catkin_ws/src/cali/checkerboard/*.jpg')

        # tqdm으로 진행 상태 표시
        for frame in tqdm(images, desc="Processing images"):
            img = cv2.imread(frame)
            self.gray = cv2.cvtColor(
                img, cv2.COLOR_BGR2GRAY)  # change to gray scale

            # find the checkerboard
            ret, corners = cv2.findChessboardCorners(
                self.gray, (wc, hc), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)  # 체스 보드 찾기

            # if ret is False, please check your checkerboard (wc, hc)
            if ret:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(
                    self.gray, corners, (10, 10), (-1, -1), criteria)
                imgpoints.append(corners2)

                # draw images using corner points
                img = cv2.drawChessboardCorners(img, (wc, hc), corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(100)  # 이미지 표시 시간을 짧게 조정

        print("Starting camera calibration...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, self.gray.shape[::-1], None, None)

        # 2D-image coordination
        points_2D = np.array([
            (31, 291),
            (131, 309),
            (279, 307),
            (363, 308),
            (510, 314),
            (614, 296)
        ], dtype="double")

        # 3D-World coordinations correspond to 2D-image coordinations
        points_3D = np.array([
            (-1.0958367586135864, -0.3753542900085449, -0.0002808570861816406),
            (-1.4953137636184692, -0.32586169242858887, 0.005124092102050781),
            (-1.5498539209365845, -0.052880287170410156, 0.003710508346557617),
            (-1.5279659032821655, 0.13236546516418457, 0.003645181655883789),
            (-1.8375319242477417, 0.5367050170898438, 0.005119681358337402),
            (-1.1265267133712769, 0.48479700088500977, 0.001104593276977539),
        ], dtype="double")

        self.cameraMatrix = mtx

        # Distortion coefficients in camera matrix, initially assuming no distortion
        self.dist_coeffs = np.array([0, 0, 0, 0], dtype=np.float32)

        # using solvePnP to calculate R, t
        retval, rvec, tvec = cv2.solvePnP(points_3D, points_2D, self.cameraMatrix,
                                          self.dist_coeffs, rvec=None, tvec=None, useExtrinsicGuess=None, flags=None)

        self.rvec, _ = cv2.Rodrigues(rvec)
        self.tvec = tvec

        # express homogeneous coordinate
        self.intrisic = np.append(self.cameraMatrix, [[0, 0, 0]], axis=0)
        self.intrisic = np.append(self.intrisic, [[0], [0], [0], [1]], axis=1)

        # intrinsic parameter
        self.intrinsic = self.intrisic
        # extrinsic parameter
        extrinsic = np.append(self.rvec, self.tvec, axis=1)
        self.extrinsic = np.append(extrinsic, [[0, 0, 0, 1]], axis=0)

        # Calibration data 저장
        np.savez("camera_calibration_data.npz", mtx=self.cameraMatrix, dist=self.dist_coeffs, rvec=self.rvec, tvec=self.tvec)

        print("Calibration completed.")
        print()
        print("intrinsic: ", end='\n')
        print(self.intrinsic)
        print("extrinsic: ", end='\n')
        print(self.extrinsic)
        print("R:", end='\n')
        print(self.rvec)
        print("t:", end='\n')
        print(self.tvec)

if __name__ == '__main__':
    cam_cali = CameraCali()
