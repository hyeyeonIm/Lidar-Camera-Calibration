import numpy as np
import cv2
from ultralytics import YOLO

# # YOLOv8 모델 로드
# model = YOLO('./model/best_hand.pt')

# YOLOv8 모델 로드 (사전 학습된 COCO 가중치 사용)
model = YOLO('yolov8n.pt')  # yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt 중 하나 선택


class CameraCali(object):
    def __init__(self):
        # 카메라 캘리브레이션 데이터 로드
        data = np.load("camera_calibration_data.npz")
        self.cameraMatrix = data['mtx']  # 카메라 매트릭스 (내부 파라미터)
        self.dist_coeffs = data['dist'].astype(np.float32)  # 렌즈 왜곡 계수

        # 왜곡 계수가 1차원 배열일 경우 2차원 배열로 변환
        if len(self.dist_coeffs.shape) == 1:
            self.dist_coeffs = self.dist_coeffs.reshape(-1, 1)

        # 카메라의 초점 거리 추출
        self.focal_length = self.cameraMatrix[0, 0]  # focal length (f_x)

def calculate_distance(focal_length, real_height, image_height):
    """
    카메라 매트릭스와 객체의 실제 크기를 기반으로 거리를 계산합니다.

    :param focal_length: 카메라의 초점 거리 (f_x 또는 f_y)
    :param real_height: 객체의 실제 높이 (미터 단위)
    :param image_height: 이미지에서 객체의 높이 (픽셀 단위)
    :return: 거리 (미터 단위)
    """
    distance = (focal_length * real_height) / image_height
    return distance

if __name__ == '__main__':
    cam_cali = CameraCali()  # 카메라 캘리브레이션 데이터 로드
    cap = cv2.VideoCapture(0)  # 카메라 비디오 스트림 시작

    REAL_HEIGHT = 0.2  # 객체의 실제 높이 예시 (단위: 미터)

    while cap.isOpened():
        success, frame = cap.read()  # 카메라로부터 프레임 캡처

        if success:
            results = model(frame)  # YOLOv8 모델을 사용해 객체 인식

            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    xyxy = box.xyxy.cpu().numpy()[0]  # 경계 상자 좌표 (왼쪽, 위, 오른쪽, 아래)
                    image_height = xyxy[3] - xyxy[1]  # 이미지에서 객체의 높이 계산

                    # 거리를 계산 (focal length와 실제 객체 높이 사용)
                    distance = calculate_distance(cam_cali.focal_length, REAL_HEIGHT, image_height)

                    # 경계 상자와 거리 정보를 프레임에 표시
                    label = f'Distance: {distance:.2f}m'
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                    cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # 이미지 출력
            cv2.imshow("YOLOv8 Distance Measurement", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
