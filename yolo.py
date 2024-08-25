from ultralytics import YOLO
import cv2

# # Load the YOLOv8 model
# model = YOLO('./model/best_hand.pt')

# YOLOv8 모델 로드 (사전 학습된 COCO 가중치 사용)
model = YOLO('yolov8n.pt')  # yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt 중 하나 선택


cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()


