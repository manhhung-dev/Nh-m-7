import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

video_path = "data/raw/video1.mp4"
cap = cv2.VideoCapture(video_path)

paused = False

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        cv2.imshow("YOLO Detection", annotated_frame)

    key = cv2.waitKey(30)

    if key == 27:  # ESC
        break

    elif key == ord(' '):  # SPACE pause/play
        paused = not paused

    elif key == ord('n'):  # next frame
        paused = True
        ret, frame = cap.read()
        if ret:
            results = model(frame)
            annotated_frame = results[0].plot()
            cv2.imshow("YOLO Detection", annotated_frame)

cap.release()
cv2.destroyAllWindows()