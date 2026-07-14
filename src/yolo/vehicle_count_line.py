import cv2
from ultralytics import YOLO

# load YOLO
model = YOLO("yolov8n.pt")

video_path = "data/raw/video1.mp4"
cap = cv2.VideoCapture(video_path)

# ===== CONFIG =====
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorbike, bus, truck
LINE_Y = 300  # vị trí line
OFFSET = 10   # sai số

count = 0
crossed_ids = set()

paused = False

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True)

        boxes = results[0].boxes

        if boxes.id is not None:
            for box, track_id in zip(boxes.xyxy, boxes.id):
                x1, y1, x2, y2 = map(int, box)
                track_id = int(track_id)

                cls = int(boxes.cls[0])

                if cls in VEHICLE_CLASSES:
                    cy = int((y1 + y2) / 2)

                    # vẽ box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.circle(frame, (int((x1+x2)/2), cy), 4, (0,0,255), -1)

                    # check qua line
                    if LINE_Y - OFFSET < cy < LINE_Y + OFFSET:
                        if track_id not in crossed_ids:
                            count += 1
                            crossed_ids.add(track_id)

        # ===== VẼ LINE =====
        cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (255,0,0), 3)

        # ===== TEXT =====
        cv2.putText(frame, f"Count: {count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

        cv2.putText(frame, "SPACE: Pause | N: Next | ESC: Exit",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        cv2.imshow("Vehicle Counting Line", frame)

    key = cv2.waitKey(30)

    if key == 27:
        break
    elif key == ord(' '):
        paused = not paused
    elif key == ord('n'):
        paused = True

cap.release()
cv2.destroyAllWindows()