import cv2
import time
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

# ======================
# LOAD MODELS
# ======================
yolo = YOLO("yolov8n.pt")

def load_model_fixed(path):
    try:
        return tf.keras.models.load_model(path, compile=False)
    except:
        from tensorflow.keras.layers import Dense
        def custom_dense(**kwargs):
            kwargs.pop("quantization_config", None)
            return Dense(**kwargs)
        return tf.keras.models.load_model(path, compile=False,
                                          custom_objects={"Dense": custom_dense})

cnn = load_model_fixed("models/traffic_model.h5")

# ======================
# CONFIG
# ======================
VIDEO_PATH = "data/raw/video1.mp4"
IMG_SIZE = 100

VEHICLE_CLASSES = [2, 3, 5, 7]

LINE_Y = 200   # chỉnh theo frame resize
OFFSET = 10

count_up = 0
count_down = 0
crossed_ids = {}

frame_count = 0
start_time = time.time()

paused = False
last_boxes = None

# ======================
# CNN
# ======================
def cnn_predict(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    pred = cnn.predict(img, verbose=0)
    conf = float(pred[0][0])

    return "CONGESTED" if conf > 0.5 else "NORMAL"

# ======================
# VIDEO
# ======================
cap = cv2.VideoCapture(VIDEO_PATH)

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        # ===== TỐI ƯU TỐC ĐỘ =====
        frame = cv2.resize(frame, (640, 360))
        frame_count += 1

        # ===== YOLO (chạy mỗi 2 frame) =====
        if frame_count % 2 == 0:
            results = yolo.track(frame, persist=True)
            boxes = results[0].boxes
            last_boxes = boxes
        else:
            boxes = last_boxes

        # ===== COUNT =====
        if boxes is not None and boxes.id is not None:
            for box, track_id, cls in zip(boxes.xyxy, boxes.id, boxes.cls):
                track_id = int(track_id)
                cls = int(cls)

                if cls not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box)
                cy = int((y1 + y2) / 2)

                # box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                # tracking hướng
                if track_id not in crossed_ids:
                    crossed_ids[track_id] = cy

                prev_y = crossed_ids[track_id]

                if prev_y < LINE_Y and cy >= LINE_Y:
                    count_down += 1
                    crossed_ids[track_id] = cy

                elif prev_y > LINE_Y and cy <= LINE_Y:
                    count_up += 1
                    crossed_ids[track_id] = cy

        # ===== METRICS =====
        elapsed = time.time() - start_time
        flow = (count_up + count_down) / elapsed if elapsed > 0 else 0
        density = len(boxes) if boxes is not None else 0

        # CNN (chạy mỗi 10 frame cho nhẹ)
        if frame_count % 10 == 0:
            cnn_result = cnn_predict(frame)

        # ===== RULE =====
        if density > 10 or flow < 1:
            status = "TRAFFIC JAM"
            color = (0,0,255)
        else:
            status = "NORMAL"
            color = (0,255,0)

        # ===== DRAW =====
        cv2.line(frame, (0, LINE_Y), (640, LINE_Y), (255,0,0), 2)

        cv2.putText(frame, f"UP: {count_up}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        cv2.putText(frame, f"DOWN: {count_down}", (10,55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        cv2.putText(frame, f"Flow: {flow:.2f}", (10,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        cv2.putText(frame, f"Density: {density}", (10,105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        if 'cnn_result' in locals():
            cv2.putText(frame, f"CNN: {cnn_result}", (10,130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.putText(frame, status, (10,170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)

        cv2.putText(frame, "SPACE: Pause | ESC: Exit",
                    (10,200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.imshow("Traffic System", frame)

    # ===== CONTROL =====
    key = cv2.waitKey(1)

    if key == 27:
        break
    elif key == ord(' '):
        paused = not paused

cap.release()
cv2.destroyAllWindows()