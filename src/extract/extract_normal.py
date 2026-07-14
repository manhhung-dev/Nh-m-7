import cv2
import os

# ====== CHỈNH Ở ĐÂY ======
video_path = "video1.mp4"
output_folder = "dataset/train/normal"
frame_interval = 5
# ==========================

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

count = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if count % frame_interval == 0:
        filename = os.path.join(output_folder, f"frame_{saved}.jpg")
        cv2.imwrite(filename, frame)
        saved += 1

    count += 1

cap.release()

print(f"Saved {saved} images")