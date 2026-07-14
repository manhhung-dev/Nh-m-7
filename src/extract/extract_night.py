import cv2
import os

video_path = "video_night.mp4"
output_folder = "dataset/train/normal"  # hoặc congested nếu là ùn tắc
frame_interval = 2

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Không mở được video")
    exit()
else:
    print("✅ Đã mở video")

count = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Không đọc được frame nữa")
        break

    if count % frame_interval == 0:
        filename = os.path.join(output_folder, f"night_{saved}.jpg")
        cv2.imwrite(filename, frame)
        saved += 1

    count += 1

cap.release()

print(f"🔥 Saved {saved} images")