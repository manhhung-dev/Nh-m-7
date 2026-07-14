import os
import random

folder = "dataset/train/normal"
files = [f for f in os.listdir(folder) if f.endswith(".jpg")]

keep = min(400, len(files))  # tránh lỗi

selected = random.sample(files, keep)

for f in files:
    if f not in selected:
        os.remove(os.path.join(folder, f))

print(f"Done, kept {keep} images")