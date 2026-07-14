import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

MODEL_PATH = "models/traffic_model.h5"
TEST_DIR = "data/external_test"
IMG_SIZE = 100

# 🔥 Label thật (bạn chỉnh đúng)
GROUND_TRUTH = {
    "img1.jpg": "congested",
    "img2.jpg": "normal",
    "img3.jpg": "congested",
    "img4.jpg": "congested",
    "img5.jpg": "normal",
    "img6.jpg": "normal",
    "img7.jpg": "congested",
    "img8.jpg": "congested",
    "img9.jpg": "congested",
    "img10.jpg": "normal",
    "img11.jpg": "normal",
    "img12.jpg": "congested",
    "img13.jpg": "normal",
    "img14.jpg": "congested",
    "img15.jpg": "congested",
}


# ===== LOAD MODEL =====
def load_model_fixed(path):
    try:
        return tf.keras.models.load_model(path, compile=False)
    except:
        from tensorflow.keras.layers import Dense

        def custom_dense(**kwargs):
            kwargs.pop("quantization_config", None)
            return Dense(**kwargs)

        return tf.keras.models.load_model(
            path,
            compile=False,
            custom_objects={"Dense": custom_dense}
        )


model = load_model_fixed(MODEL_PATH)


# ===== PREDICT =====
def predict(img):
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_resized = img_resized / 255.0
    img_resized = np.reshape(img_resized, (1, IMG_SIZE, IMG_SIZE, 3))

    pred = model.predict(img_resized, verbose=0)
    conf = float(pred[0][0])

    if conf > 0.5:
        return "congested"
    else:
        return "normal"


# ===== EVALUATE =====
correct = 0
total = 0

for file in os.listdir(TEST_DIR):
    if file.lower().endswith(".jpg"):
        path = os.path.join(TEST_DIR, file)

        img = cv2.imread(path)
        if img is None:
            continue

        pred_label = predict(img)
        true_label = GROUND_TRUTH.get(file)

        if true_label is None:
            continue

        total += 1

        if pred_label == true_label:
            correct += 1

wrong = total - correct
accuracy = correct / total if total > 0 else 0

print(f"Correct: {correct}")
print(f"Wrong: {wrong}")
print(f"Accuracy: {accuracy:.2f}")


# ===== PLOT =====

# 1. Correct vs Wrong
plt.figure()
plt.bar(["Correct", "Wrong"], [correct, wrong])
plt.title("Prediction Results")
plt.xlabel("Category")
plt.ylabel("Number of Images")
plt.savefig("outputs/correct_vs_wrong.png")


# 2. Accuracy
plt.figure()
plt.bar(["Accuracy"], [accuracy])
plt.ylim(0, 1)
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.savefig("outputs/accuracy.png")

print("\nSaved to outputs/")