import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

MODEL_PATH = "models/traffic_model.h5"
TEST_DIR = "data/external_test"
IMG_SIZE = 100

# 🔥 Ground truth
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

# ===== COLLECT DATA =====
y_true = []
y_pred = []

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

        y_true.append(true_label)
        y_pred.append(pred_label)

# ===== CONFUSION MATRIX =====
labels = ["normal", "congested"]
cm = confusion_matrix(y_true, y_pred, labels=labels)

print("Confusion Matrix:")
print(cm)

# ===== PLOT =====
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()

plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.show()