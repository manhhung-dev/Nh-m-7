import os
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "models/traffic_model.h5"
TEST_DIR = "data/external_test"
IMG_SIZE = 100

# 🔥 LABEL THỰC TẾ (bạn sửa theo ảnh của bạn)
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
        return "congested", conf
    else:
        return "normal", conf


# ===== MAIN =====
correct = 0
total = 0

for file in os.listdir(TEST_DIR):
    if file.lower().endswith(".jpg"):
        path = os.path.join(TEST_DIR, file)

        img = cv2.imread(path)
        if img is None:
            continue

        pred_label, conf = predict(img)
        true_label = GROUND_TRUTH.get(file, "unknown")

        total += 1

        if pred_label == true_label:
            result = "CORRECT"
            color = (0, 255, 0)
            correct += 1
        else:
            result = "WRONG"
            color = (0, 0, 255)

        # text hiển thị
        text1 = f"Pred: {pred_label} ({conf:.2f})"
        text2 = f"True: {true_label}"
        text3 = result

        cv2.putText(img, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(img, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(img, text3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        cv2.imshow("Evaluation", img)
        cv2.waitKey(0)

cv2.destroyAllWindows()

print(f"\nAccuracy: {correct}/{total} = {correct/total:.2f}")