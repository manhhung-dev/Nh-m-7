import os
import numpy as np
import cv2
import tensorflow as tf

# ======================
# CONFIG
# ======================
MODEL_PATH = "models/traffic_model.h5"
TEST_DIR = "data/external_test"
IMG_SIZE = 100   # ⚠️ phải đúng với lúc bạn train


# ======================
# LOAD MODEL (FIX LỖI VERSION)
# ======================
def load_model_fixed(path):
    try:
        return tf.keras.models.load_model(path, compile=False)
    except:
        print("⚠️ Fixing model load (quantization_config error)...")

        from tensorflow.keras.layers import Dense

        def custom_dense(**kwargs):
            kwargs.pop("quantization_config", None)
            return Dense(**kwargs)

        return tf.keras.models.load_model(
            path,
            compile=False,
            custom_objects={"Dense": custom_dense}
        )


print("Loading model...")
model = load_model_fixed(MODEL_PATH)
print("Model loaded!\n")


# ======================
# PREDICT FUNCTION (FIX SIGMOID)
# ======================
def predict_image(img_path):
    img = cv2.imread(img_path)

    if img is None:
        print(f"❌ Cannot read: {img_path}")
        return None, None

    # resize + normalize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    # predict
    pred = model.predict(img, verbose=0)

    # 🔥 FIX QUAN TRỌNG: sigmoid output
    confidence = float(pred[0][0])

    if confidence > 0.5:
        label = "congested"
    else:
        label = "normal"

    return label, confidence


# ======================
# MAIN
# ======================
print("Start predicting...\n")

files = os.listdir(TEST_DIR)

if len(files) == 0:
    print("❌ No images found!")

for file in files:
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(TEST_DIR, file)

        label, conf = predict_image(path)

        if label is not None:
            print(f"{file} → {label} ({conf:.2f})")

print("\nDone!")