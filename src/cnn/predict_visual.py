import os
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "models/traffic_model.h5"
TEST_DIR = "data/external_test"
IMG_SIZE = 100


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


def predict(img):
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_resized = img_resized / 255.0
    img_resized = np.reshape(img_resized, (1, IMG_SIZE, IMG_SIZE, 3))

    pred = model.predict(img_resized, verbose=0)
    conf = float(pred[0][0])

    if conf > 0.5:
        label = "Congested"
        color = (0, 0, 255)  # đỏ
    else:
        label = "Normal"
        color = (0, 255, 0)  # xanh

    return label, conf, color


for file in os.listdir(TEST_DIR):
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(TEST_DIR, file)

        img = cv2.imread(path)
        if img is None:
            continue

        label, conf, color = predict(img)

        text = f"{label} ({conf:.2f})"
        cv2.putText(img, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 2)

        cv2.imshow("Result", img)
        cv2.waitKey(0)

cv2.destroyAllWindows()