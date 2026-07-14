import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ======================
# CONFIG
# ======================
MODEL_PATH = "models/traffic_model.h5"
TEST_DIR = "data/external_test"
IMG_SIZE = 100

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

# ======================
# LOAD MODEL
# ======================
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

print("Loading model...")
model = load_model_fixed(MODEL_PATH)
print("Model loaded!\n")

# ======================
# PREDICT
# ======================
def predict(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    pred = model.predict(img, verbose=0)
    conf = float(pred[0][0])

    if conf > 0.5:
        return "congested", conf
    else:
        return "normal", conf


# ======================
# MAIN EVALUATION
# ======================
correct = 0
total = 0

y_true = []
y_pred = []

print("Start evaluation...\n")

for file in sorted(os.listdir(TEST_DIR)):
    if file.endswith(".jpg"):
        path = os.path.join(TEST_DIR, file)

        img = cv2.imread(path)
        if img is None:
            continue

        pred_label, conf = predict(img)
        true_label = GROUND_TRUTH.get(file)

        if true_label is None:
            continue

        total += 1
        y_true.append(true_label)
        y_pred.append(pred_label)

        # ===== CHECK =====
        if pred_label == true_label:
            result = "CORRECT"
            color = (0, 255, 0)
            correct += 1
        else:
            result = "WRONG"
            color = (0, 0, 255)

        # ===== TEXT =====
        text1 = f"Pred: {pred_label} ({conf:.2f})"
        text2 = f"True: {true_label}"
        text3 = result
        text4 = "Press N: Next | ESC: Exit"

        cv2.putText(img, text1, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.putText(img, text2, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.putText(img, text3, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        cv2.putText(img, text4, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Evaluation", img)

        # ===== CONTROL =====
        while True:
            key = cv2.waitKey(0)

            if key == 27:  # ESC
                cv2.destroyAllWindows()
                exit()

            if key == ord('n'):  # next
                break

cv2.destroyAllWindows()

# ======================
# METRICS
# ======================
wrong = total - correct
accuracy = correct / total if total > 0 else 0

print("\n===== RESULT =====")
print(f"Correct: {correct}")
print(f"Wrong: {wrong}")
print(f"Accuracy: {accuracy:.2f}")

# ======================
# BAR CHART
# ======================
plt.figure()
plt.bar(["Correct", "Wrong"], [correct, wrong])
plt.title("Prediction Results")
plt.savefig("outputs/result_bar.png")

# ======================
# CONFUSION MATRIX
# ======================
labels = ["normal", "congested"]
cm = confusion_matrix(y_true, y_pred, labels=labels)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")

plt.show()

# ======================
# AUTO ANALYSIS
# ======================
print("\n===== ANALYSIS =====")

if accuracy >= 0.85:
    print("Model hoạt động TỐT, khả năng tổng quát hóa cao.")
elif accuracy >= 0.7:
    print("Model hoạt động KHÁ, nhưng vẫn còn lỗi.")
else:
    print("Model hoạt động CHƯA TỐT, cần cải thiện.")

print("\nNguyên nhân sai:")
print("- Ảnh có mật độ xe không rõ ràng")
print("- Góc camera khác dataset")
print("- Điều kiện ánh sáng khác")

# python src/cnn/full_evaluation.py