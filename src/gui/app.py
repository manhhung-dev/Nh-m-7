import tkinter as tk
from tkinter import filedialog
import cv2
import time
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# ======================
# LOAD MODEL
# ======================
model = YOLO("yolov8n.pt")

video_path = None

# ======================
# CHỌN VIDEO
# ======================
def choose_video():
    global video_path
    video_path = filedialog.askopenfilename()
    label.config(text=f"📂 {video_path}")

# ======================
# PHAN TICH + VE BIEU DO
# ======================
def show_analysis(status_list):

    # clear old charts
    for widget in chart_frame.winfo_children():
        widget.destroy()

    # ===== DEM DU LIEU THAT =====
    normal = status_list.count("BINH THUONG")
    heavy = status_list.count("DONG XE")
    jam = status_list.count("TAC DUONG")

    labels = ["BINH THUONG", "DONG XE", "TAC DUONG"]
    values = [normal, heavy, jam]

    # ===== % =====
    total = sum(values)
    percent = [v/total*100 if total > 0 else 0 for v in values]

    # ===== BAR CHART =====
    fig1 = plt.Figure(figsize=(5,4))
    ax1 = fig1.add_subplot(111)

    ax1.bar(labels, percent)
    ax1.set_title("PHAN BO TRANG THAI (%)")
    ax1.set_ylabel("%")

    canvas1 = FigureCanvasTkAgg(fig1, master=chart_frame)
    canvas1.draw()
    canvas1.get_tk_widget().pack(side=tk.LEFT, padx=20)

    # ===== MATRIX (KHONG FAKE) =====
    cm = np.array([
        [normal, 0, 0],
        [0, heavy, 0],
        [0, 0, jam]
    ])

    fig2 = plt.Figure(figsize=(5,4))
    ax2 = fig2.add_subplot(111)

    cax = ax2.matshow(cm)
    fig2.colorbar(cax)

    ax2.set_xticks([0,1,2])
    ax2.set_yticks([0,1,2])

    ax2.set_xticklabels(labels)
    ax2.set_yticklabels(labels)

    ax2.set_title("MA TRAN PHAN BO TRANG THAI")

    for i in range(3):
        for j in range(3):
            ax2.text(j, i, cm[i, j], ha='center', va='center')

    canvas2 = FigureCanvasTkAgg(fig2, master=chart_frame)
    canvas2.draw()
    canvas2.get_tk_widget().pack(side=tk.RIGHT, padx=20)

    # ===== KET LUAN TIENG VIET KHONG DAU =====
    final_status = max(set(status_list), key=status_list.count)

    if final_status == "TAC DUONG":
        note = "Mat do cao, xe di chuyen cham"
    elif final_status == "DONG XE":
        note = "Nhieu xe nhung van di chuyen duoc"
    else:
        note = "Giao thong thong thoang"

    result_label.config(text=f"KET LUAN: {final_status}\n{note}")

# ======================
# RUN VIDEO
# ======================
def run_video():
    global video_path

    if video_path is None:
        label.config(text="❌ Chua chon video")
        return

    cap = cv2.VideoCapture(video_path)

    VEHICLE_CLASSES = [2,3,5,7]
    LINE_Y = 200

    count_up = 0
    count_down = 0

    track_history = {}
    counted_ids = set()

    start_time = time.time()
    densities = []
    status_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640,360))

        results = model.track(frame, persist=True)
        boxes = results[0].boxes

        current_ids = []

        if boxes.id is not None:
            for box, track_id, cls in zip(boxes.xyxy, boxes.id, boxes.cls):

                track_id = int(track_id)
                cls = int(cls)

                if cls not in VEHICLE_CLASSES:
                    continue

                x1,y1,x2,y2 = map(int, box)
                cy = int((y1+y2)/2)

                current_ids.append(track_id)

                if track_id not in track_history:
                    track_history[track_id] = []

                track_history[track_id].append(cy)

                if len(track_history[track_id]) > 2:
                    track_history[track_id] = track_history[track_id][-2:]

                if len(track_history[track_id]) == 2:
                    prev_y, curr_y = track_history[track_id]

                    if track_id not in counted_ids:

                        if prev_y < LINE_Y and curr_y >= LINE_Y:
                            count_down += 1
                            counted_ids.add(track_id)

                        elif prev_y > LINE_Y and curr_y <= LINE_Y:
                            count_up += 1
                            counted_ids.add(track_id)

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        density = len(current_ids)
        densities.append(density)

        elapsed = time.time() - start_time
        total = count_up + count_down
        flow = total / elapsed if elapsed > 0 else 0

        recent = densities[-30:] if len(densities)>30 else densities
        avg_density = sum(recent)/len(recent) if recent else 0

        # ===== RULE =====
        if avg_density > 12 and flow < 1:
            status = "TAC DUONG"
            color = (0,0,255)

        elif avg_density > 8:
            status = "DONG XE"
            color = (0,165,255)

        else:
            status = "BINH THUONG"
            color = (0,255,0)

        status_list.append(status)

        # ===== DRAW =====
        cv2.line(frame,(0,LINE_Y),(640,LINE_Y),(255,0,0),2)

        cv2.putText(frame,f"UP: {count_up}",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)

        cv2.putText(frame,f"DOWN: {count_down}",(10,55),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)

        cv2.putText(frame,f"Flow: {flow:.2f}",(10,80),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

        cv2.putText(frame,f"Density: {avg_density:.1f}",(10,105),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

        cv2.putText(frame,status,(10,140),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,color,3)

        cv2.imshow("Traffic System", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    show_analysis(status_list)

# ======================
# GUI DESIGN (DEP HON)
# ======================
root = tk.Tk()
root.title("🚦 HE THONG PHAN TICH GIAO THONG")
root.geometry("1000x650")
root.configure(bg="#f5f5f5")

top_frame = tk.Frame(root, bg="#f5f5f5")
top_frame.pack(pady=10)

btn_choose = tk.Button(top_frame, text="📂 Chon Video",
                       command=choose_video,
                       bg="#4CAF50", fg="white",
                       font=("Arial",10,"bold"), width=18)
btn_choose.grid(row=0, column=0, padx=10)

btn_run = tk.Button(top_frame, text="▶ Bat dau",
                    command=run_video,
                    bg="#2196F3", fg="white",
                    font=("Arial",10,"bold"), width=18)
btn_run.grid(row=0, column=1, padx=10)

label = tk.Label(root, text="Chua chon video",
                 fg="blue", bg="#f5f5f5")
label.pack()

result_label = tk.Label(root,
                        text="",
                        font=("Arial",16,"bold"),
                        fg="red",
                        bg="#f5f5f5")
result_label.pack(pady=10)

chart_frame = tk.Frame(root, bg="#f5f5f5")
chart_frame.pack(pady=20)

root.mainloop()
# python src/gui/app.py