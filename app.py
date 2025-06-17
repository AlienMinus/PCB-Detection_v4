import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

st.title("YOLOv8 PCB Component Detection")

# Load model once
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# Webcam or image/video upload
option = st.radio("Choose input source:", ("Webcam", "Image/Video Upload"))

def draw_boxes(frame, results, model):
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[int(box.cls[0])]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    return frame

if option == "Webcam":
    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])
    cap = None

    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to grab frame. Try File Upload instead.")
                break
            results = model(frame)[0]
            frame = draw_boxes(frame, results, model)
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if cap:
            cap.release()
else:
    uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4"])
    if uploaded_file is not None:
        if uploaded_file.type.startswith("image"):
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            results = model(img)[0]
            img = draw_boxes(img, results, model)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Detected Components", use_column_width=True)
        elif uploaded_file.type == "video/mp4":
            tfile = open("temp_video.mp4", "wb")
            tfile.write(uploaded_file.read())
            tfile.close()
            cap = cv2.VideoCapture("temp_video.mp4")
            FRAME_WINDOW = st.image([])
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame)[0]
                frame = draw_boxes(frame, results, model)
                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()