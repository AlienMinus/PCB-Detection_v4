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

# Webcam or image upload
option = st.radio("Choose input source:", ("Webcam", "Image/Video Upload"))

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
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls[0])]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if cap:
            cap.release()
else:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "mp4"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        results = model(img)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Detected Components", use_column_width=True)