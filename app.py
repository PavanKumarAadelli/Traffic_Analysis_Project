import os
import tempfile

# Fix OpenCV threading
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

st.set_page_config(page_title="Traffic Analyzer", layout="wide")
st.title("🚗 Traffic Flow Analyzer")
st.markdown("Upload a video to detect and count vehicles.")

@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

model = load_yolo_model()

st.sidebar.header("Settings")
conf = st.sidebar.slider("Confidence", 0.1, 1.0, 0.5)
line_pct = st.sidebar.slider("Line Position (%)", 10, 90, 50)
video_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if video_file is not None:
    
    # Create temp file
    with tempfile.NamedTemporaryFile(delete=False) as tfile:
        tfile.write(video_file.read())
        temp_path = tfile.name

    # OpenCV Video Capture
    cap = cv2.VideoCapture(temp_path)
    
    # Dimensions
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_y = int(h * (line_pct / 100))
    
    # Placeholders
    vid_place = st.empty()
    count_place = st.empty()
    
    count = 0
    seen_ids = set()
    
    if st.sidebar.button("Start Analysis"):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Track
            results = model.track(frame, persist=True, conf=conf, classes=[2, 3, 5, 7], verbose=False)
            
            annotated_frame = frame
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()
                ids = results[0].boxes.id.int().cpu().tolist()
                
                # Plot boxes
                annotated_frame = results[0].plot()
                
                # Draw Line
                cv2.line(annotated_frame, (0, line_y), (w, line_y), (0, 255, 255), 2)
                
                # Count Logic
                for box, track_id in zip(boxes, ids):
                    _, y, _, _ = box
                    if int(y) > line_y and track_id not in seen_ids:
                        count += 1
                        seen_ids.add(track_id)
            else:
                cv2.line(annotated_frame, (0, line_y), (w, line_y), (0, 255, 255), 2)

            # Convert BGR to RGB
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            vid_place.image(annotated_frame, channels="RGB", use_column_width=True)
            count_place.markdown(f"### Total Count: **{count}**")

        cap.release()
        os.remove(temp_path)
