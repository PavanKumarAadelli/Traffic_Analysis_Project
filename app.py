import os
import sys
import tempfile

# Ensure we are using the correct OpenCV backend for server environments
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# --------------------------------------------------------
# CONFIG
# --------------------------------------------------------
st.set_page_config(page_title="Traffic Analyzer", layout="wide")

# --------------------------------------------------------
# MODEL LOADER
# --------------------------------------------------------
@st.cache_resource
def load_yolo_model():
    # Force download of the model if missing
    return YOLO("yolov8n.pt")

model = load_yolo_model()

# --------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------
st.sidebar.title("Settings")
conf_thres = st.sidebar.slider("Confidence", 0.25, 0.90, 0.50)
line_pct = st.sidebar.slider("Line Position (%)", 10, 90, 50)
video_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

# --------------------------------------------------------
# MAIN PROCESSING
# --------------------------------------------------------
if video_file is not None:
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(delete=False) as tfile:
        tfile.write(video_file.read())
        temp_path = tfile.name

    cap = cv2.VideoCapture(temp_path)
    
    # Get dimensions
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_y = int(h * (line_pct / 100))
    
    # UI Holders
    frame_holder = st.empty()
    count_holder = st.empty()
    
    count = 0
    seen_ids = set()
    
    if st.sidebar.button("Start Analysis"):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Tracking
            # persist=True is essential for counting IDs across frames
            results = model.track(
                frame, 
                persist=True, 
                conf=conf_thres, 
                classes=[2, 3, 5, 7], # Car, Moto, Bus, Truck
                verbose=False
            )
            
            # Plotting
            annotated_frame = frame
            
            # Check if tracking found anything
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()
                ids = results[0].boxes.id.int().cpu().tolist()
                
                # Draw boxes and labels
                annotated_frame = results[0].plot()
                
                # Draw Line
                cv2.line(annotated_frame, (0, line_y), (w, line_y), (0, 255, 255), 2)
                
                # Logic
                for box, track_id in zip(boxes, ids):
                    _, y, _, _ = box
                    # Check if crossed line (Top -> Down)
                    if int(y) > line_y:
                        if track_id not in seen_ids:
                            count += 1
                            seen_ids.add(track_id)
            else:
                # No detections, just line
                cv2.line(annotated_frame, (0, line_y), (w, line_y), (0, 255, 255), 2)

            # Update UI
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_holder.image(annotated_frame, channels="RGB", use_column_width=True)
            count_holder.markdown(f"### Counted Vehicles: **{count}**")

        cap.release()
        os.remove(temp_path)
