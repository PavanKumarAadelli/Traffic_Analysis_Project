import os
import sys
import tempfile

# 1. FORCE HEADLESS OPENCV (Safety check)
# This line helps prevent the GL library error before we even import cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1" 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# --------------------------------------------------------
# SETUP
# --------------------------------------------------------
st.set_page_config(page_title="Traffic Flow Analyzer", layout="wide")
st.title("🚗 Traffic Flow Analyzer")
st.write("Upload a video to count vehicles.")

# --------------------------------------------------------
# LOAD MODEL
# --------------------------------------------------------
@st.cache_resource
def load_model():
    # We force the model to download fresh to ensure no corruption
    return YOLO("yolov8n.pt")

model = load_model()

# --------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------
st.sidebar.header("Settings")
conf = st.sidebar.slider("Confidence", 0.1, 1.0, 0.5)
line_pos = st.sidebar.slider("Line Position (%)", 10, 90, 50)
video_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])

# --------------------------------------------------------
# MAIN LOGIC
# --------------------------------------------------------
if video_file is not None and model is not None:
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False) as tfile:
        tfile.write(video_file.read())
        temp_path = tfile.name

    cap = cv2.VideoCapture(temp_path)
    
    # Get video info
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_y = int(height * (line_pos / 100))
    
    # Placeholders
    vid_place = st.empty()
    stat_place = st.empty()
    
    # Counters
    count = 0
    seen_ids = set()
    
    if st.sidebar.button("Start"):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Track
            results = model.track(frame, persist=True, conf=conf, classes=[2, 3, 5, 7], verbose=False)
            
            # Process
            annot_frame = frame
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()
                ids = results[0].boxes.id.int().cpu().tolist()
                
                annot_frame = results[0].plot()
                
                # Draw Line
                cv2.line(annot_frame, (0, line_y), (width, line_y), (0, 255, 255), 2)
                
                # Count
                for box, track_id in zip(boxes, ids):
                    _, y, _, _ = box
                    if int(y) > line_y and track_id not in seen_ids:
                        count += 1
                        seen_ids.add(track_id)
            else:
                cv2.line(annot_frame, (0, line_y), (width, line_y), (0, 255, 255), 2)

            # Display
            annot_frame = cv2.cvtColor(annot_frame, cv2.COLOR_BGR2RGB)
            vid_place.image(annot_frame, channels="RGB", use_column_width=True)
            stat_place.markdown(f"### Count: {count}")

        cap.release()
        os.remove(temp_path)
