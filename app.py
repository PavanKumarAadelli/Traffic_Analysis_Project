import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

# 1. PAGE SETUP
st.set_page_config(page_title="Traffic Flow Analyzer", layout="wide")
st.title("🚗 Real-Time Traffic Flow Analyzer")
st.write("Upload a traffic video to detect and count vehicles using YOLOv8.")

# 2. LOAD MODEL (We cache this so it doesn't reload every time)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# 3. SIDEBAR SETTINGS
st.sidebar.header("Settings")
confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5)
line_position = st.sidebar.slider("Counting Line Position (%)", 10, 90, 50)

# 4. FILE UPLOADER
uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    # Save uploaded file to a temp file so OpenCV can read it
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    # Get video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Calculate line position based on slider
    line_y = int(height * (line_position / 100))
    
    # Placeholders for the video and stats
    st_frame = st.empty()
    st_markdown = st.empty()
    
    # Variables for counting
    total_count = 0
    counted_ids = []
    
    # Button to start/stop
    if st.sidebar.button("Start Analysis"):
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                st.write("Video Ended")
                break
            
            # RUN DETECTION
            # classes [2,3,5,7] = car, motorcycle, bus, truck
            results = model.track(frame, persist=True, conf=confidence, classes=[2, 3, 5, 7], verbose=False)
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                
                # Visualize the boxes
                annotated_frame = results[0].plot()
                
                # DRAW THE LINE
                cv2.line(annotated_frame, (0, line_y), (width, line_y), (255, 0, 0), 2)
                
                # COUNTING LOGIC
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    if y > line_y and track_id not in counted_ids:
                        total_count += 1
                        counted_ids.append(track_id)
            else:
                annotated_frame = frame # If no detection, show normal frame
                cv2.line(annotated_frame, (0, line_y), (width, line_y), (255, 0, 0), 2)

            # UPDATE UI
            # Convert color BGR to RGB for Streamlit
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st_frame.image(annotated_frame, channels="RGB", use_column_width=True)
            st_markdown.markdown(f"### Total Vehicles Counted: **{total_count}**")

        cap.release()