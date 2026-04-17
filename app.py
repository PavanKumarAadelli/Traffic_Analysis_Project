import os
import tempfile

# Fix for potential OpenCV/KMP library conflict on some systems
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# 1. PAGE SETUP
st.set_page_config(page_title="Traffic Flow Analyzer", layout="wide")
st.title("🚗 Real-Time Traffic Flow Analyzer")
st.write("Upload a traffic video to detect and count vehicles using YOLOv8. Adjust the settings in the sidebar and click **Start Analysis**.")

# 2. LOAD MODEL (Cached to avoid reloading on every interaction)
@st.cache_resource
def load_model():
    # 'yolov8n.pt' is the nano version (fastest). 
    # It will download automatically on the first run.
    return YOLO("yolov8n.pt")

model = load_model()

# 3. SIDEBAR SETTINGS
st.sidebar.header("⚙️ Settings")
confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.05)
line_position = st.sidebar.slider("Counting Line Position (%)", 10, 90, 50)

# Information about classes being detected
st.sidebar.markdown("""
**Detecting:**
- 🚗 Cars (Class 2)
- 🏍️ Motorcycles (Class 3)
- 🚌 Buses (Class 5)
- 🚛 Trucks (Class 7)
""")

# 4. FILE UPLOADER
uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv'])

if uploaded_file is not None:
    # Create a temporary file to store the uploaded video bytes
    # delete=False allows us to open it with OpenCV
    with tempfile.NamedTemporaryFile(delete=False) as tfile:
        tfile.write(uploaded_file.read())
        temp_video_path = tfile.name

    # Open the video file using OpenCV
    cap = cv2.VideoCapture(temp_video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Calculate the Y coordinate of the counting line
    line_y = int(height * (line_position / 100))
    
    # Placeholders for video display and statistics
    st_frame = st.empty()
    st_stats = st.empty()
    
    # Variables for counting logic
    total_count = 0
    counted_ids = [] # Keep track of IDs that have already crossed the line
    
    start_btn = st.sidebar.button("▶️ Start Analysis")
    
    if start_btn:
        # Status bar
        status_text = st.empty()
        status_text.text("Processing video... Please wait.")
        
        while cap.isOpened():
            success, frame = cap.read()
            
            if not success:
                status_text.text("✅ Video processing finished.")
                break
            
            # RUN DETECTION & TRACKING
            # persist=True ensures tracking IDs remain consistent across frames
            results = model.track(
                frame, 
                persist=True, 
                conf=confidence, 
                classes=[2, 3, 5, 7], # Filter for Car, Motorcycle, Bus, Truck
                verbose=False
            )
            
            # Check if any boxes were detected
            if results[0].boxes is not None and results[0].boxes.id is not None:
                # Get boxes (x_center, y_center, width, height)
                boxes = results[0].boxes.xywh.cpu()
                
                # Get tracking IDs
                track_ids = results[0].boxes.id.int().cpu().tolist()
                
                # Plot results (draws boxes and labels)
                annotated_frame = results[0].plot()
                
                # DRAW THE COUNTING LINE
                cv2.line(annotated_frame, (0, line_y), (width, line_y), (0, 255, 255), 2)
                
                # COUNTING LOGIC
                # Iterate over detected boxes
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    
                    # Logic: If center of object (y) is below the line
                    # AND we haven't counted this ID yet
                    if int(y) > line_y and track_id not in counted_ids:
                        total_count += 1
                        counted_ids.append(track_id)
                        
                        # Optional: Visual feedback (print to terminal)
                        # print(f"Vehicle ID {track_id} crossed the line. Count: {total_count}")
            else:
                # No detections in this frame
                annotated_frame = frame
                cv2.line(annotated_frame, (0, line_y), (width, line_y), (0, 255, 255), 2)

            # UPDATE UI
            # Convert color from BGR (OpenCV) to RGB (Streamlit)
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Display the frame
            st_frame.image(annotated_frame, channels="RGB", use_column_width=True)
            
            # Display the count
            st_stats.markdown(f"### Total Vehicles Counted: <span style='color:green'>{total_count}</span>", unsafe_allow_html=True)

        cap.release()
        
        # Clean up the temporary file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path) #test
