import os
import tempfile

# Fix for OpenCV threading issues on Linux
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# --------------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------------
st.set_page_config(
    page_title="Traffic Flow Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚗 Real-Time Traffic Flow Analyzer")
st.markdown("Upload a traffic video to detect and count vehicles using YOLOv8.")

# --------------------------------------------------------
# LOAD MODEL
# --------------------------------------------------------
@st.cache_resource
def get_model():
    """
    Loads the YOLO model.
    We use @st.cache_resource so it only loads once, not every rerun.
    """
    try:
        # 'yolov8n.pt' downloads automatically if not present
        model = YOLO("yolov8n.pt")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = get_model()

# --------------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------------
st.sidebar.header("⚙️ Settings")

# Detection Confidence
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

# Counting Line Position
line_position = st.sidebar.slider("Line Position (%)", 10, 90, 50)

# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv'])

# Information
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Detected Classes:**
- 🚗 Car
- 🏍️ Motorcycle
- 🚌 Bus
- 🚛 Truck
""")

# --------------------------------------------------------
# MAIN APP LOGIC
# --------------------------------------------------------
if uploaded_file is not None and model is not None:
    
    # Save uploaded video to a temporary file
    # We use a temporary file because OpenCV needs a file path, not raw bytes
    with tempfile.NamedTemporaryFile(delete=False) as tfile:
        tfile.write(uploaded_file.read())
        temp_video_path = tfile.name

    # Initialize Video Capture
    cap = cv2.VideoCapture(temp_video_path)
    
    # Get Video Properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Calculate Line Y Coordinate
    line_y = int(height * (line_position / 100))
    
    # UI Placeholders
    video_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    # Counting Variables
    vehicle_count = 0
    processed_ids = set() # Using a set is faster for checking if ID exists
    
    # Start Button
    if st.sidebar.button("▶️ Start Analysis"):
        
        status_msg = st.empty()
        status_msg.info("Processing video... This might take a moment depending on video length.")
        
        while cap.isOpened():
            success, frame = cap.read()
            
            if not success:
                status_msg.success("✅ Video processing finished.")
                break
            
            # 1. RUN DETECTION & TRACKING
            # persist=True keeps track IDs consistent across frames
            results = model.track(
                frame, 
                persist=True, 
                conf=confidence, 
                classes=[2, 3, 5, 7], # Car, Motorcycle, Bus, Truck
                verbose=False
            )
            
            # 2. PROCESS RESULTS
            annotated_frame = frame
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()  # (x_center, y_center, width, height)
                track_ids = results[0].boxes.id.int().cpu().tolist()
                
                # Plot boxes on frame
                annotated_frame = results[0].plot()
                
                # Draw Counting Line
                cv2.line(annotated_frame, (0, line_y), (width, line_y), (0, 255, 255), 2)
                
                # Counting Logic
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    
                    # Logic: If the center of the vehicle crosses the line
                    # Change '>' to '<' if traffic is moving Up
                    if int(y) > line_y:
                        if track_id not in processed_ids:
                            vehicle_count += 1
                            processed_ids.add(track_id)
            
            else:
                # If no boxes found, just draw the line on the raw frame
                cv2.line(annotated_frame, (0, line_y), (width, line_y), (0, 255, 255), 2)

            # 3. UPDATE DISPLAY
            # Convert BGR (OpenCV) to RGB (Streamlit)
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            video_placeholder.image(annotated_frame, channels="RGB", use_column_width=True)
            stats_placeholder.markdown(f"### Total Vehicles: <span style='color:green; font-size:24px;'>{vehicle_count}</span>", unsafe_allow_html=True)

        # Release resources
        cap.release()
        
        # Clean up the temporary file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
