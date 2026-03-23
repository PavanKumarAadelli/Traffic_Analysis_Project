import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
from ultralytics import YOLO
import supervision as sv

def main():
    # 1. SETUP THE MODEL
    # 'yolov8n.pt' is a small, fast AI model pre-trained to detect objects.
    # The first time you run this, it will download the model automatically from the internet.
    model = YOLO("yolov8n.pt")

    # 2. SETUP THE VIDEO
    # We open the video file you downloaded.
    video_path = "traffic.mp4" 
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video. Check the file name/path.")
        return

    # 3. DEFINE THE COUNTING LINE
    # We need a line on the road to count cars crossing it.
    # Let's draw a line in the middle of the screen.
    # Get video width and height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Start point and End point of the line
    line_start = (0, height // 2) # Middle left
    line_end = (width, height // 2) # Middle right
    
    # Variable to keep track of total cars counted
    total_count = 0
    
    # To remember cars we already counted (so we don't count the same car twice)
    counted_ids = []

    # 4. PROCESS THE VIDEO FRAME BY FRAME
    while True:
        # Read one frame (picture) from the video
        success, frame = cap.read()
        
        # If video ends, break the loop
        if not success:
            break

        # 5. DETECT OBJECTS
        # We ask the model to track objects in this frame.
        # persist=True means it remembers the car from the previous frame.
        # classes=[2, 3, 5, 7] are codes for Car, Motorcycle, Bus, Truck.
        results = model.track(frame, persist=True, classes=[2, 3, 5, 7])

        # 6. DRAW BOXES AND TRACK
        if results[0].boxes.id is not None:
            # Get the detection boxes and their tracking IDs
            boxes = results[0].boxes.xywh.cpu() # Center x, y, width, height
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            # Visualize the boxes on the screen
            frame = results[0].plot()

            # 7. COUNTING LOGIC
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                
                # Check if the center of the car crossed the line
                # y is the vertical position. If y > line_y, it is below the line.
                # We assume cars move top-to-bottom for this example.
                
                current_y = float(y)
                line_y = line_start[1]

                # If car crosses the line AND we haven't counted it yet
                if current_y > line_y and track_id not in counted_ids:
                    total_count += 1
                    counted_ids.append(track_id)

        # Draw the counting line on the video (Blue line)
        cv2.line(frame, line_start, line_end, (255, 0, 0), 2)
        
        # Write the total count on the screen
        cv2.putText(frame, f"Cars Counted: {total_count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 8. SHOW THE VIDEO
        cv2.imshow("Traffic Counter", frame)
        
        # Wait 1 millisecond. Press 'q' to quit.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()