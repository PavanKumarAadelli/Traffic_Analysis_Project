import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
from ultralytics import YOLO
import supervision as sv

def main():
    model = YOLO("yolov8n.pt")
    video_path = "traffic.mp4" 
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video. Check the file name/path.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    line_start = (0, height // 2) # Middle left
    line_end = (width, height // 2) # Middle right
    
    total_count = 0
    
    counted_ids = []

    while True:
        success, frame = cap.read()
        
        if not success:
            break

        results = model.track(frame, persist=True, classes=[2, 3, 5, 7])

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu() # Center x, y, width, height
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            frame = results[0].plot()

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                
                
                current_y = float(y)
                line_y = line_start[1]

                if current_y > line_y and track_id not in counted_ids:
                    total_count += 1
                    counted_ids.append(track_id)

        cv2.line(frame, line_start, line_end, (255, 0, 0), 2)
        
        cv2.putText(frame, f"Cars Counted: {total_count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Traffic Counter", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
