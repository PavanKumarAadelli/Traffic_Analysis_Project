# 🚗 Real-Time Traffic Flow Analyzer

An advanced Computer Vision application designed to optimize traffic management. This project uses the YOLOv8 object detection model to detect, track, and count vehicles in real-time video feeds. It simulates smart traffic systems that can dynamically adjust signal timers based on vehicle density.

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red?logo=streamlit)

---

## 🎥 Demo



https://github.com/user-attachments/assets/5b10a001-77ca-49f1-9e8c-6b908a497224


---

## 🚀 Key Features

- **Real-Time Object Detection:** Utilizes the state-of-the-art YOLOv8 model for fast and accurate vehicle detection.
- **Object Tracking:** Implements ID persistence to track unique vehicles across frames, preventing double-counting.
- **Virtual Counting Line:** Dynamic line drawing to count vehicles crossing a specific threshold.
- **Dual Interface:**
  1. **Script Mode:** High-performance local processing using OpenCV.
  2. **Web App Mode:** Interactive user interface built with Streamlit for easy demonstration.
- **Customizable Settings:** Adjustable confidence thresholds and counting line positions via the UI.

---

## 🛠️ Tech Stack & Libraries

- **Language:** Python 3.9
- **Computer Vision:** OpenCV (cv2)
- **Deep Learning Model:** YOLOv8n (Nano version for speed)
- **Framework:** Ultralytics, Streamlit
- **Data Handling:** NumPy, Pandas

---

## ⚙️ Installation & Setup

Follow these steps to run the project on your local machine.

### 1. Clone the Repository
```bash
git clone https://github.com/PavanKumarAadelli/Traffic-Flow-Analyzer.git
cd Traffic-Flow-Analyzer
```

### 2. Create a Virtual Environment (Recommended)
```bash
conda create -n traffic_project python=3.9
conda activate traffic_project
```

### 3. Install Dependencies
Install all required libraries using the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

This project offers two ways to run the analysis:

### Option A: Web Application (Streamlit)
Best for presentations and live demos.
```bash
streamlit run app.py
```
*Then, upload a traffic video file via the sidebar and click "Start Analysis".*

### Option B: Local Script (OpenCV)
Best for high-speed processing on your local machine.
1. Place your video file in the project folder and name it `traffic.mp4`.
2. Run the script:
```bash
python main.py
```
3. Press `q` to quit the application.

---

## 🧠 How It Works (The Logic)

1.  **Frame Extraction:** The video is read frame by frame using OpenCV.
2.  **Detection & Tracking:** Each frame is passed to the YOLOv8 model. The model identifies objects and assigns unique IDs to each vehicle (Car, Truck, Bus, Motorcycle).
3.  **Coordinate Mapping:** The center point (centroid) of each detected bounding box is calculated.
4.  **Counting Logic:** A virtual horizontal line is drawn on the frame.
    *   If a vehicle's centroid crosses this line (moving top-to-bottom), the counter increments.
    *   The system checks if the ID has already been counted to ensure accuracy.

---

## 📂 Project Structure

```
Traffic-Flow-Analyzer/
│
├── main.py              # Local script for OpenCV processing
├── app.py               # Streamlit web application interface
├── requirements.txt     # List of dependencies
├── yolov8n.pt           # Model weights (auto-downloaded on first run)
└── README.md            # Project documentation
```

---

## 🔮 Future Scope

- **Speed Estimation:** Calculate the speed of vehicles based on pixel movement per frame.
- **License Plate Recognition (ANPR):** Integrate OCR to identify license plates for security purposes.
- **Traffic Light Logic:** Build a logic gate to simulate turning lights Green/Red based on queue length.

---

## 🤝 Connect with Me

**[PavanKumar Aadelli]**
*   LinkedIn: [https://www.linkedin.com/in/pavan-kumar-aadelli-1998043a0/]
*   Email: [pkaadelli@gmail.com]
```


Now you have a professional GitHub repository ready to upload
