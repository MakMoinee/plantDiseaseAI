import cv2
import torch
import numpy as np
from flask import Flask, Response
import pathlib
import time

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load YOLOv5 model (ensure it's in the same directory as the script)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt')  # Replace with your model path

# Initialize Flask app
app = Flask(__name__)

# Stream from ESP32 camera
url = 'http://192.168.1.7:4747'  # Replace with your ESP32 camera stream URL
cap = cv2.VideoCapture(url)

# Check if the stream opened successfully
if not cap.isOpened():
    print("Error: Could not open camera stream.")
    exit()

def detect_objects(frame):
    """ Perform object detection with YOLOv5, retry on failure """
    try:
        # Perform detection
        results = model(frame)
        results.render()  # Draw bounding boxes and labels on the frame
        return frame  # Return the frame with detections
    except Exception as e:
        # Log the error and retry detection
        print(f"Detection error: {e}. Retrying detection...")
        return None  # Return None to indicate an error occurred

def gen_frames():
    retry_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            continue

        # Try to detect objects and retry if detection fails
        detected_frame = detect_objects(frame)
        
        # Retry if detection failed (e.g., due to memory error)
        while detected_frame is None and retry_count < 3:
            print("Retrying detection...")
            time.sleep(2)  # Wait before retrying detection
            detected_frame = detect_objects(frame)
            retry_count += 1
        
        if detected_frame is not None:
            # Encode frame as JPEG and send it over HTTP
            _, buffer = cv2.imencode('.jpg', detected_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            print("Detection failed after retries. Skipping frame.")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Run the Flask app
