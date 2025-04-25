import cv2
import torch
import numpy as np

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load YOLOv5 model (make sure the model is in the same directory as the script)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt')  # Replace with your model path

# Stream from ESP32 camera
url = 'http://192.168.1.18:81/stream'  # Replace with your ESP32 camera stream URL
cap = cv2.VideoCapture(url)

# Check if the stream opened successfully
if not cap.isOpened():
    print("Error: Could not open camera stream.")
    exit()

while True:
    # Read frame from the stream
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to retrieve frame.")
        break

    # Perform object detection with YOLOv5
    results = model(frame)  # Perform detection

    # Render results on the frame
    results.render()  # Draw bounding boxes and labels on the frame

    # Display the resulting frame
    cv2.imshow("ESP32 Camera Stream with YOLOv5", frame)

    # Wait for key press and exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
