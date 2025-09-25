#!/usr/bin/env python3
from flask import Flask, request, jsonify, send_from_directory
import torch
from PIL import Image
import io
import base64
from flask_cors import CORS
import pathlib
import os
import cv2
import time

app = Flask(__name__)
CORS(app)

# Ensure static folder exists
STATIC_FOLDER = os.path.join(os.getcwd(), "static")
os.makedirs(STATIC_FOLDER, exist_ok=True)

# --- Model Paths ---
TRAINED_MODEL_FULL_PATH = "./last.pt"
DEFAULT_YOLOV5_MODEL = ""

# --- Globals ---
inference_model = None
current_loaded_model_type = "unknown"
class_name_to_id = {}

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# --- Load Model ---
def load_detection_model():
    global inference_model, current_loaded_model_type

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    try:
        if os.path.exists(TRAINED_MODEL_FULL_PATH):
            model_path = TRAINED_MODEL_FULL_PATH
            current_loaded_model_type = 'trained'
        else:
            model_path = DEFAULT_YOLOV5_MODEL
            current_loaded_model_type = 'default'

        inference_model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        inference_model.conf = 0.5 if current_loaded_model_type == 'trained' else 0.2

        print(f"[INFO] Loaded {current_loaded_model_type} model from: {model_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        inference_model = None

# --- Inference Endpoint for Single Frame ---
@app.route('/detect_frame', methods=['POST'])
def detect_frame_api():
    try:
        data = request.get_json()
        img_b64 = data['image']
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        results = inference_model(img)
        detections_df = results.pandas().xyxy[0]

        faces = []
        model_class_names = inference_model.names
        target_person_name = request.args.get('person_name')

        if target_person_name and current_loaded_model_type == 'trained':
            if target_person_name not in class_name_to_id:
                target_person_name = None

        for _, row in detections_df.iterrows():
            detected_class_id = int(row['class'])
            detected_class_name = model_class_names[detected_class_id]

            if target_person_name and detected_class_name != target_person_name:
                continue

            if current_loaded_model_type == 'default' and detected_class_name != 'person':
                continue

            if current_loaded_model_type == 'trained' and target_person_name is None:
                if detected_class_name not in class_name_to_id:
                    continue

            faces.append({
                "xmin": float(row['xmin']),
                "ymin": float(row['ymin']),
                "xmax": float(row['xmax']),
                "ymax": float(row['ymax']),
                "confidence": float(row['confidence']),
                "class": detected_class_id,
                "name": detected_class_name
            })

        return jsonify({"faces": faces})
    except Exception as e:
        print(f"[ERROR] detect_frame_api failed: {e}")
        return jsonify({"error": str(e), "faces": []}), 500

# --- Inference Endpoint for Video ---
@app.route('/detect_video', methods=['POST'])
def detect_video_api():
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        video_file = request.files['video']

        # Save uploaded video temporarily
        temp_input_path = os.path.join(STATIC_FOLDER, f"input_{int(time.time())}.mp4")
        video_file.save(temp_input_path)

        # Prepare output video path (in static folder)
        output_filename = f"output_{int(time.time())}.mp4"
        temp_output_path = os.path.join(STATIC_FOLDER, output_filename)

        # Open video
        cap = cv2.VideoCapture(temp_input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLOv5 inference
            results = inference_model(frame)
            annotated_frame = results.render()[0]

            out.write(annotated_frame)

        cap.release()
        out.release()

        return jsonify({
            "output_video_path": f"/static/{output_filename}"
        })

    except Exception as e:
        print(f"[ERROR] detect_video_api failed: {e}")
        return jsonify({"error": str(e)}), 500
    
# --- Inference Endpoint for Image ---
@app.route('/detect_image', methods=['POST'])
def detect_image_api():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['image']

        # Save uploaded image temporarily
        input_filename = f"input_{int(time.time())}.jpg"
        temp_input_path = os.path.join(STATIC_FOLDER, input_filename)
        image_file.save(temp_input_path)

        # Prepare output image path (in static folder)
        output_filename = f"output_{int(time.time())}.jpg"
        temp_output_path = os.path.join(STATIC_FOLDER, output_filename)

        # Run YOLOv5 inference
        results = inference_model(temp_input_path)
        results.save(save_dir=STATIC_FOLDER, exist_ok=True)  # YOLO saves annotated image automatically

        # YOLO usually saves under static/ with "input_filename" name
        # To make sure we return the correct output, rename
        detected_output_path = os.path.join(STATIC_FOLDER, os.path.basename(temp_input_path))
        if os.path.exists(detected_output_path):
            os.rename(detected_output_path, temp_output_path)

        return jsonify({
            "output_image_path": f"/static/{output_filename}"
        })

    except Exception as e:
        print(f"[ERROR] detect_image_api failed: {e}")
        return jsonify({"error": str(e)}), 500


# --- Serve static files (processed videos) ---
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_FOLDER, filename)

# --- Initialize ---
if __name__ == '__main__':
    load_detection_model()
    app.run(host='0.0.0.0', port=5000)
