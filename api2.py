#!/usr/bin/env python3
from flask import Flask, request, jsonify
import torch
from PIL import Image
import io
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- Model Paths ---
TRAINED_MODEL_FULL_PATH = "./last.pt"
DEFAULT_YOLOV5_MODEL = ""

# --- Globals ---
inference_model = None
current_loaded_model_type = "unknown"
class_name_to_id = {}

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

# --- Inference Endpoint ---
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

# --- Initialize ---
if __name__ == '__main__':
    import os
    load_detection_model()
    app.run(host='0.0.0.0', port=5000)
