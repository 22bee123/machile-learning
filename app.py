from flask import Flask, render_template, request, jsonify, Response
import cv2
import os
import numpy as np
import time
from datetime import datetime
from ultralytics import YOLO
from pathlib import Path
import base64

app = Flask(__name__)
CORS(app)

# Check if model exists, if not download it
model_path = os.path.join("crop-disease-detection-using-yolov8", "Crop Disease Detection Using YOLOv8", "runs", "detect", "train3", "weights", "best.pt")
if not os.path.exists(model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    # Import and run the download script
    import download_model
    if __name__ == "__main__":
        download_model.download_file_from_google_drive(
            "11VI1oY4g-EF0QFZWnQ5XhNFIu470_N5X", 
            model_path
        )

# Load the trained model
model = YOLO(model_path)

# Global variables
camera = None
confidence_threshold = 0.5
live_detections = []
detection_summary = "No detections yet"

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return camera

def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None

def process_image(image, conf_threshold=0.5):
    """Process a single image with the YOLOv8 model and return results"""
    global live_detections, detection_summary
    # Resize image to 320x320 for lower memory usage
    image_resized = cv2.resize(image, (320, 320))
    results = model(image_resized, conf=conf_threshold)
    
    # Get detection results
    detections = []
    disease_detections = []
    avg_confidence = 0
    
    if len(results[0].boxes) > 0:
        classes = results[0].names
        confidence_values = []
        
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            if cls_id in classes:
                class_name = classes[cls_id]
                detections.append({"class": class_name, "confidence": conf})
                
                # Only count as disease if it's not just a leaf detection
                # Check if class name contains words that indicate actual diseases
                if not class_name.lower().endswith("leaf") and not class_name.lower().endswith("leafs"):
                    disease_detections.append({"class": class_name, "confidence": conf})
                    confidence_values.append(conf)
        
        if confidence_values:
            avg_confidence = sum(confidence_values) / len(confidence_values)
            
        # Update the global variables for front-end updates with actual diseases only
        live_detections = disease_detections
        num_disease_detections = len(disease_detections)
        
        if num_disease_detections > 0:
            detection_summary = f"Found {num_disease_detections} disease detection{'s' if num_disease_detections > 1 else ''}"
        else:
            detection_summary = "No diseases detected - Plant appears healthy"
    else:
        live_detections = []
        detection_summary = "No diseases detected - Plant appears healthy"
    
    # Create annotated image
    annotated_frame = results[0].plot()
    
    return {
        "annotated_image": annotated_frame,
        "detections": detections,
        "disease_detections": disease_detections,
        "num_detections": len(detections),
        "num_disease_detections": len(disease_detections),
        "avg_confidence": avg_confidence
    }

def generate_frames():
    """Generate frames for the video stream"""
    cap = get_camera()
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Process the frame
        result = process_image(frame, confidence_threshold)
        annotated_frame = result["annotated_image"]
        
        # Convert to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_image', methods=['POST'])
def detect_image():
    """Process an image from the client-side webcam"""
    try:
        data = request.json
        image_data = data['image']
        confidence = data.get('confidence', confidence_threshold)
        
        # Remove the data URL prefix
        image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Process the image
        result = process_image(opencv_image, confidence)
        
        # Convert the annotated image to base64 for sending back to client
        _, buffer = cv2.imencode('.jpg', result['annotated_image'])
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'annotated_image': f'data:image/jpeg;base64,{annotated_base64}',
            'detections': result['detections'],
            'disease_detections': result['disease_detections']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/live_status', methods=['GET'])
def live_status():
    """Get the current detection status for the live feed"""
    return jsonify({
        "detections": live_detections,
        "summary": detection_summary
    })

@app.route('/adjust_confidence', methods=['POST'])
def adjust_confidence():
    """Adjust the confidence threshold"""
    global confidence_threshold
    data = request.json
    if 'value' in data:
        confidence_threshold = float(data['value'])
        return jsonify({"confidence_threshold": confidence_threshold})
    return jsonify({"error": "Invalid request"}), 400

@app.route('/shutdown')
def shutdown():
    """Shutdown the server and release resources"""
    release_camera()
    # Only works in development mode
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return 'Server shutting down...'

@app.route('/detect_image', methods=['POST'])
def detect_image():
    try:
        if 'image' in request.files:
            # Image sent as file
            file = request.files['image']
            npimg = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        else:
            # Image sent as base64
            data = request.get_json()
            if data and 'image' in data:
                image_data = data['image']
                header, encoded = image_data.split(',', 1) if ',' in image_data else ('', image_data)
                img_bytes = base64.b64decode(encoded)
                npimg = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            else:
                return jsonify({'error': 'No image provided'}), 400
        conf = float(request.form.get('confidence', 0.5)) if 'confidence' in request.form else float(request.args.get('confidence', 0.5))
        result = process_image(img, conf)
        # Encode annotated image to base64
        _, buffer = cv2.imencode('.jpg', result['annotated_image'])
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({
            'detections': result['detections'],
            'disease_detections': result['disease_detections'],
            'num_detections': result['num_detections'],
            'num_disease_detections': result['num_disease_detections'],
            'avg_confidence': result['avg_confidence'],
            'annotated_image': f"data:image/jpeg;base64,{annotated_b64}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        # Get the PORT from environment variable for Railway
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False)
    finally:
        release_camera()