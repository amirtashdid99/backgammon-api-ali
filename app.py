"""
Flask API Server for YOLO Backgammon Detection - Render.com Deployment
Serves the trained YOLOv8 model via REST API
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os
import sys

app = Flask(__name__)

# Configure CORS for production
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Load the trained model
print("🔄 Loading trained YOLO model...")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")

# Model configuration
MODEL_PATH = 'best.pt'
GOOGLE_DRIVE_ID = '1Ov3gwbRkD3Nd9cdfn1WLvDljPGfaAXrh'

def download_model():
    """Download model from Google Drive using gdown"""
    if os.path.exists(MODEL_PATH):
        print(f"✅ Model already exists: {MODEL_PATH}")
        return True
    
    print(f"📥 Downloading model from Google Drive...")
    try:
        import gdown
        url = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}'
        gdown.download(url, MODEL_PATH, quiet=False)
        print(f"✅ Model downloaded: {MODEL_PATH}")
        return True
    except ImportError:
        print("⚠️  gdown not installed, trying urllib...")
        try:
            import urllib.request
            url = f'https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_ID}'
            urllib.request.urlretrieve(url, MODEL_PATH)
            print(f"✅ Model downloaded: {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

# Try to load custom trained model
try:
    if download_model() and os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print(f"✅ Custom model loaded: {MODEL_PATH}")
        print(f"Model size: {os.path.getsize(MODEL_PATH) / 1024 / 1024:.2f} MB")
    else:
        raise FileNotFoundError("Model not available")
except Exception as e:
    print(f"⚠️  Using pretrained model as fallback: {e}")
    model = YOLO('yolov8n.pt')
    print("✅ Pretrained YOLOv8n loaded")

# Class names
CLASS_NAMES = [
    'black_checker',
    'dice_1', 'dice_2', 'dice_3', 'dice_4', 'dice_5', 'dice_6',
    'red_checker',
    'white_checker'
]

@app.route('/', methods=['GET'])
def index():
    """API info page"""
    return jsonify({
        'name': 'Backgammon Detection API',
        'version': '1.0',
        'status': 'running',
        'endpoints': {
            'GET /health': 'Health check',
            'POST /detect': 'Detect objects in image'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    model_exists = os.path.exists(MODEL_PATH)
    return jsonify({
        'status': 'ok',
        'model_loaded': True,
        'model_path': MODEL_PATH,
        'custom_model': model_exists,
        'model_size_mb': round(os.path.getsize(MODEL_PATH) / 1024 / 1024, 2) if model_exists else 0
    })

@app.route('/detect', methods=['POST'])
def detect():
    """
    Detect objects in an image
    Expects: base64 encoded image or file upload
    Returns: JSON with detections and annotated image
    """
    try:
        # Get image from request
        if 'image' in request.files:
            # File upload
            image_file = request.files['image']
            image_bytes = image_file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif request.json and 'image' in request.json:
            # Base64 encoded
            image_data = request.json['image']
            if 'base64,' in image_data:
                image_data = image_data.split('base64,')[1]
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            return jsonify({'error': 'No image provided'}), 400

        if img is None:
            return jsonify({'error': 'Invalid image'}), 400

        # Run inference with optimized settings
        results = model(img, conf=0.25, iou=0.45, verbose=False, device='cpu')
        
        # Parse results and draw annotations
        detections = []
        counts = {}
        
        # Create annotated image
        annotated_img = img.copy()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get confidence and class
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f'class_{cls}'
                
                # Add to detections
                detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': {
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2)
                    }
                })
                
                # Count by class
                counts[class_name] = counts.get(class_name, 0) + 1
                
                # Draw on image
                color = (0, 255, 0)  # Green
                cv2.rectangle(annotated_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Add label
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(annotated_img, label, (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Convert annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated_img)
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')
        annotated_data_url = f"data:image/jpeg;base64,{annotated_base64}"
        
        return jsonify({
            'success': True,
            'detections': detections,
            'total_objects': len(detections),
            'counts': counts,
            'annotated_image': annotated_data_url
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
