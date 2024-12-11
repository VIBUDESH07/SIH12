
import eventlet
eventlet.monkey_patch()

from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import base64
import numpy as np
from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof
from deepface import DeepFace
from pymongo import MongoClient

# Flask and SocketIO setup
app = Flask(__name__)
# Enable CORS with specific allowed origins
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Ensure SocketIO allows the same CORS policy

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# MongoDB setup
client = MongoClient("mongodb+srv://vibudesh:040705@cluster0.bojv6ut.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["SIH"]  # Database name
collection = db["face"]  # Collection name

# Model initialization
face_detector = YOLOv5('saved_models/yolov5s-face.onnx')
anti_spoof = AntiSpoof('saved_models/AntiSpoofing_bin_1.5_128.onnx')

# Session management
client_aadhar_map = {}

def fetch_known_face(aadhaar_number):
    """Fetch the known face image for the given Aadhaar number."""
    user_data = collection.find_one({"roll_number": aadhaar_number})
    if not user_data:
        raise ValueError(f"Aadhaar number {aadhaar_number} not found in the database.")
    
    # Extract the Binary field containing the base64 data
    binary_data = user_data["image"]
    
    # Convert the binary data into base64 string
    base64_data = base64.b64encode(binary_data).decode('utf-8')
    
    # Decode the base64 string to raw binary data
    img_data = base64.b64decode(base64_data)
    
    # Convert the binary data to a NumPy array
    np_arr = np.frombuffer(img_data, np.uint8)
    
    # Decode the NumPy array into an image
    known_face_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # Decode image data
    
    return known_face_img

def increased_crop(img, bbox: tuple, bbox_inc: float = 1.5):
    real_h, real_w = img.shape[:2]
    x, y, w, h = bbox
    w, h = w - x, h - y
    l = max(w, h)
    xc, yc = x + w / 2, y + h / 2
    x, y = int(xc - l * bbox_inc / 2), int(yc - l * bbox_inc / 2)
    x1 = 0 if x < 0 else x
    y1 = 0 if y < 0 else y
    x2 = real_w if x + l * bbox_inc > real_w else x + int(l * bbox_inc)
    y2 = real_h if y + l * bbox_inc > real_h else y + int(l * bbox_inc)
    img = img[y1:y2, x1:x2, :]
    img = cv2.copyMakeBorder(img, y1 - y, int(l * bbox_inc - y2 + y), x1 - x, int(l * bbox_inc) - x2 + x, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img

def make_prediction(img, face_detector, anti_spoof):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bbox = face_detector([img])[0]
    if bbox.shape[0] > 0:
        bbox = bbox.flatten()[:4].astype(int)
    else:
        return None
    pred = anti_spoof([increased_crop(img, bbox, bbox_inc=1.5)])[0]
    score = pred[0][0]
    label = np.argmax(pred)
    return bbox, label, score

def decode_base64_image(image_base64):
    """Decode base64 image data into an OpenCV image."""
    image_data = base64.b64decode(image_base64.split(',')[1])
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

def process_image(image, aadhaar_number):
    """Process the uploaded image for face matching."""
    
    # Use the session ID (client_id) to get the stored Aadhaar number
    client_id = request.sid

    # Fetch the stored known_face_img using the session ID
    client_data = client_aadhar_map.get(client_id)

    # Ensure that the session has the Aadhaar number and associated face image
    if not client_data or "known_face_img" not in client_data:
        return "Known face image not found for this Aadhaar number."

    known_face_img = client_data["known_face_img"]
    
    # Now perform face detection and verification
    pred = make_prediction(image, face_detector, anti_spoof)
    if pred is not None:
        (x1, y1, x2, y2), label, score = pred
        face_crop = image[y1:y2, x1:x2]
      
        if label == 0 and score > 0.5:  # Ensuring it's a real face
            try:
                result = DeepFace.verify(face_crop, known_face_img, model_name="VGG-Face")
                if result["verified"]:
                    return "REAL and MATCHED"
                else:
                    return "NOT MATCHING"
            except Exception as e:
                return f"Processing..."
        else:
            return "FAKE"
    return "No Face Detected"

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print(f"Client connected: {request.sid}")

@socketio.on('aadhaar_number')
def handle_aadhaar(data):
    """Store the Aadhaar number for the connected client."""
    aadhaar_number = data.get('aadhaar')
    if aadhaar_number:
        client_id = request.sid  # Unique session ID for the client
        client_aadhar_map[client_id] = {"aadhaar": aadhaar_number}
        print(f"Aadhaar received: {aadhaar_number} from client: {client_id}")
        
        try:
            # Fetch and store the known face image for the Aadhaar number
            known_face_img = fetch_known_face(aadhaar_number)
            client_aadhar_map[client_id]["known_face_img"] = known_face_img
        except Exception as e:
            print(f"Error fetching known face image: {e}")
    else:
        print("No Aadhaar number provided")

@socketio.on('process_image')
def handle_process_image(data):
    """Process the image sent by the client."""
    client_id = request.sid
    image_base64 = data.get('image')

    if not image_base64:
        emit('response', {"message": "No image provided."})
        return

    if client_id not in client_aadhar_map:
        emit('response', {"message": "No Aadhaar number linked to the session."})
        return

    aadhaar_number = client_aadhar_map[client_id]["aadhaar"]
    try:
        image = decode_base64_image(image_base64)
        result_message = process_image(image, aadhaar_number)
        emit('response', {"message": result_message})
    except Exception as e:
        emit('response', {"message": f"Error: {str(e)}"})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    client_id = request.sid
    if client_id in client_aadhar_map:
        del client_aadhar_map[client_id]  # Clean up stored Aadhaar and face image
    print(f"Client {client_id} disconnected")

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)
