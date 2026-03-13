import numpy as np
try:
    # Use the lightweight runtime for cloud deployment
    import tflite_runtime.interpreter as tflite
except ImportError:
    # Fallback for local testing if you have full tensorflow installed
    import tensorflow.lite as tflite

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import base64
import mediapipe as mp
import os

app = Flask(__name__)
CORS(app)

# ================= CONFIG =================
FRAMES = 30
FEATURES = 126
MODEL_PATH = 'sign_transformer.tflite' # Make sure to upload the .tflite file!

# Load TFLite Model
interpreter = None
input_details = None
output_details = None
SIGNS = []

try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("✅ TFLite Model loaded successfully!")

    # Load Sign Labels
    if os.path.exists("dataset"):
        SIGNS = sorted(os.listdir("dataset"))
    else:
        # Fallback: Replace with your actual list of signs if dataset folder isn't uploaded
        # Example: SIGNS = ["HELLO", "THANK YOU", "GOODBYE"]
        num_classes = output_details[0]['shape'][1]
        SIGNS = [f"Sign_{i}" for i in range(num_classes)]

except Exception as e:
    print(f"❌ Error loading TFLite model: {e}")

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

sequence_buffer = []

# ================= LANDMARK EXTRACTION =================
def extract_hand_landmarks(results):
    features = np.zeros(FEATURES, dtype=np.float32)

    if not results.multi_hand_landmarks:
        return None

    idx = 0
    for hand in results.multi_hand_landmarks[:2]:
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32)
        pts -= pts[0]
        pts /= (np.linalg.norm(pts) + 1e-6)
        features[idx:idx + 63] = pts.flatten()
        idx += 63

    return features

# ================= PREDICTION ENDPOINT =================
@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    global sequence_buffer

    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "No image provided"}), 400

        # 1. Decode Image
        img_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # 3. Extract & Buffer
        landmarks = extract_hand_landmarks(results)

        if landmarks is not None:
            sequence_buffer.append(landmarks)
            sequence_buffer = sequence_buffer[-FRAMES:]
        else:
            sequence_buffer.clear()
            return jsonify({"status": "buffering", "message": "No hands detected", "frames_collected": 0})

        # 4. TFLite Prediction
        if len(sequence_buffer) == FRAMES:
            # Prepare input tensor
            input_data = np.expand_dims(sequence_buffer, axis=0).astype(np.float32)
            
            # Set tensor and invoke
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            # Get results
            preds = interpreter.get_tensor(output_details[0]['index'])[0]
            confidence = float(np.max(preds))
            idx = int(np.argmax(preds))

            if confidence >= 0.88:
                word = SIGNS[idx] if idx < len(SIGNS) else str(idx)
                return jsonify({
                    "status": "success",
                    "prediction": word,
                    "confidence": confidence,
                    "frames_collected": 30
                })
            else:
                return jsonify({
                    "status": "low_confidence",
                    "prediction": "Unknown",
                    "confidence": confidence,
                    "frames_collected": 30
                })
        else:
            return jsonify({"status": "buffering", "frames_collected": len(sequence_buffer)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
