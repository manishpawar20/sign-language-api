import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import base64
import mediapipe as mp
import os

app = Flask(__name__)
CORS(app)

# ================= CONFIG (Matches your train.py) =================
FRAMES = 30
FEATURES = 126
MODEL_PATH = 'sign_transformer.h5'

# Load Model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Transformer Model loaded successfully!")

    # We need the list of signs to return the actual word, not just an index number.
    # We assume your 'dataset' folder is structured like: dataset/GOOD/stable/...
    # If the folder doesn't exist on the server, we just return the index.
    if os.path.exists("dataset"):
        SIGNS = sorted(os.listdir("dataset"))
    else:
        # Fallback if you didn't copy the dataset folder to the backend
        SIGNS = [f"Sign_{i}" for i in range(model.output_shape[1])]

except Exception as e:
    print(f"❌ Error loading model: {e}")

# Setup MediaPipe Hands (Exactly as in your predict_live.py)
import mediapipe as mp
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Global buffer to hold the 30 frames
sequence_buffer = []


# ================= LANDMARK EXTRACTION (Copied from your code) =================
def extract_hand_landmarks(results):
    features = np.zeros(FEATURES, dtype=np.float32)

    if not results.multi_hand_landmarks:
        return None

    idx = 0
    for hand in results.multi_hand_landmarks[:2]:
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32)
        # Your specific normalization!
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

        # 1. Decode Image from Flutter
        img_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. Process with MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # 3. Extract and Buffer
        landmarks = extract_hand_landmarks(results)

        if landmarks is not None:
            sequence_buffer.append(landmarks)
            # Keep only the last 30 frames
            sequence_buffer = sequence_buffer[-FRAMES:]
        else:
            # If hands drop out of frame, clear the buffer
            sequence_buffer.clear()
            return jsonify({
                "status": "buffering",
                "message": "No hands detected",
                "frames_collected": 0
            })

        # 4. Predict if we have 30 frames
        if len(sequence_buffer) == FRAMES:
            input_data = np.expand_dims(sequence_buffer, axis=0)  # Shape: (1, 30, 126)

            preds = model.predict(input_data, verbose=0)[0]
            confidence = float(np.max(preds))
            idx = int(np.argmax(preds))

            # Use your logic from predict_live.py
            if confidence >= 0.88:  # Your CONF_THRESHOLD
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
            return jsonify({
                "status": "buffering",
                "frames_collected": len(sequence_buffer)
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # This block is ignored by Gunicorn
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port))
