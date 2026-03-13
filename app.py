import numpy as np
import tflite_runtime.interpreter as tflite
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
MODEL_PATH = 'sign_transformer.tflite' 

# Load TFLite Model
try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ TFLite Model loaded successfully!")

    # Load Sign Labels (Fallback if dataset folder isn't there)
    if os.path.exists("dataset"):
        SIGNS = sorted(os.listdir("dataset"))
    else:
        num_classes = output_details[0]['shape'][1]
        SIGNS = [f"Sign_{i}" for i in range(num_classes)]
except Exception as e:
    print(f"❌ Error loading TFLite model: {e}")

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6)
sequence_buffer = []

def extract_hand_landmarks(results):
    features = np.zeros(FEATURES, dtype=np.float32)
    if not results.multi_hand_landmarks: return None
    idx = 0
    for hand in results.multi_hand_landmarks[:2]:
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32)
        pts -= pts[0]
        pts /= (np.linalg.norm(pts) + 1e-6)
        features[idx:idx + 63] = pts.flatten()
        idx += 63
    return features

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    global sequence_buffer
    try:
        data = request.get_json()
        img_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        landmarks = extract_hand_landmarks(results)

        if landmarks is not None:
            sequence_buffer.append(landmarks)
            sequence_buffer = sequence_buffer[-FRAMES:]
        else:
            sequence_buffer.clear()
            return jsonify({"status": "buffering", "frames_collected": 0})

        if len(sequence_buffer) == FRAMES:
            input_data = np.expand_dims(sequence_buffer, axis=0).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])[0]
            confidence = float(np.max(preds))
            idx = int(np.argmax(preds))

            return jsonify({
                "status": "success" if confidence >= 0.88 else "low_confidence",
                "prediction": SIGNS[idx] if idx < len(SIGNS) else str(idx),
                "confidence": confidence
            })
        return jsonify({"status": "buffering", "frames_collected": len(sequence_buffer)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
