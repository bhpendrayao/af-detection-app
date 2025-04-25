from flask import Flask, request, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)

model_path = "trained_gru_model.h5"
model = load_model(model_path)

# Image target size for model
TARGET_WIDTH, TARGET_HEIGHT = 70, 75  

@app.route('/upload_image', methods=['POST'])
def predict_from_image():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        filename = file.filename.lower()

        if not filename.endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({"error": "Unsupported file format"}), 400

        # Read the image file
        img_bytes = file.read()
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # Read as color image

        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        # Preprocess image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT)) / 255.0
        img = np.expand_dims(img, axis=0)  # Shape: (1, 75, 70, 3)

        # Predict AF
        prediction = model.predict(img)
        result = "AF DETECTED" if prediction[0][0] > 0.5 else "NON_AF DETECTED"

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)