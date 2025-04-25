from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
import io
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load trained GRU model for AF detection
model_path = "trained_gru_model.h5"
model = load_model(model_path)

# Image target size for model
TARGET_WIDTH, TARGET_HEIGHT = 70, 75  

@app.route('/upload_csv', methods=['POST'])
def predict_from_csv():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        filename = file.filename.lower()

        # Check file extension and read accordingly
        if filename.endswith('.csv'):
            df = pd.read_csv(file, header=None)
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file, header=None, engine='openpyxl')
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        # Extract PPG signal from the first column
        ppg_signal = df.iloc[:, 0].values

        if len(ppg_signal) < 10:
            return jsonify({"error": "Invalid sensor data"}), 400

        # Perform CWT on entire signal (WITHOUT NORMALIZATION)
        scales = np.arange(1, 65)
        coeffs, _ = pywt.cwt(ppg_signal, scales=scales, wavelet='gaus1')

        # Generate and save scalogram
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(np.abs(coeffs), aspect='auto', cmap='viridis')
        ax.axis('off')

        # Convert to image array
        img_io = io.BytesIO()
        plt.savefig(img_io, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)  # Close figure to free memory
        img_io.seek(0)

        img_array = np.frombuffer(img_io.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # Read as color (BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT)) / 255.0
        img = np.expand_dims(img, axis=0)  # Shape: (1, 75, 70, 3)

        # Predict AF
        prediction = model.predict(img)
        result = "AF DETECTED" if prediction[0][0] > 0.5 else "NON_AF DETECTED"

        return jsonify({"result": result})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
