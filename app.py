from flask import Flask, request, jsonify
from flask_cors import CORS
from model_loader import load_model
from utils import preprocess_image
from werkzeug.datastructures import FileStorage
import io
import base64
import os
import uuid

# --- App setup
app = Flask(__name__)
CORS(app)

MODEL_INPUT_SIZES = {
    "mobilenetv2": (128, 128),
    "cnn": (224, 224),  # example, adjust if your cnn uses 224x224
 }

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Preload models cache
_models_cache = {}

def get_model(name):
    if name not in _models_cache:
        _models_cache[name] = load_model(name)
    return _models_cache[name]

# --- Routes


@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Skin Cancer Detection API is running."})

@app.route("/predict/<model_name>", methods=["POST"])
def predict(model_name):
    try:
        model = get_model(model_name)
    except Exception:
        return jsonify({"error": f"Model '{model_name}' is not available."}), 400

    if not request.is_json:
        return jsonify({"error": "Request content-type must be application/json."}), 400

    data = request.get_json()
    file_b64 = data.get('base64file')
    if not file_b64:
        return jsonify({"error": "No base64file key found in JSON body."}), 400

    try:
        if file_b64.startswith('data:'):
            file_b64 = file_b64.split(',', 1)[1]

        decoded = base64.b64decode(file_b64)
        bytes_io = io.BytesIO(decoded)
        bytes_io.name = 'uploaded_image.jpg'  # safe name

        filename_to_save = f"{uuid.uuid4().hex}_uploaded_image.jpg"
        save_path = os.path.join(UPLOAD_FOLDER, filename_to_save)
        with open(save_path, 'wb') as f:
            f.write(decoded)

        file = FileStorage(stream=bytes_io, filename=bytes_io.name, content_type='image/jpg')
    except Exception:
        return jsonify({"error": "Invalid base64 file data."}), 400

    # --- Run prediction
    try:
        file.stream.seek(0)  # reset stream position
        target_size = MODEL_INPUT_SIZES.get(model_name, (224, 224))  # default fallback
        img_array = preprocess_image(file, model_name, target_size= target_size)
        prediction = model.predict(img_array)[0][0]
        threshold = 0.5
        result = "Malignant" if prediction < threshold else "Benign"

        return jsonify({
            "model": model_name,
            "prediction": float(prediction),
            "result": result,
            "saved_filename": filename_to_save
        })
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": "Internal server error during prediction."}), 500

# --- Main entry point
if __name__ == "__main__":
    app.run(debug=True)