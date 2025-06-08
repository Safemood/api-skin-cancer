from flask import Flask, request, jsonify
from flask_cors import CORS
from model_loader import load_model
from utils import preprocess_image

app = Flask(__name__)

CORS(app)


# Preload all models at startup for efficiency
for model_name in [  'cnn', 'mobilenetv2' ]:
    try:
        load_model(model_name)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")

@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Skin Cancer Detection API is running."})

@app.route("/predict/<model_name>", methods=["POST"])
def predict(model_name):
    try:
        model = load_model(model_name)
    except ValueError:
        return jsonify({"error": f"Model '{model_name}' is not available."}), 400



    print( request.files['file'])

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files['file']


    if not file.content_type.startswith('image/'):
        return jsonify({"error": "Invalid file type. Please upload an image."}), 401

    try:
        img_array = preprocess_image(file, model_name)
        prediction = model.predict(img_array)[0][0]
        threshold = 0.5  # default threshold, can be tweaked if needed
        result = "Malignant" if prediction < threshold else "Benign"
        return jsonify({
            "model": model_name,
            "prediction": float(prediction),
            "result": result
        })
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": "Internal server error during prediction."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8082, debug=True)
