# convert_to_tflite.py

import tensorflow as tf
import os

# Directory to save TFLite models
TFLITE_FOLDER = 'models'
os.makedirs(TFLITE_FOLDER, exist_ok=True)

# List of your models and their filenames
models = [
    ('models/cnn_skin_cancer_model.keras', 'cnn_skin_cancer_model.tflite'),
    ('models/mobilenetv2_skin_cancer_model.keras', 'mobilenetv2_skin_cancer_model.tflite'),
]

def convert_to_tflite(keras_model_path, tflite_model_path):
    print(f"Converting {keras_model_path} → {tflite_model_path}")

    # Load model (supports .keras or .h5)
    model = tf.keras.models.load_model(keras_model_path)

    # Convert to TFLite with optimizations
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enable quantization
    tflite_model = converter.convert()

    # Save TFLite model
    tflite_model_path_full = os.path.join(TFLITE_FOLDER, tflite_model_path)
    with open(tflite_model_path_full, 'wb') as f:
        f.write(tflite_model)

    # Print size
    size_mb = os.path.getsize(tflite_model_path_full) / (1024 * 1024)
    print(f"✅ Saved {tflite_model_path} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    for keras_path, tflite_path in models:
        if os.path.exists(keras_path):
            convert_to_tflite(keras_path, tflite_path)
        else:
            print(f"⚠️ Model file not found: {keras_path}")
