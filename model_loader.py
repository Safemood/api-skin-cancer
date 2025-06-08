import os
import tensorflow as tf

MODEL_PATHS = {
    "cnn": os.path.join(os.getcwd(), "models", "cnn_skin_cancer_model.keras"),
    "mobilenetv2": os.path.join(os.getcwd(), "models", "mobilenetv2_skin_cancer_model.keras"),
}

models = {}

def load_model(model_name):
    if model_name not in models:
        if model_name in MODEL_PATHS:
            models[model_name] = tf.keras.models.load_model(MODEL_PATHS[model_name])
            print(f"Model '{model_name}' loaded.")
        else:
            raise ValueError(f"Model '{model_name}' not found.")
    return models[model_name]
