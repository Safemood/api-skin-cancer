import os
import tensorflow as tf

MODEL_PATHS = {
    "cnn": os.path.join(os.getcwd(), "models", "cnn_skin_cancer_model.tflite"),
    "mobilenetv2": os.path.join(os.getcwd(), "models", "mobilenetv2_skin_cancer_model.tflite"),
}

models = {}

def load_model(model_name):
    if model_name not in models:
        if model_name in MODEL_PATHS and os.path.exists(MODEL_PATHS[model_name]):
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=MODEL_PATHS[model_name])
            interpreter.allocate_tensors()
            models[model_name] = interpreter
            print(f"TFLite model '{model_name}' loaded.")
        else:
            raise ValueError(f"Model '{model_name}' not found.")
    return models[model_name]

def predict_tflite(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data
