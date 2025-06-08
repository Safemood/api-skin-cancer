import tensorflow as tf
import numpy as np
from PIL import Image

def preprocess_image(file, model_name):
    img = Image.open(file.stream).convert("RGB")

    if model_name.startswith("cnn"):
        img_size = 128
    else:
        img_size = 224  # for ViT or other models

    img = img.resize((img_size, img_size))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalisation
    img_array = np.expand_dims(img_array, axis=0)  # Batch dimension

    return img_array
