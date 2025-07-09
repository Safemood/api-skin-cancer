import tensorflow as tf
import numpy as np
from PIL import Image

def preprocess_image(file, model_name=None, target_size=(224, 224)):
    img = Image.open(file.stream).convert("RGB")

    # Always use the passed target_size
    img = img.resize(target_size)

    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array
