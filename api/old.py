import tensorflow as tf


old_model = tf.keras.models.load_model("models/melanoma_model.h5", compile=False)

# Optionnel : recompiler proprement
old_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Re-save dans le bon format
old_model.save("models/melanoma_model.keras")
