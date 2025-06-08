import os
import tensorflow as tf
import keras_cv
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import ParameterGrid

# 📌 Hyperparamètres de base
IMG_SIZE = 224  # recommandé pour ViT
BATCH_SIZE = 8
EPOCHS = 30

# 📌 Paths
base_dir = os.getcwd()
train_dir = os.path.join(base_dir, "Train")
test_dir = os.path.join(base_dir, "Test")

# 📌 Datasets
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=42
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=42
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(AUTOTUNE)
val_dataset = val_dataset.prefetch(AUTOTUNE)
test_dataset = test_dataset.prefetch(AUTOTUNE)

# 📌 Fonction de création de modèle ViT
def build_vit_model(learning_rate, dropout_rate):
    model = keras_cv.models.VisionTransformerClassifier(
        image_size=IMG_SIZE,
        patch_size=16,
        num_layers=8,
        num_heads=8,
        mlp_dim=2048,
        dropout_rate=dropout_rate,
        num_classes=1,
        activation='sigmoid'
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )
    return model

# 📌 GridSearch params
param_grid = {
    'learning_rate': [1e-4, 5e-5],
    'dropout_rate': [0.1, 0.3]
}

# 📌 Callbacks
checkpoint_path = os.path.join(base_dir, 'models/vit_best_model.keras')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)

# 📌 GridSearch
best_val_acc = 0
best_params = None

for params in ParameterGrid(param_grid):
    print(f"\n🔍 Testing params: {params}")

    model = build_vit_model(params['learning_rate'], params['dropout_rate'])

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[early_stop, model_checkpoint],
        verbose=1
    )

    val_acc = max(history.history['val_accuracy'])
    print(f"Validation Accuracy: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_params = params

print(f"\n✅ Best params: {best_params} with Val Accuracy: {best_val_acc:.4f}")
print(f"Best model saved at: {checkpoint_path}")

# 📌 Évaluation sur Test Dataset
best_model = tf.keras.models.load_model(checkpoint_path)
test_loss, test_accuracy, test_precision, test_recall, test_auc = best_model.evaluate(test_dataset)

print("\n📊 Test Results:")
print(f"Test Loss      : {test_loss:.4f}")
print(f"Test Accuracy  : {test_accuracy * 100:.2f}%")
print(f"Test Precision : {test_precision * 100:.2f}%")
print(f"Test Recall    : {test_recall * 100:.2f}%")
print(f"Test AUC       : {test_auc:.4f}")
