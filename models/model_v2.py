import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models



# Load base model (exclude top layers)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base layers initially
base_model.trainable = False

# Build model with embedded normalization
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),  # Regularization
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])



model.compile(
    optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'] 
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('ResNet50-Based Model.h5', monitor='val_accuracy', save_best_only=True)
]
