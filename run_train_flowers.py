# run_train_flowers.py
# CNN Implementation for Multiclass Flower Classification (5 Species)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np

# --- 1. Model Configuration ---
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 5 # Daisy, Dandelion, Rose, Sunflower, Tulip
CHANNELS = 3 

# --- 2. Data Preparation: Setting up Data Generators ---
print("--- 1. Setting up Data Generators and Preprocessing ---")

# BASE_DIR is the path to your flower_dataset folder
BASE_DIR = 'data/flower_dataset/' 

# Data Augmentation and Rescaling for Training Data
train_datagen = ImageDataGenerator(
    rescale=1./255, # Normalize pixel values to 0-1
    rotation_range=20, # Rotate images up to 20 degrees
    width_shift_range=0.2, # Shift images horizontally
    height_shift_range=0.2, # Shift images vertically
    horizontal_flip=True, # Flip images horizontally
    validation_split=0.2 # Use 20% of the data for validation
)

# Load training data
training_set = train_datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical', # Multiclass classification (more than 2 classes)
    subset='training'
)

# Load validation data
validation_set = train_datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# --- 3. Define CNN Model Architecture (Keras Sequential Model) ---
def create_cnn_model():
    model = Sequential([
        # Block 1: Feature Extraction
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], CHANNELS)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25), 

        # Block 2: Deeper Feature Extraction
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Classification Head (Dense Layers)
        Flatten(), # Convert 2D feature maps to 1D vector
        Dense(128, activation='relu'), # Intermediate dense layer
        Dropout(0.5), 
        Dense(NUM_CLASSES, activation='softmax') # Output layer for 5 classes
    ])
    return model

model = create_cnn_model()
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', # Loss function for multi-class problems
              metrics=['accuracy'])

print("\n--- Model Architecture Summary ---")
model.summary()

# --- 4. Train the Model ---
print("\n--- 4. Training the Model ---")
history = model.fit(
    training_set,
    steps_per_epoch=training_set.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_set,
    validation_steps=validation_set.samples // BATCH_SIZE
)

# --- 5. Evaluate and Report ---
print("\n--- 5. Final Evaluation ---")
# Evaluate on the validation set after training is complete
loss, accuracy = model.evaluate(validation_set, steps=validation_set.samples // BATCH_SIZE)

print(f"\nValidation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

# --- IMPORTANT NEXT STEP ---
# Copy the final Validation Accuracy and Loss values for your documentation.