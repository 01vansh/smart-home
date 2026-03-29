"""
Train Emotion CNN Model on FER2013 dataset.
Run this script ONCE to create models/emotion_model.h5
Training takes ~15-30 minutes depending on hardware.
"""

import os
import sys

# Fix Windows encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


def train_emotion_model():
    # Dataset paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(base_dir, 'emotion face detection', 'train')
    test_dir = os.path.join(base_dir, 'emotion face detection', 'test')
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'emotion_model.h5')

    print(f"Train dir: {train_dir}")
    print(f"Test dir: {test_dir}")
    print(f"Save path: {save_path}")

    if not os.path.exists(train_dir):
        print(f"ERROR: Train directory not found at {train_dir}")
        return False

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    print("\nLoading training data...")
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=64,
        class_mode='categorical',
        shuffle=True
    )

    print("Loading test data...")
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=64,
        class_mode='categorical',
        shuffle=False
    )

    print(f"\nClasses: {train_gen.class_indices}")
    print(f"Training samples: {train_gen.samples}")
    print(f"Test samples: {test_gen.samples}")

    # Build CNN model
    print("\nBuilding CNN model...")
    model = Sequential([
        # Block 1
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Dense layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.00001, verbose=1),
        ModelCheckpoint(save_path, save_best_only=True, verbose=1)
    ]

    # Train
    print("\n" + "="*60)
    print("  TRAINING STARTED")
    print("="*60)

    history = model.fit(
        train_gen,
        epochs=30,
        validation_data=test_gen,
        callbacks=callbacks
    )

    # Save final model
    model.save(save_path)
    print(f"\nModel saved to {save_path}")

    # Print final accuracy
    val_loss, val_acc = model.evaluate(test_gen, verbose=0)
    print(f"\nFinal Validation Accuracy: {val_acc*100:.2f}%")
    print(f"Final Validation Loss: {val_loss:.4f}")

    return True


if __name__ == '__main__':
    print("="*60)
    print("  Emotion CNN Model Training")
    print("  Dataset: FER2013 (7 emotion classes)")
    print("="*60)
    train_emotion_model()
