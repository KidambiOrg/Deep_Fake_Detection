"""
Training script for CNN V3 FineTuned model based on InceptionResNetV2
This script trains a deepfake detection model using transfer learning
Credit: https://github.com/THULASISANDEEP/Fake-Image-Detection-using-Deep-Learning
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import threading

# Thread-safe model saving lock
save_lock = threading.Lock()

# Constants
IMAGE_SIZE = (224, 224)
FAKE_DIR = r"C:\Kidambi\WorkRelated\DeepFake\sample_images\testthresholdvoting\fake"
REAL_DIR = r"C:\Kidambi\WorkRelated\DeepFake\sample_images\testthresholdvoting\real"
MODEL_SAVE_DIR = r"Models\CNN_V3_FineTuned"
BATCH_SIZE = 16
EPOCHS = 100


def load_and_preprocess_images(image_folder, label):
    """
    Load and preprocess images from a folder
    
    Args:
        image_folder: Path to folder containing images
        label: Label to assign (0 for fake, 1 for real)
    
    Returns:
        Tuple of (images array, labels array)
    """
    images = []
    labels = []
    
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Directory not found: {image_folder}")
    
    image_files = [f for f in os.listdir(image_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Loading {len(image_files)} images from {image_folder}...")
    
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img = cv2.resize(img, IMAGE_SIZE)
            img = img.astype('float32') / 255.0
            
            images.append(img)
            labels.append(label)
            
        except Exception as e:
            print(f"Error loading {img_file}: {e}")
            continue
    
    return np.array(images), np.array(labels)


def build_model():
    """
    Build InceptionResNetV2-based model for deepfake detection
    
    Returns:
        Compiled Keras model
    """
    base_model = InceptionResNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(1024, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def main():
    """Main training function"""
    print("=== CNN V3 FineTuned Training Script ===")
    print(f"Fake images directory: {FAKE_DIR}")
    print(f"Real images directory: {REAL_DIR}")
    print(f"Model save directory: {MODEL_SAVE_DIR}")
    print()
    
    # Load fake images
    fake_images, fake_labels = load_and_preprocess_images(FAKE_DIR, 0)
    print(f"Loaded {len(fake_images)} fake images")
    
    # Load real images
    real_images, real_labels = load_and_preprocess_images(REAL_DIR, 1)
    print(f"Loaded {len(real_images)} real images")
    
    # Combine datasets
    X = np.concatenate([fake_images, real_images], axis=0)
    y = np.concatenate([fake_labels, real_labels], axis=0)
    
    # Shuffle dataset
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    print(f"\nTotal dataset size: {X.shape[0]} images")
    print(f"Fake images: {np.sum(y == 0)}")
    print(f"Real images: {np.sum(y == 1)}")
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    
    # Build model
    print("\nBuilding model...")
    model = build_model()
    model.summary()
    
    # Data augmentation for training
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train)
    
    # Train model
    print(f"\nStarting training for {EPOCHS} epochs...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=EPOCHS
    )
    
    # Evaluate final performance
    print("\n=== Final Evaluation ===")
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    
    # Save model (thread-safe) using native Keras format
    with save_lock:
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_SAVE_DIR, 'cnn_v3_finetuned_model.keras')
        model.save(model_path)
        print(f"\nModel saved to: {model_path}")
    
    print("\n=== Training Complete ===")


if __name__ == "__main__":
    main()
