"""
Training script for CNN V3 FineTuned model based on InceptionResNetV2
This script trains a deepfake detection model using transfer learning with native TensorFlow tensors
Credit: https://github.com/THULASISANDEEP/Fake-Image-Detection-using-Deep-Learning
"""

import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
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
    Load and preprocess images from a folder using native TensorFlow operations
    
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
            # Read and decode image using TensorFlow
            img_bytes = tf.io.read_file(img_path)
            img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
            
            # Ensure correct shape
            img = tf.ensure_shape(img, [None, None, 3])
            
            # Resize using TensorFlow
            img = tf.image.resize(img, IMAGE_SIZE)
            
            # Normalize to [0, 1]
            img = tf.cast(img, tf.float32) / 255.0
            
            # Convert to numpy for compatibility with train_test_split
            images.append(img.numpy())
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
    
    # Create augmented dataset using tf.data
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=len(X_train))
    train_dataset = train_dataset.batch(BATCH_SIZE)
    
    # Apply augmentation using map
    def augment(image, label):
        # Random rotation
        image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
        # Random flip
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        # Random brightness
        image = tf.image.random_brightness(image, max_delta=0.1)
        # Random contrast
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        # Clip values to [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label
    
    train_dataset = train_dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Create validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Train model
    print(f"\nStarting training for {EPOCHS} epochs...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
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
