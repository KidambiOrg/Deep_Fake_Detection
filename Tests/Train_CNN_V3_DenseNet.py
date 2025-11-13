"""
Alternative model using DenseNet201 for better gradient flow
DenseNet is better at detecting subtle features due to dense connections
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
)
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

IMAGE_SIZE = (224, 224)  # DenseNet typically uses 224
BATCH_SIZE = 32
EPOCHS = 150
MODEL_SAVE_DIR = r"Models\CNN_V3_FineTuned"
MODEL_NAME = "cnn_v3_finetuned_DenseNet_model.keras"


def build_densenet_model():
    """
    Build DenseNet201 model for subtle artifact detection
    DenseNet has excellent gradient flow and feature reuse
    """
    model = Sequential(name="CNN_V3_DenseNet")
    
    model.add(Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)))
    
    # DenseNet201 - larger and more complex than InceptionResNetV2
    base_model = DenseNet201(
        weights='imagenet',
        include_top=False,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )
    base_model.trainable = False  # Frozen initially
    model.add(base_model)
    
    model.add(GlobalAveragePooling2D())
    
    # Strong feature processing for subtle artifacts
    model.add(BatchNormalization())
    model.add(Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
    model.add(Dropout(0.4))
    
    model.add(BatchNormalization())
    model.add(Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
    model.add(Dropout(0.4))
    
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
    model.add(Dropout(0.3))
    
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
    model.add(Dropout(0.2))
    
    model.add(Dense(1, activation='sigmoid'))
    
    # Use SGD with momentum for better convergence on difficult tasks
    model.compile(
        optimizer=SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    
    return model, base_model


def load_images(folder, label):
    """Load and preprocess images"""
    images = []
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Loading {len(files)} images...")
    for i, img_file in enumerate(files):
        if (i + 1) % max(1, len(files) // 10) == 0:
            print(f"  {i + 1}/{len(files)}")
        
        try:
            img_path = os.path.join(folder, img_file)
            img_bytes = tf.io.read_file(img_path)
            img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
            img = tf.image.resize(img, IMAGE_SIZE)
            img = tf.cast(img, tf.float32) / 255.0
            images.append(img.numpy())
        except:
            continue
    
    labels = [label] * len(images)
    return np.array(images), np.array(labels)


def train_densenet_model(fake_folder, real_folder):
    """Train DenseNet model"""
    print("\n" + "=" * 80)
    print("Training DenseNet201 Model for Subtle Features")
    print("=" * 80)
    
    print("\nLoading data...")
    fake_images, fake_labels = load_images(fake_folder, 0)
    real_images, real_labels = load_images(real_folder, 1)
    
    X = np.concatenate([fake_images, real_images])
    y = np.concatenate([fake_labels, real_labels])
    
    print(f"\nDataset: {len(X)} images ({np.sum(y==0)} fake, {np.sum(y==1)} real)")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(len(X_train))
    train_dataset = train_dataset.batch(BATCH_SIZE)
    
    # Augmentation
    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, 0.15)
        image = tf.image.random_contrast(image, 0.85, 1.15)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label
    
    train_dataset = train_dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
    
    print("\nBuilding model...")
    model, base_model = build_densenet_model()
    print(f"Model parameters: {model.count_params():,}")
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5, verbose=1)
    ]
    
    print("\nPhase 1: Training with frozen base (75 epochs)...")
    history1 = model.fit(train_dataset, validation_data=val_dataset, epochs=75, callbacks=callbacks, verbose=1)
    
    print("\nPhase 2: Fine-tuning top layers...")
    num_layers = len(base_model.layers)
    for layer in base_model.layers[:int(num_layers * 0.6)]:
        layer.trainable = False
    for layer in base_model.layers[int(num_layers * 0.6):]:
        layer.trainable = True
    
    model.compile(
        optimizer=SGD(learning_rate=0.001, momentum=0.9, nesterov=True),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')]
    )
    
    history2 = model.fit(train_dataset, validation_data=val_dataset, epochs=75, callbacks=callbacks, verbose=1)
    
    # Evaluate
    print("\n" + "=" * 80)
    print("=== Final Evaluation ===")
    train_loss, train_acc, _, _, _ = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc, _, _, _ = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"Training: Loss={train_loss:.4f}, Accuracy={train_acc*100:.2f}%")
    print(f"Validation: Loss={val_loss:.4f}, Accuracy={val_acc*100:.2f}%")
    
    preds_train = (model.predict(X_train, verbose=0) > 0.5).astype(int).flatten()
    preds_val = (model.predict(X_val, verbose=0) > 0.5).astype(int).flatten()
    
    print(f"\n=== Per-Class Accuracy ===")
    print("Training:")
    print(f"  Fake: {100*np.mean(preds_train[y_train==0]==0):.2f}%")
    print(f"  Real: {100*np.mean(preds_train[y_train==1]==1):.2f}%")
    print("Validation:")
    print(f"  Fake: {100*np.mean(preds_val[y_val==0]==0):.2f}%")
    print(f"  Real: {100*np.mean(preds_val[y_val==1]==1):.2f}%")
    
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
    model.save(model_path)
    print(f"\n✓ Model saved to: {model_path}")


if __name__ == "__main__":
    fake_folder = r"C:\Kidambi\WorkRelated\DeepFake\sample_images\testthresholdvoting\fake"
    real_folder = r"C:\Kidambi\WorkRelated\DeepFake\sample_images\testthresholdvoting\real"
    
    train_densenet_model(fake_folder, real_folder)
