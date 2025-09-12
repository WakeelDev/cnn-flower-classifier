import os
import json
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# === CONFIGURATION ===
DATA_DIR = "images"  # Folder containing subfolders for each flower class
IMG_SIZE = 180
BATCH_SIZE = 32
EPOCHS = 20
MODEL_NAME = "mobilenetv2_model.keras"   # <-- switched to .keras
CLASS_FILE = "class_names.json"

# === STEP 1: Count Images & Get Class Names ===
def count_images(data_dir):
    total = 0
    class_dirs = sorted(os.listdir(data_dir))  # Alphabetical order
    print("Classes:", class_dirs)
    for class_name in class_dirs:
        class_path = os.path.join(data_dir, class_name)
        num_images = len(os.listdir(class_path))
        print(f"{class_name}: {num_images} images")
        total += num_images
    print("Total images:", total)
    return class_dirs

# === STEP 2: Load Dataset ===
def load_dataset(data_dir, img_size, batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_size, img_size),
        batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_size, img_size),
        batch_size=batch_size,
    )
    return train_ds, val_ds

# === STEP 3: Optimize Pipeline ===
def prepare_pipeline(train_ds, val_ds):
    AUTOTUNE = tf.data.AUTOTUNE
    preprocess_layer = layers.Rescaling(1./127.5, offset=-1)  # [-1,1] scaling

    train_ds = train_ds.map(lambda x, y: (preprocess_layer(x), y), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (preprocess_layer(x), y), num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds

# === STEP 4: Build Model (MobileNetV2 Transfer Learning) ===
def build_model(img_size, num_classes):
    base_model = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Freeze base model

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# === STEP 5: Train & Save ===
def train_and_save(model, train_ds, val_ds, epochs, model_name, class_names):
    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.2, patience=2),
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )
    # Save in .keras format
    model.save(model_name)
    print(f"✅ Model saved as {model_name}")

    with open(CLASS_FILE, "w") as f:
        json.dump(class_names, f)
    print(f"✅ Class names saved to {CLASS_FILE}")

    # === STEP 6: Quick Test Predictions ===
    print("\n🔎 Checking predictions on 10 validation images...")
    image_batch, label_batch = next(iter(val_ds))
    preds = model.predict(image_batch[:10])
    for i in range(10):
        true_label = class_names[label_batch[i].numpy()]
        pred_label = class_names[np.argmax(preds[i])]
        confidence = np.max(preds[i])
        print(f"Image {i+1}: True={true_label}, Pred={pred_label}, Conf={confidence:.2f}")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    class_names = count_images(DATA_DIR)
    train_ds, val_ds = load_dataset(DATA_DIR, IMG_SIZE, BATCH_SIZE)
    train_ds, val_ds = prepare_pipeline(train_ds, val_ds)
    model = build_model(IMG_SIZE, len(class_names))
    train_and_save(model, train_ds, val_ds, EPOCHS, MODEL_NAME, class_names)
