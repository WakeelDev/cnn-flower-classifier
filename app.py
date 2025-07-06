import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

# STEP 1: Count images in each folder
base_dir = r'C:\Users\klh\Desktop\CNN_Image_Classifier\images'  # ✅ Point to image dataset directory
dirs = os.listdir(base_dir)
print("Classes:", dirs)

count = 0
for dir in dirs:
    files = os.listdir(os.path.join(base_dir, dir))  # ✅ safer path joining
    print(f"{dir} folder has = {len(files)} images")
    count += len(files)
print('Total Images =', count)

# STEP 2: Load and split dataset
img_size = 180
batch = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    seed=123,
    validation_split=0.2,
    subset='training',
    image_size=(img_size, img_size),
    batch_size=batch
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    seed=42,
    validation_split=0.2,
    subset='validation',
    image_size=(img_size, img_size),
    batch_size=batch
)

class_names = train_ds.class_names
print("Class Names:", class_names)

# STEP 3: Cache and prefetch
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# STEP 4: Data Augmentation
data_augmentation = Sequential([
    layers.RandomFlip('horizontal', input_shape=(img_size, img_size, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# STEP 5: Define model
model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    Conv2D(16, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names))
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# STEP 6: Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15
)

model.save('cnn_model.h5')
print("✅ Model saved as cnn_model.h5")
