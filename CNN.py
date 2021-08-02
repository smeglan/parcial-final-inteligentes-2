import numpy as np
import tensorflow as tf
from tensorflow import keras
height = 180
width = 180
train_path = "dataset/train"
test_path = "raw_set"
num_classes = 10
epochs = 22
batch_size = 5
inputs = keras.Input(shape=(height, width, 3))
train_data = keras.preprocessing.image_dataset_from_directory(
    train_path, batch_size=64, labels="inferred", label_mode="categorical", image_size=(height, width))
test_data = keras.preprocessing.image_dataset_from_directory(
    test_path, batch_size=64, labels="inferred", label_mode="categorical", image_size=(height, width))

print("-----------model1----------")

# Apply some convolution and pooling layers
model1 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, kernel_size=(
        2, 3), activation='relu', input_shape=[180, 180, 3]),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, kernel_size=(2, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.20),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model1.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy', tf.keras.metrics.Recall()]
)
model1.fit(
    train_data,
    epochs=epochs,
    validation_data=test_data
)
model1.summary()
# model1.save('best_model.h5')
# best 0.9714

print("-----------model2----------")

model2 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, kernel_size=(
        3, 3), activation='relu', input_shape=[180, 180, 3]),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.10),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='linear'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model2.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy', tf.keras.metrics.Recall()]
)
model2.fit(
    train_data,
    epochs=epochs,
    validation_data=test_data
)
model2.summary()
# Best 0.9643

print("-----------model3----------")

model3 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, kernel_size=5, filters=12,
                           activation='relu', input_shape=[180, 180, 3]),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.10),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(16),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model3.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy', tf.keras.metrics.Recall()],
)
model3.fit(
    train_data,
    epochs=epochs,
    validation_data=test_data
)
model3.summary()
#Best 0.95
