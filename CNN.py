from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

height = 180
width = 180
train_path = "dataset/train"
test_path = "raw_set"
num_classes = 10
epochs = 22
channels = 1
batch_size = 5
inputs = keras.Input(shape=(height, width, 3))
train_data = keras.preprocessing.image_dataset_from_directory(
    train_path, labels="inferred", label_mode="int", image_size=(height, width))
# test_data = keras.preprocessing.image_dataset_from_directory(
#    test_path, labels="inferred", label_mode="int", image_size=(height, width))

print("-----------model1----------")

# Apply some convolution and pooling layers
model1 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, kernel_size=(
        2, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
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
# model1.fit(
#    train_data,
#    epochs=epochs,
#    validation_data=test_data
# )
# model1.summary()
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
# model2.fit(
#    train_data,
#    epochs=epochs,
#    validation_data=test_data
# )
# model2.summary()
# Best 0.9643

print("-----------model3----------")

model3 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, kernel_size=5,
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
# model3.fit(
#    train_data,
#    epochs=epochs,
#    validation_data=test_data
# )
# model3.summary()
# Best 0.95

def kfoldValidator(model, trainData):
    for images, labels in trainData.take(1):
        y = np.array(labels)
        X = np.array(images)
        kfd = KFold(n_splits=10, shuffle=True)
        for t, test in kfd.split(X, y):
            trainX, testX = X[t], X[test]
            trainY, testY = y[t], y[test]
            trainY = tf.keras.utils.to_categorical(
                trainY, num_classes=10, dtype='float32'
            )
            testY = tf.keras.utils.to_categorical(
                testY, num_classes=10, dtype='float32'
            )
            model.fit(tf.stack(trainX), trainY, validation_data=(testX, testY), epochs=22)
            
print("model1")
kfoldValidator(model1, train_data)
print("model2")
kfoldValidator(model2, train_data)
print("model3")
kfoldValidator(model3, train_data)
