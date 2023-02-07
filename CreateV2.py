import cv2
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Build a convolutional neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load the training data from the "train" folder
train_data = []
train_labels = []
for filename in os.listdir('train'):
    image = cv2.imread(os.path.join('train', filename))
    image = cv2.resize(image, (256, 256))
    train_data.append(image)
    label = 1
    train_labels.append(label)
train_data = np.array(train_data) / 255.
train_labels = np.array(train_labels)

# Load the validation data from the "val" folder
val_data = []
val_labels = []
for filename in os.listdir('val'):
    image = cv2.imread(os.path.join('val', filename))
    image = cv2.resize(image, (256, 256))
    val_data.append(image)
    label = 1
    val_labels.append(label)
    
val_data = np.array(val_data) / 255.
val_labels = np.array(val_labels)

# Train the model
history = model.fit(
    train_data,
    train_labels,
    batch_size=64,
    epochs=100,
    validation_data=(val_data, val_labels)
)

# Save the model
model.save('model.h5')

# Plot the training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()