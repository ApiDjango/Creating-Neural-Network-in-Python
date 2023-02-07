import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

# Create the neural network architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(225,225,3)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load the training and validation images using OpenCV
train_images = []
train_labels = []
val_images = []
val_labels = []

i=0
for filename in os.listdir('train'):
    img = cv2.imread(os.path.join('train', filename))
    if img is not None:
        img = cv2.resize(img, (225,225))
        train_images.append(img)
        train_labels.append(i)
        i+=1
i=0
for filename in os.listdir('val'):
    img = cv2.imread(os.path.join('val', filename))
    if img is not None:
        img = cv2.resize(img, (225,225))
        val_images.append(img)
        val_labels.append(i)
        i+=1

# Convert the lists to numpy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)
val_images = np.array(val_images)
val_labels = np.array(val_labels)

# Normalize the data
train_images = train_images / 255.0
val_images = val_images / 255.0

# Train the model
history = model.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels))

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
