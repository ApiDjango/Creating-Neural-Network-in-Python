from keras.models import load_model
import numpy as np
import cv2

# Load the saved model
model = load_model('model.h5')

# Read an image using OpenCV
img = cv2.imread('2.jpg')


# Preprocess the image
if img is not None:
    img = cv2.resize(img, (225, 225))  # Resize to the input shape of the model
    img = img / 255.  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add an extra dimension for batch size

    # Make predictions
    prediction = model.predict(img)

    # Get the class prediction by taking the index of the maximum value in the prediction array
    predicted_class = np.argmax(prediction[0])
    predicted_class_prob = prediction[0][predicted_class]

    # Print the prediction
    print("Predicted class:", predicted_class)
    print("Predicted class probability:", predicted_class_prob)

else:
 print("Error: Image not found or empty")
