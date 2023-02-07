import cv2
import numpy as np
from keras.models import load_model

# Load the saved model
model = load_model('model.h5')

# Load the image to be validated
image = cv2.imread("2.jpg")
image = cv2.resize(image, (256, 256))
image = np.array(image) / 255.
image = np.expand_dims(image, axis=0)

# Make a prediction using the model
prediction = model.predict(image)

# Convert the prediction to a binary classification result
result = int(prediction > 0.5)

# Print the result along with the predicted probability
if result == 0:
    print("В изображении нет оружия. (Вероятность: {:.2f}%)".format(prediction[0][0] * 100))
else:
    print("В изображении есть оружие. (Вероятность: {:.2f}%)".format(prediction[0][0] * 100))
