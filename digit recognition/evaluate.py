import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load Model
model = tf.keras.models.load_model('DigitsRecognition.h5')

# Evaluate Model
# val_loss, val_acc = model.evaluate(x_test, y_test)
# print(val_loss, val_acc)

# Predictions
image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        img = cv2.resize(img, (28, 28))
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print("This digit is probably a:", np.argmax(prediction))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()

    except Exception as e: 
        print(e)

    finally:
        image_number += 1