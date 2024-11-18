import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
import os

# Load the trained model
model_path = model_path = '/home/srushti/Documents/RDD/saved_model/retinal_detachment_detection_model.h5'

model = tf.keras.models.load_model(model_path)

# Function to preprocess input image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(299, 299))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale image
    return img_array

# Function to predict and visualize bounding box on image
def detect_retinal_detachment(image_path):
    # Preprocess image
    img_array = preprocess_image(image_path)

    # Run prediction
    predictions = model.predict(img_array)
    classification_pred = predictions[0][0]  # Probability of retinal detachment
    bbox_pred = predictions[1][0]  # Bounding box coordinates

    # Load original image for visualization
    original_img = cv2.imread(image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    height, width, _ = original_img.shape

    # If the model predicts retinal detachment
    if classification_pred > 0.5:  # Threshold can be adjusted
        print("Retinal detachment detected")

        # Scale bounding box coordinates back to original image size
        x, y, w, h = bbox_pred
        x = int(x * width)
        y = int(y * height)
        w = int(w * width)
        h = int(h * height)

        # Draw bounding box
        cv2.rectangle(original_img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle

        # Display result
        plt.figure(figsize=(8, 8))
        plt.imshow(original_img)
        plt.axis("off")
        plt.show()
    else:
        print("No retinal detachment detected")
        plt.imshow(original_img)
        plt.axis("off")
        plt.show()

# Usage example
image_path = '/home/srushti/Documents/RDD/DATASET/TEST/rpimg.jpeg'  # Replace with the path to the image you want to test
detect_retinal_detachment(image_path)

