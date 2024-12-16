# Image_Classification
The Project was done using Big Data Tools, such as MongoDB for Data Storage and Apache Spark for Data Preprocessing. The image was classified using a Convolution Neural Network(CNN) and Deployed  with Gradio.

# Running Setup
Healthy Vs Rotten Fruits and Vegetable Detection
DATA 603 - FINAL PROJECT

Problem Statement:
The project aims to create a machine-learning model to identify photos of healthy or unhealthy fruits and vegetables. Using deep learning techniques and Convolutional Neural Networks (CNN), the model is trained with TensorFlow and Keras on a labeled dataset containing a wide range of fruits and vegetables in both healthy and rotten states. The project enables real-time predictions, allowing users to upload images of fruits and vegetables for classification.

Key Components of the Project:
Dataset:

The dataset contains images of fruits and vegetables categorized into two states: healthy and rotten.
The dataset is split into three main subsets:
Training Set: Used to train the model.
Test Set: Used to evaluate the model's performance.
Validation Set: Used during training to tune model parameters and prevent overfitting.
Model (fruitmodel.h5):

The model uses the EfficientNetV2_B3 architecture, a highly efficient convolutional neural network (CNN) designed for image classification tasks.
EfficientNetV2_B3 balances accuracy and computational efficiency, utilizing advanced techniques like Fused-MBConv layers and scalable algorithms.
The model is pre-trained on the ImageNet dataset, making it suitable for transfer learning. The trained model, fruit model.h5, is fine-tuned to classify images of healthy and rotten fruits and vegetables.
Jupyter Notebook:

The notebook contains the full workflow, from data preprocessing to model training and evaluation.
It is used for experimentation and evaluating the performance of the model before deployment.
Gradio Interface (for Deployment):

The project uses Gradio to create a simple web interface. Users can upload an image of a fruit or vegetable, and the model will predict whether the fruit/vegetable is healthy or rotten.
The interface displays the classification result and accuracy of the prediction.
How the Project Works:
Image Upload:

The user uploads an image of a fruit or vegetable via the Gradio interface.
Model Prediction:

The uploaded image is processed and passed through the trained CNN model to predict whether it is healthy or rotten.
The predicted class (Healthy or Rotten) and the model's accuracy for the prediction are displayed.
Real-Time Deployment:

The Gradio interface provides real-time predictions. The project can be deployed locally or on a server for broader accessibility.


How the Project Works:
Image Upload:

The user uploads an image of a fruit or vegetable via the Gradio interface.
Model Prediction:

The uploaded image is processed and passed through the trained CNN model to predict whether it is healthy or rotten.
The predicted class (Healthy or Rotten) and the model's accuracy for the prediction are displayed.
Real-Time Deployment:

The Gradio interface provides real-time predictions. The project can be deployed locally or on a server for broader accessibility.


How to Run the Project:
Steps:
Ensure the necessary files (model, dataset, and notebook) are in the correct directory.

Load the trained model in the notebook using the following code:

import tensorflow as tf
# Load the saved model
model = tf.keras.models.load_model('fruitmodel.h5')

# Compile the model with the same configuration used during training
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.AdamW(1e-4),
    metrics=['accuracy']
)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_data)
print(f"The model loss on the test dataset is {round(loss, 4)}.")
print(f"The model accuracy on the test dataset is {round(accuracy, 2) * 100}%.")
Evaluate the model or classify a test image:
python
Copy code
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
    return img_array

def classify_image(img_array, model, test_classes):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return test_classes[predicted_class], predictions

# Example image path
image_path = 'Path to image file'
img_array = preprocess_image(image_path)
predicted_class_name, predictions = classify_image(img_array, model, test_classes)

# Display the result
print(f"Predicted Class: {predicted_class_name}")
print(f"Prediction Confidence: {np.max(predictions) * 100:.2f}%")
Gradio Interface:
python
Copy code
import gradio as gr
from tensorflow.keras.preprocessing import image
import numpy as np

# Define the prediction function
def predict_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    class_labels = {0: "Healthy", 1: "Rotten"}  # Adjust class labels
    result = class_labels[predicted_class]
    accuracy = np.max(predictions[0]) * 100  # Convert to percentage
    return f"Classification: {result}", f"Accuracy: {accuracy:.2f}%"

# Create the Gradio interface
iface = gr.Interface(fn=predict_image,
                     inputs=gr.Image(type="pil"),
                     outputs=[gr.Text(label="Classification"), gr.Text(label="Accuracy")],
                     title="Fruit and Vegetable Disease Classifier",
                     description="Upload an image of a fruit or vegetable to predict whether it's healthy or rotten.")

iface.launch()

Thank You!



