# importing libraries
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np


# function to process the image and make the prediction
def import_and_predict(file, model):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)  # converting the file to a numpy array
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # using opencv to read the image data
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converting the color from BGR to RGB
    img = cv2.resize(img, (224, 224))  # resize the image to 224 x 224
    img = np.expand_dims(img, axis=0)  # add batch dimension
    prediction = model.predict(img)  # making the prediction
    return prediction


# loading the h5 model from the 'models' directory and saving it in a cache memory
@st.cache(allow_output_mutation=True)  # to prevent loading the model multiple times every time the app is run
def load_h5_model():  # function to load the model
    model_path = './models/efficientnet_model_0769.h5'  # selecting the desired model
    model = load_model(model_path)
    return model

with st.spinner('Model is being loaded..'):  # to show a spinner in the Streamlit app UI while the model is being loaded
    model = load_h5_model()

# providing a title for the website
st.write("""
         # Facial Expression Recognition from Images
         """
         )

# loading the image from the user
file = st.file_uploader("Please upload an image containing a facial expression", type=["jpg", "png"])

# making the prediction
classes = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

if file is None:
    st.text("Please upload an image file")
else:
    st.image(file, caption='Uploaded Image.', use_column_width=True)  # to display the image
    predictions = import_and_predict(file, model)  # making the predictions
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    st.write(f"This image most likely belongs to {classes[predicted_class]} with a {100 * confidence:.2f}% confidence.")
