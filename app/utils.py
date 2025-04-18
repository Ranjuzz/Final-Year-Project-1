import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import streamlit as st 
import os

MODEL_PATH = os.path.join("model", "cnn_model.h5")

@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)

def preprocess_image(img_file, target_size=(64, 64)):
    img = image.load_img(img_file, target_size=target_size, color_mode='rgb')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_seizure(model, processed_image):
    prediction = model.predict(processed_image)
    return "Seizure" if prediction[0][0] > 0.5 else "No Seizure"
