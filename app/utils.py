from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def load_cnn_model():
    model = load_model('model/model_v2.keras')
    return model

def predict_from_npz(model, npz_data):
    if 'test_signals' not in npz_data:
        raise ValueError("Missing 'test_signals' key in uploaded file.")
    
    X = npz_data['test_signals']
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(X[0].T)
    ax.set_title("EEG Signal")
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Amplitude')
    ax.grid(True)
    st.pyplot(fig)
    X = X[..., np.newaxis]  # Ensure (samples, 23, 256, 1)
    predictions = model.predict(X)
    print(predictions)
    if predictions[0][1] > 0.5:
        label = "Possiblity of Epilepsy"
    else:
        label = "No Possibility"

    return label

def is_valid_eeg_image(image_file) -> bool:
    try:
        img = Image.open(image_file).convert("L")  # Convert to grayscale
        img = img.resize((23, 256))  # Normalize size
        img_array = np.array(img)

        # Check if image has enough variance
        variance = np.var(img_array)
        if variance < 20:  # Tweak threshold as needed
            return False
        
        return True
    except Exception as e:
        return False

def predict_seizure(model, image):
    # Simulating a prediction (normally you'd call model.predict here)
    prediction = model.predict(image)
    print(np.argmax(prediction))
    label = "Seizure" if np.argmax(prediction) == 1 else "No Seizure"
    print("Prediction:", label)
    return label
