from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def load_cnn_model():
    model = load_model('model/my_cnn_model.keras')
    return model

def preprocess_image(image_file):
    img = Image.open(image_file).convert("L")

    # Resize to (256, 23) — we’ll transpose to (23, 256)
    img = img.resize((256, 23))
    
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    
    # Transpose to (23, 256)
    img_array = img_array.astype(np.float32)
    
    # Add channel dimension -> (23, 256, 1)
    img_array = img_array[..., np.newaxis]

    # Add batch dimension -> (1, 23, 256, 1)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    return img_array


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
    label = "Seizure" if np.argmax(prediction) == 1 else "No Seizure"
    print("Prediction:", label)
    return label
