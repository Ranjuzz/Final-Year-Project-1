from PIL import Image
import numpy as np
import tensorflow as tf

def load_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # For binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def preprocess_image(image_file):
    img = Image.open(image_file).convert("L")  # Convert to grayscale
    img = img.resize((128, 128))  # Resize to (128, 128)
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = img_array.reshape(-1, 128, 128, 1)  # Add batch dimension
    return img_array

def is_valid_eeg_image(image_file) -> bool:
    try:
        img = Image.open(image_file).convert("L")  # Convert to grayscale
        img = img.resize((128, 128))  # Normalize size
        img_array = np.array(img)

        # Check if image has enough variance
        variance = np.var(img_array)
        if variance < 20:  # Tweak threshold as needed
            return False
        
        return False
    except Exception as e:
        return False

def predict_seizure(model, image):
    # Simulating a prediction (normally you'd call model.predict here)
    prediction = model.predict(image)
    return "Seizure Likely" if prediction > 0.5 else "No Seizure"
