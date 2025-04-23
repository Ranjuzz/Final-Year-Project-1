import numpy as np
import matplotlib.pyplot as plt
from scipy import signal  # this is fine to keep for signal processing if needed
from tensorflow.keras.models import load_model
import os

def convert_npz_to_image(npz_path, save_dir="converted_images"):
    data = np.load(npz_path)

    # Create directories if not exist
    os.makedirs("seizure_signals", exist_ok=True)
    os.makedirs("non_seizure_signals", exist_ok=True)

    print(data.files)  # Check what keys are inside

    X_test = data['test_signals']
    
    # Load model once, not inside the loop
    model = load_model("model/model_v2.keras")

    for i in range(0, 50):
        eeg_signal = X_test[i]
        X = np.array([eeg_signal])[..., np.newaxis]  # (1, 23, 256, 1)
        predictions = model.predict(X)
        print(predictions)

        if predictions[0][1] > 0.5:
            folder = "seizure_signals"
        else:
            folder = "non_seizure_signals"

        filename = f"{folder}/test_signal_{i}.npz"
        np.savez(filename, test_signals=np.array([eeg_signal]))

        # Optional: save as an image too
        plt.figure(figsize=(10, 3))
        plt.plot(eeg_signal.T)
        plt.title(f'Signal {i} - {"Seizure" if predictions[0][1] > 0.5 else "No Seizure"}')
        plt.xlabel('Time steps')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{folder}/test_signal_{i}.png")
        plt.close()

convert_npz_to_image("../Data/eeg-seizure_test.npz")
