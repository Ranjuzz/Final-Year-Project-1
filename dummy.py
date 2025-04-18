import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

def convert_npz_to_image(npz_path, save_dir="converted_images"):

    data = np.load(npz_path)

    X_train = data['train_signals']  # EEG signals
    y_train = data['train_labels']   # Labels
    j = 0
    for i in range(5):
        plt.figure(figsize=(10, 3))
        plt.plot(X_train[i].T)
        plt.title(f'EEG Signal {i+1} - Label: {y_train[i]}')
        plt.xlabel('Time steps')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'eeg_signal_{i}.png')
        j=i

    image_path = os.path.join(save_dir, f"eeg_signal_{j}.png")
    return image_path


img_path = convert_npz_to_image("../Data/eeg-predictive_train.npz")
print("Saved image to:", img_path)
