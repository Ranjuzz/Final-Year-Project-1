import numpy as np
import os
import matplotlib.pyplot as plt

# Load the data
data = np.load("../Data/eeg-predictive_train.npz")  # Replace with your actual .npz file path
X_test = data['train_signals']
y_test = data['train_labels']

# Create directories if not exist
os.makedirs("seizure_signals", exist_ok=True)
os.makedirs("non_seizure_signals", exist_ok=True)

# Save each signal separately with proper classification
for i in range(0, 50):
    signal = X_test[i]
    label = y_test[i]

    if label == 1:
        folder = "seizure_signals"
        filename = f"{folder}/test_signal_{i}.npz"
        np.savez(filename, test_signals=np.array([signal]))
    else:
        folder = "non_seizure_signals"

    # filename = f"{folder}/test_signal_{i}.npz"
    # np.savez(filename, test_signals=np.array([signal]))

    # Optional: also save as image
    plt.figure(figsize=(10, 3))
    plt.plot(signal.T)
    plt.title(f'EEG Signal {i} - Label: {"Seizure" if label == 1 else "No Seizure"}')
    plt.xlabel('Time Steps')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{folder}/test_signal_{i}.png")
    plt.close()
