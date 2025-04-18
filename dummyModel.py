import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import os

# Create fake image data for binary classification (Seizure / No Seizure)
X_train = np.random.rand(200, 64, 64, 3)  # 200 images of size 64x64x3
y_train = np.random.randint(0, 2, 200)
y_train = to_categorical(y_train, 2)

# Define simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train on fake data (for demonstration purposes)
model.fit(X_train, y_train[:, 1], epochs=5, batch_size=16, verbose=1)

# Save model
os.makedirs("model", exist_ok=True)
model.save("model/cnn_model.h5")
print("Model saved to model/cnn_model.h5")
