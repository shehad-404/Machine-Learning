import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import numpy as np

# Suppress TensorFlow warnings (optional)
tf.get_logger().setLevel('ERROR')

# Example data for univariate regression
# Replace this with your actual data loading process
x_train = np.array([[1], [2], [3], [4], [5]], dtype=float)
y_train = np.array([[2], [4], [6], [8], [10]], dtype=float)

# Define the model
model = Sequential([
    Input(shape=(1,)),  # Input layer with a shape of (1,) for univariate input
    Dense(10, activation='relu'),  # First hidden layer
    Dense(1)  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=100, verbose=1)

# Make predictions (example)
x_test = np.array([[6], [7]], dtype=float)
predictions = model.predict(x_test)
print("Predictions:", predictions)
