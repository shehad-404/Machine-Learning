import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Load the data
file_path = '/home/shehad/Desktop/MLP/CNN/monthly-beer-production-in-austr.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the data
print(data.head())

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data['Monthly_beer_production_scaled'] = scaler.fit_transform(data[['Monthly beer production']])


# Create sequences for the CNN model
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# Define the sequence length
sequence_length = 12  # Number of months in each input sequence (e.g., 1 year)

# Prepare the data for supervised learning
data_values = data['Monthly_beer_production_scaled'].values

X, y = create_sequences(data_values, sequence_length)

# Split the data into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Print shapes of the datasets
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_test: {y_test.shape}")

# Reshape data for the CNN
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define the CNN Model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1)  # Output layer for regression
])

# Compile the Model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the Model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate the Model
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Inverse transform predictions
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)

# Inverse transform actual values
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot training vs test predictions
plt.figure(figsize=(14, 7))
plt.plot(y_train_inv, label='Actual Training Data', color='blue')
plt.plot(np.arange(sequence_length, len(train_predictions) + sequence_length),
         train_predictions, label='Predicted Training Data', color='orange')
plt.plot(np.arange(len(train_predictions) + sequence_length * 2,
                   len(train_predictions) + sequence_length * 2 + len(test_predictions)),
         y_test_inv, label='Actual Test Data', color='green')
plt.plot(np.arange(len(train_predictions) + sequence_length * 2,
                   len(train_predictions) + sequence_length * 2 + len(test_predictions)),
         test_predictions, label='Predicted Test Data', color='red')
plt.title('Training vs Test Predictions')
plt.xlabel('Time Steps')
plt.ylabel('Electric Production')
plt.legend()
plt.grid()
plt.savefig('output_plot.png')


