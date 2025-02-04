import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten  # Include other layers as needed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#import tensorflow as tf
#from tf.keras.models import Sequential
#from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout


# 1. Load the CSV file
data = pd.read_csv('/home/shehad/Desktop/MLP/CNN/sample_time_series.csv')

# 2. Drop the 'timestamp' column since it's non-numeric
data.drop(columns=['timestamp'], inplace=True)

# 3. Define the target variable
target_column = 'value'

# Extract features (X_raw) and target (y_raw)
X_raw = data.drop(columns=[target_column])  # No additional features here, so we drop the target column
y_raw = data[target_column]

# If X_raw is empty (univariate time series), use y_raw itself as the feature
X_raw = y_raw.values.reshape(-1, 1)  # Reshape to (n_samples, 1)

# 4. Normalize the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X_raw)  # Scale features
y_scaled = scaler_y.fit_transform(y_raw.values.reshape(-1, 1))  # Scale target

# 5. Create sliding windows
def create_sliding_windows(X, y, window_size):
    X_windows, y_windows = [], []
    for i in range(len(X) - window_size):
        X_windows.append(X[i:i + window_size])
        y_windows.append(y[i + window_size])
    return np.array(X_windows), np.array(y_windows)

# Set the window size (number of past time steps to consider)
window_size = 30

# Create sliding windows
X, y = create_sliding_windows(X_scaled, y_scaled, window_size)

# 6. Split the data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 7. Build the CNN model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(100, activation='relu'),
    Dropout(0.3),
    Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 8. Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    verbose=2
)

# 9. Evaluate the model
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# 10. Make predictions
y_pred = model.predict(X_test)
y_pred_rescaled = scaler_y.inverse_transform(y_pred)  # Rescale predictions back to original scale
y_test_rescaled = scaler_y.inverse_transform(y_test)  # Rescale true values

# 11. Visualize predictions vs true values
plt.figure(figsize=(10, 6))
plt.plot(y_test_rescaled, label='True Values', marker='o')
plt.plot(y_pred_rescaled, label='Predictions', marker='x')
plt.legend()
plt.title('Predicted vs True Values')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.show()

# 12. Save the model
model.save('time_series_cnn_model.h5')
print("Model saved as 'time_series_cnn_model.h5'")

