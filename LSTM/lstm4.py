import random
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Simulate dataset (replace with actual data)
def create_dummy_data(num_samples=1000, time_steps=20, input_dim=10, output_dim=1):
    X = np.random.rand(num_samples, time_steps, input_dim)
    y = np.random.rand(num_samples, output_dim)
    return X, y

# Define the LSTM model with hyperparameters
def build_lstm_model(hp):
    model = Sequential()
    
    # Add LSTM layers
    model.add(LSTM(
        units=hp['lstm_units'], 
        activation='relu',
        return_sequences=hp['num_layers'] > 1,  # Return sequences if multiple layers
        input_shape=(hp['sequence_length'], hp['input_dim'])
    ))
    
    for i in range(hp['num_layers'] - 1):
        model.add(LSTM(units=hp['lstm_units'], activation='relu', return_sequences=i < hp['num_layers'] - 2))
    
    # Dropout
    model.add(Dropout(hp['dropout_rate']))
    
    # Dense output layer
    model.add(Dense(hp['output_dim'], activation='linear'))
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=hp['learning_rate']),
        loss='mse',  # Replace with appropriate loss function
        metrics=['mae']  # Replace with appropriate metrics
    )
    
    return model

# Generate dummy dataset
X, y = create_dummy_data()
x_train, x_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# Define the search space
search_space = {
    'lstm_units': [32, 64, 128, 256],
    'num_layers': [1, 2, 3],
    'dropout_rate': [0.1, 0.2, 0.3, 0.5],
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64, 128],
    'sequence_length': [20],  # Adjust based on your data
    'input_dim': [10],        # Adjust based on your data
    'output_dim': [1]         # Adjust based on your data
}

# Random search
num_trials = 10  # Set the number of random trials
best_model = None
best_score = np.inf

for trial in range(num_trials):
    # Randomly sample hyperparameters
    hp = {key: random.choice(values) for key, values in search_space.items()}
    print(f"Trial {trial + 1}/{num_trials} with hyperparameters: {hp}")
    
    # Build and train the model
    model = build_lstm_model(hp)
    history = model.fit(
        x_train, y_train,
        epochs=10,  # Use a small number of epochs for tuning
        batch_size=hp['batch_size'],
        validation_data=(x_val, y_val),
        verbose=0
    )
    
    # Get the validation loss
    val_loss = min(history.history['val_loss'])
    print(f"Validation loss: {val_loss}")
    
    # Save the best model
    if val_loss < best_score:
        best_score = val_loss
        best_model = model
        best_hyperparameters = hp

print(f"\nBest hyperparameters: {best_hyperparameters}")
print(f"Best validation loss: {best_score}")

# Save the optimized model
best_model.save("optimized_lstm_model.h5")
print("Optimized model saved.")

# Evaluate the model on the test data
final_eval = best_model.evaluate(x_test, y_test)
print(f"Final evaluation on test data: {final_eval}")

