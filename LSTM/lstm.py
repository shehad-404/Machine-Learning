import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras_tuner import Hyperband

def generate_time_series_data(seq_length, num_samples):
    X = np.random.randn(num_samples, seq_length)
    y = np.random.randn(num_samples)
    return X, y

sequence_length = 50
num_samples = 1000
X, y = generate_time_series_data(seq_length=sequence_length, num_samples=num_samples)

X = X.reshape((X.shape[0], X.shape[1], 1))

scaler = MinMaxScaler()
X = scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
y = scaler.fit_transform(y.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(hp):
    model = Sequential()
    
    num_layers = hp.Int('num_layers', min_value=1, max_value=4)
    neuron_options = [32, 64, 128, 256]
    activation_functions = ['relu', 'sigmoid', 'softmax']
    neuron_counts = []
    activation_choices = []

    for i in range(num_layers):
        neurons = hp.Choice(f'neurons_{i+1}', values=neuron_options)
        activation = hp.Choice(f'activation_{i+1}', values=activation_functions)
        neuron_counts.append(neurons)
        activation_choices.append(activation)
        
        model.add(
            LSTM(
                units=neurons,
                activation=activation,
                return_sequences=(i < num_layers - 1),
                input_shape=(X_train.shape[1], X_train.shape[2]) if i == 0 else None
            )
        )
    
    model.add(Dense(1))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        ),
        loss='mse'
    )
    
    print(f"Building model with {num_layers} layers:")
    for idx, (neurons, activation) in enumerate(zip(neuron_counts, activation_choices), 1):
        print(f"Layer {idx}: Neurons {neurons}, Activation Function: {activation}")
    
    return model

hypermodel = Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=20,
    factor=3,
    directory='my_dir',
    project_name='lstm_tuning_layers_neurons_activations'
)

hypermodel.search(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=32, verbose=1)

best_hps = hypermodel.get_best_hyperparameters(num_trials=1)[0]
num_layers = best_hps.get('num_layers')
print(f"\nBest Model Configuration:")
print(f"  Number of Layers: {num_layers}")
for i in range(1, num_layers + 1):
    print(f"  Layer {i}: Neurons {best_hps.get(f'neurons_{i}')}, Activation Function: {best_hps.get(f'activation_{i}')}")
print(f"  Best Learning Rate: {best_hps.get('learning_rate')}")

best_model = hypermodel.hypermodel.build(best_hps)

history = best_model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=32, verbose=1)

loss = best_model.evaluate(X_test, y_test, verbose=0)
print(f'\nTest Loss: {loss}')

print(f"\nTest Loss: {loss}")
print(f"Model Summary:")
print(f"  Number of Layers: {num_layers}")
for i in range(1, num_layers + 1):
    print(f"  Layer {i}: Neurons {best_hps.get(f'neurons_{i}')}, Activation Function: {best_hps.get(f'activation_{i}')}")
    
y_pred = best_model.predict(X_test)

y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(y_test_rescaled, label='Actual')
plt.plot(y_pred_rescaled, label='Predicted')
plt.legend()
plt.title('LSTM Predictions vs Actual Values')
plt.show()


