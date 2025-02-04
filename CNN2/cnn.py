import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.datasets import cifar10 as cf10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision, Recall

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cf10.load_data()

# Display some images
def show_images(train_images, class_names, train_labels, nb_samples=12, nb_row=4):
    plt.figure(figsize=(12, 12))
    for i in range(nb_samples):
        plt.subplot(nb_row, nb_row, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

show_images(train_images, class_names, train_labels)

# Data Preprocessing
max_pixel_value = 255
train_images = train_images / max_pixel_value
test_images = test_images / max_pixel_value

train_labels = to_categorical(train_labels, len(class_names))
test_labels = to_categorical(test_labels, len(class_names))

# Model Architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

# Compile the model
BATCH_SIZE = 32
EPOCHS = 30
METRICS = ['accuracy', Precision(name='precision'), Recall(name='recall')]

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=METRICS)

# Train the model
training_history = model.fit(
    train_images, train_labels,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_data=(test_images, test_labels)
)

# Plot Performance Curves
def show_performance_curve(training_result, metric, metric_label):
    train_perf = training_result.history[str(metric)]
    validation_perf = training_result.history['val_' + str(metric)]
    
    intersection_idx = np.argwhere(np.isclose(train_perf, validation_perf, atol=1e-2)).flatten()[0]
    intersection_value = train_perf[intersection_idx]

    plt.figure()
    plt.plot(train_perf, label=metric_label)
    plt.plot(validation_perf, label='val_' + str(metric))
    plt.axvline(x=intersection_idx, color='r', linestyle='--', label='Intersection')
    plt.annotate(f'Optimal Value: {intersection_value:.4f}',
                 xy=(intersection_idx, intersection_value),
                 xycoords='data', fontsize=10, color='green')
    plt.xlabel('Epoch')
    plt.ylabel(metric_label)
    plt.legend(loc='lower right')
    plt.show()

show_performance_curve(training_history, 'accuracy', 'Accuracy')
show_performance_curve(training_history, 'precision', 'Precision')

# Evaluate the Model
test_predictions = model.predict(test_images)
test_predicted_labels = np.argmax(test_predictions, axis=1)
test_true_labels = np.argmax(test_labels, axis=1)

cm = confusion_matrix(test_true_labels, test_predicted_labels)
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

cmd.plot(include_values=True, cmap='viridis', xticks_rotation='horizontal')
plt.show()
