import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to [0,1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Class names for CIFAR-10 dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Plot Accuracy and Loss Curves
plt.figure(figsize=(15, 5))

# Accuracy Plot
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

# Loss Plot
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

# Training vs Validation Loss Difference
plt.subplot(1, 3, 3)
plt.plot(np.abs(np.array(history.history['loss']) - np.array(history.history['val_loss'])), label='Loss Difference')
plt.xlabel('Epochs')
plt.ylabel('Loss Difference')
plt.legend()
plt.title('Train vs Validation Loss Difference')

plt.show()

# Confusion Matrix
y_pred = np.argmax(model.predict(x_test), axis=1)
y_test_flat = y_test.flatten()

conf_matrix = confusion_matrix(y_test_flat, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Class-wise Accuracy Bar Chart
class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)  # Compute per-class accuracy
plt.figure(figsize=(12, 6))
sns.barplot(x=class_names, y=class_accuracies, palette='coolwarm')
plt.xlabel("Class")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.title("Class-wise Accuracy")
plt.xticks(rotation=45)
plt.show()

# Prediction Confidence Distribution
confidence_scores = np.max(model.predict(x_test), axis=1)
plt.figure(figsize=(12, 6))
sns.histplot(confidence_scores, bins=20, kde=True, color='blue')
plt.xlabel("Prediction Confidence")
plt.ylabel("Frequency")
plt.title("Prediction Confidence Distribution")
plt.show()

# Display Misclassified Images
misclassified_indices = np.where(y_pred != y_test_flat)[0][:10]  # Get first 10 misclassified samples

plt.figure(figsize=(15, 6))
for i, idx in enumerate(misclassified_indices):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[idx], interpolation='none')  # Ensures no blurring
    pred_label = class_names[y_pred[idx]]
    true_label = class_names[y_test_flat[idx]]
    plt.title(f"Pred: {pred_label}\nTrue: {true_label}", fontsize=12, fontweight='bold', color='red')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

plt.tight_layout()
plt.show()

# Predict and visualize some test images
predictions = model.predict(x_test[:10])

plt.figure(figsize=(15, 6))  # Increase figure size for better clarity

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i], interpolation='none')  # Ensures no blurring
    pred_label = np.argmax(predictions[i])
    confidence = np.max(predictions[i]) * 100  # Get confidence percentage
    plt.title(f"{class_names[pred_label]}\n({confidence:.2f}%)", fontsize=12, fontweight='bold', color='blue')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

plt.tight_layout()
plt.show()
