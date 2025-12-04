# -------------------------------------------------------------
# Handwritten Digit Recognition using MNIST (Single File Code)
# -------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------
# 1. Load MNIST Dataset
# -----------------------------------------
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# Normalize (0–1 scale)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for CNN (28x28x1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# -----------------------------------------
# 2. Build CNN Model
# -----------------------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Show model summary
print("\nMODEL SUMMARY:")
model.summary()

# -----------------------------------------
# 3. Train the Model
# -----------------------------------------
print("\nTraining the model...\n")
history = model.fit(
    x_train, y_train,
    epochs=5,
    validation_split=0.1
)

# -----------------------------------------
# 4. Evaluate the Model
# -----------------------------------------
print("\nEvaluating the model on test data...")
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)

# -----------------------------------------
# 5. Predict a sample digit
# -----------------------------------------
index = 0   # choose any image from test set
sample = x_test[index].reshape(28, 28)

plt.imshow(sample, cmap='gray')
plt.title("Sample Image")
plt.show()

prediction = model.predict(np.array([x_test[index]]))
predicted_digit = np.argmax(prediction)

print("\nPredicted Digit:", predicted_digit)
print("Actual Digit:", y_test[index])

# -----------------------------------------
# 6. Save the Trained Model
# -----------------------------------------
model.save("mnist_digit_model.h5")
print("\nModel saved as mnist_digit_model.h5")
