import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Charger les données MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normaliser les données
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Redimensionner les données
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Aplatir l'image 28x28
    layers.Dense(128, activation='relu'),   # Couche dense avec 128 neurones
    layers.Dense(10, activation='softmax')   # Couche de sortie avec 10 neurones (classes)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)

predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)  # Obtenir la classe avec la plus haute probabilité
