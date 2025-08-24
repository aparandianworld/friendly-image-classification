#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import os

def main():
    # Load and normalize dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    
    # Build CNN model
    model = models.Sequential()
    
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    
    # Compile model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    # Train model
    history = model.fit(train_images, train_labels, epochs=15,
                        validation_data=(test_images, test_labels))
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"\nTest loss is: {test_loss}")
    print(f"Test accuracy is: {test_acc}")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save model
    model.save("models/cifar10_model.h5")

if __name__ == "__main__":
    main()