import tensorflow as tf
import numpy as np

# Set the random seed for reproducibility
np.random.seed(123)

# Define the input sequence length and vocabulary size
max_len = 20
vocab_size = 50

# Generate some random input data
input_data = np.random.randint(0, vocab_size, size=(1000, max_len))

# Define the RNN model
model = tf.keras.models.Sequential([
    # Embedding layer to convert input sequence to dense vectors
    tf.keras.layers.Embedding(vocab_size, 64, input_length=max_len),
    # Simple RNN layer with 64 hidden units
    tf.keras.layers.SimpleRNN(64),
    # Dense layer to output predictions
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model with binary cross-entropy loss and Adam optimizer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on the input data with binary labels
model.fit(input_data, np.random.randint(0, 2, size=(1000, 1)), epochs=10, batch_size=32)
