import tensorflow as tf
import numpy as np
import tensorflow

foo = tensorflow.keras
from tensorflow import keras

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
 
# Set the random seed for reproducibility
np.random.seed(123)

# Set the vocabulary size and maximum sequence length
vocab_size = 10000
max_len = 200

# Load the IMDB dataset and split into training and testing sets
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Pad the sequences to a fixed length
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# Define the RNN model
model = tf.keras.models.Sequential([
    # Embedding layer to convert input sequence to dense vectors
    tf.keras.layers.Embedding(vocab_size, 64, input_length=max_len),
    # Simple RNN layer with 64 hidden units
    tf.keras.layers.SimpleRNN(64),
    # Dense layer with sigmoid activation function to output binary predictions
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model with binary cross-entropy loss and Adam optimizer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on the IMDB training data and evaluate on the test data
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)
