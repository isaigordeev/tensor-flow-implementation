# from tensorflow import keras
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential

import tensorflow.python.keras
# from tensorflow
# import keras

model = Sequential()

model.add(Dense(10, input_dim = 4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print a summary of the model
model.summary()
