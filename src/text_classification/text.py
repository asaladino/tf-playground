from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

from text_classification.helper import format_data, prediction_for_one_entry

imdb = keras.datasets.imdb

# load the data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# Format the data
reverse_word_index, train_data, test_data = format_data(word_index, train_data, test_data)

# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

# Build the model
model = keras.Sequential([
    keras.layers.Embedding(vocab_size, 16),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

x_val = train_data[:vocab_size]
part_x_train = train_data[vocab_size:]

y_val = train_labels[:vocab_size]
part_y_train = train_labels[vocab_size:]

# Train the model
history = model.fit(part_x_train, part_y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)
# Evaluate the model
test_loss, test_acc = model.evaluate(test_data, test_labels)
print('Test accuracy:', test_acc)

prediction_for_one_entry(test_data[0], model, reverse_word_index)
prediction_for_one_entry(test_data[1], model, reverse_word_index)
prediction_for_one_entry(test_data[2], model, reverse_word_index)
prediction_for_one_entry(test_data[3], model, reverse_word_index)
prediction_for_one_entry(test_data[4], model, reverse_word_index)
