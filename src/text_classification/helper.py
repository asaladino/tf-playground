import numpy as np
from keras_preprocessing import sequence
import matplotlib.pyplot as plt

from basic_classification.helper import plot_value_array


def decode_review(reverse_word_index, text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


def format_data(word_index, train_data, test_data):
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    train_data = sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding='post', maxlen=256)
    test_data = sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding='post', maxlen=256)
    return reverse_word_index, train_data, test_data


def graph_training_data(history):
    history_dict = history.history
    history_dict.keys()

    acc = history_dict['acc']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def prediction_for_one_entry(test_data, model, reverse_word_index):
    print(test_data.shape)
    text_review = decode_review(reverse_word_index, test_data)
    print(text_review)
    test_data = (np.expand_dims(test_data, 0))
    print(test_data.shape)
    predictions = model.predict(test_data)
    print(predictions)
