import tensorflow as tf
from tensorflow.python.keras.utils import plot_model


def create_model(_vocab_size, embedding_size=128, hidden_size=256, stacked=False):
    # input layer
    unigrams = tf.keras.layers.Input(shape=(None,), name='unigrams_input')
    bigrams = tf.keras.layers.Input(shape=(None,), name='bigrams_input')

    # embeddings layer
    unigram_embeddings = tf.keras.layers.Embedding(
        _vocab_size, embedding_size, mask_zero=True, name='unigram_embeddings')(unigrams)

    bigram_embeddings = tf.keras.layers.Embedding(
        _vocab_size, embedding_size, mask_zero=True, name='bigram_embeddings')(bigrams)

    embedding_output = tf.keras.layers.add([
        tf.keras.layers.Dense(hidden_size, activation='linear')(unigram_embeddings),
        tf.keras.layers.Dense(hidden_size, activation='linear')(bigram_embeddings)
    ])

    masked_embeddings = tf.keras.layers.Masking(name='masked_embeddings')(embedding_output)

    # Bi-LSTM
    if stacked:
        # backward
        bilstm = lstm_layer(hidden_size, embedding_size, backwards=True)(masked_embeddings)
        # forward
        bilstm = lstm_layer(hidden_size, embedding_size)(bilstm)
    else:
        bilstm = tf.keras.layers.Bidirectional(
            lstm_layer(hidden_size, embedding_size),
            merge_mode='sum',  # ?????
            name='Bi-LSTM'
        )(masked_embeddings)

    output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(5, activation='softmax'), name='softmax_output')(bilstm)

    _model = tf.keras.Model(inputs=[unigrams, bigrams], outputs=output)

    _model.compile(
        loss='categorical_crossentropy', metrics=['categorical_accuracy'],
        optimizer='adam'
    )

    return _model


def lstm_layer(hidden_size, embedding_size, backwards=False):
    return tf.keras.layers.LSTM(
        hidden_size, return_sequences=True, go_backwards=backwards,
        dropout=0.2, recurrent_dropout=0.2, input_shape=(None, None, embedding_size))


if __name__ == '__main__':
    m = create_model(770324)
    m.summary()
    plot_model(m, to_file='model.png')
