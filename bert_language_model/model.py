import tensorflow as tf


def create_model(embedding_size=256, hidden_size=256, stacked=False):
    # input layer
    unigrams = tf.keras.layers.Input(shape=(None, embedding_size))

    masked_input = tf.keras.layers.Masking()(unigrams)

    # Bi-LSTM
    if stacked:
        # backward
        bilstm = lstm_layer(hidden_size, embedding_size, backwards=True)(masked_input)
        # forward
        bilstm = lstm_layer(hidden_size, embedding_size)(bilstm)
    else:
        bilstm = tf.keras.layers.Bidirectional(
            lstm_layer(hidden_size, embedding_size),
            merge_mode='sum'  # ?????
        )(masked_input)

    output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(5, activation='softmax'))(bilstm)

    _model = tf.keras.Model(inputs=unigrams, outputs=output)

    _model.compile(
        loss='categorical_crossentropy', metrics=['categorical_accuracy'],
        optimizer='adam'
    )

    return _model


def lstm_layer(hidden_size, embedding_size, backwards=False):
    return tf.keras.layers.LSTM(
        hidden_size, return_sequences=True, go_backwards=backwards,
        dropout=0.2, recurrent_dropout=0.2, input_shape=(None, None, embedding_size))
