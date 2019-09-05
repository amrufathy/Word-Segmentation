import numpy as np
import pandas as pd
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils import to_categorical


def read_sentences(snt_files):
    """
    reads lines from multiple files into a single
        list. Removes spaces from lines.
    """
    sentences = []
    for f in snt_files:
        with open(f, 'r') as h:
            sentences.extend(h.read().encode('utf-8').decode('utf-8-sig').splitlines())

    # process sentences
    sentences = [''.join(s.split()) for s in sentences]

    return sentences


def process_sentences(sentences, tokenizer):
    """
    Transforms `sentences` into sequences of integers of
        character unigrams and bigrams.

    """
    unigrams = tokenizer.vectorize(sentences)
    bigrams = tokenizer.vectorize(sentences, _ngrams=2)

    return unigrams, bigrams


def read_labels(lbl_files):
    """
    reads BIES formatted files and transforms labels
        to integers for training
    """
    # read to data frame
    labels = []
    for f in lbl_files:
        with open(f, 'r') as h:
            labels.extend(h.read().encode('utf-8').decode('utf-8-sig').splitlines())

    lbl_df = pd.DataFrame(labels, columns=['labels'])

    # process labels
    lbl_df = lbl_df.applymap(lambda x: x.replace('B', '1').replace('I', '2')
                             .replace('E', '3').replace('S', '4'))
    labels = lbl_df.labels.apply(lambda x: [int(s) for s in x]).tolist()

    return np.asarray(labels)


def train_data_generator(data, batch_size, shuffle=False):
    """
    Generates a batch of equally padded unigrams, bigrams and labels.

    Padding is computed per batch to support variable sentences length
        during train.
    """
    unigrams, bigrams, labels = data

    if not shuffle:
        perm = np.array(range(len(unigrams)))
    else:
        perm = np.random.permutation(len(unigrams))

    for start in range(0, len(unigrams), batch_size):
        end = start + batch_size

        unigrams_batch = unigrams[perm[start:end]]
        bigrams_batch = bigrams[perm[start:end]]
        labels_batch = labels[perm[start:end]]

        max_length = len(max(unigrams_batch, key=len))

        o_unigrams = pad_sequences(unigrams_batch, padding='post', maxlen=max_length)
        o_bigrams = pad_sequences(bigrams_batch, padding='post', maxlen=max_length)
        o_labels = pad_sequences(labels_batch, padding='post', maxlen=max_length)

        yield o_unigrams, o_bigrams, to_categorical(o_labels)


def test_data_generator(data, batch_size):
    """
    Generates a batch of equally padded unigrams and bigrams.

    Padding is computed per batch to support variable sentences length
        during predict.
    """
    unigrams, bigrams = data

    for start in range(0, len(unigrams), batch_size):
        end = start + batch_size

        unigrams_batch = unigrams[start:end]
        bigrams_batch = bigrams[start:end]

        max_length = len(max(unigrams_batch, key=len))

        o_unigrams = pad_sequences(unigrams_batch, padding='post', maxlen=max_length)
        o_bigrams = pad_sequences(bigrams_batch, padding='post', maxlen=max_length)

        yield o_unigrams, o_bigrams
