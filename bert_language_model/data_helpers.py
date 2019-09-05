import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils import to_categorical

from bert_helpers import InputExample


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


def convert_sentences_to_examples(sentences, idx_offset=0):
    examples = []
    for idx, sent in enumerate(sentences, start=idx_offset):
        examples.append(
            InputExample(unique_id=idx, text_a=sent.strip())
        )

    return examples


def read_examples(snt_files):
    """
    Read a list of `InputExample`s from an input files.
    """
    sentences = read_sentences(snt_files)
    return convert_sentences_to_examples(sentences)


def train_data_generator(data, batch_size, shuffle=False):
    """
    Generates a batch of equally padded unigrams, bigrams and labels.

    Padding is computed per batch to support variable sentences length
        during train.
    """
    features, labels = data

    if not shuffle:
        perm = np.array(range(len(features)))
    else:
        perm = np.random.permutation(len(features))

    for start in range(0, len(features), batch_size):
        end = start + batch_size

        features_batch = features[perm[start:end]]
        labels_batch = labels[perm[start:end]]

        max_length = len(max(features_batch, key=len))

        o_features = pad_sequences(features_batch, padding='post', maxlen=max_length, value=np.zeros(256))
        o_labels = pad_sequences(labels_batch, padding='post', maxlen=max_length)

        yield o_features, to_categorical(o_labels)


def test_data_generator(data, batch_size):
    """
    Generates a batch of equally padded unigrams, bigrams and labels.

    Padding is computed per batch to support variable sentences length
        during train.
    """
    features = data

    for start in range(0, len(features), batch_size):
        end = start + batch_size

        features_batch = features[start:end]

        max_length = len(max(features_batch, key=len))

        o_features = pad_sequences(features_batch, padding='post', maxlen=max_length, value=np.zeros(768))

        yield o_features
