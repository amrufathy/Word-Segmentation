import pickle

import numpy as np


class Tokenizer:
    def __init__(self, sentences=None, verbose=False):
        self.__verbose = verbose
        self.__sentences = sentences
        self.__dictionary = dict()
        self.__vocab_size = len(self.__dictionary)

    def fit(self):
        """
        Extracts all unique characters (unigrams) from a corpus and
            creates a char-int mapping
        """
        sentences = ''.join(self.__sentences)  # concatenate all sentences
        chars = sorted(list(set(sentences)))  # extract unique characters (unigrams)
        bigrams = sorted(list(set(self.ngrams(sentences, 2))))
        all_grams = chars + bigrams + ['unk']  # add unknown character

        self.__dictionary = dict((c, i) for i, c in enumerate(all_grams, start=1))
        self.__vocab_size = len(self.__dictionary)

        if self.__verbose:
            print('Vocab size:', self.__vocab_size)

    def vectorize(self, sentences, _ngrams=1):
        """
        Transforms each sentence to a sequence of integers, given the ngrams of
            each sentence and a dictionary for the conversion.
        """

        if self.__verbose:
            print('Vectorizing', len(sentences), 'sentences')

        vectors = []

        for sent in sentences:
            v = []
            for gram in self.ngrams(sent, _ngrams):
                if gram in self.__dictionary:
                    v.append(self.__dictionary[gram])
                else:
                    v.append(self.__dictionary['unk'])
            vectors.append(v)

        return np.asarray(vectors)

    @staticmethod
    def ngrams(text, n):
        """
        Generates ngrams from `text`.
        """
        grams = zip(*[text[i:] for i in range(n)])
        return [''.join(gram) for gram in grams]

    def vocab_size(self):
        """
        Returns vocab size
        """
        return self.__vocab_size

    def set_verbosity(self, _v):
        """
        Sets verbosity flag
        """
        self.__verbose = _v

    def save(self, path='dict.pic'):
        """
        Saves dictionary to `path`.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.__dictionary, f)

    def load(self, path='dict.pic'):
        """
        Loads dictionary from `path`.
        """
        with open(path, 'rb') as f:
            self.__dictionary = pickle.load(f)
            self.__vocab_size = len(self.__dictionary)

            if self.__verbose:
                print('Loading Tokenizer, vocab size:', self.vocab_size())
