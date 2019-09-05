# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract pre-computed feature vectors from BERT."""

import os

import numpy as np
from tqdm import tqdm

from bert_helpers import *
from data_helpers import read_examples
from tokenizer import Tokenizer

# https://github.com/google-research/bert/issues/190
MAX_SEQ_LEN = 128
BATCH_SIZE = 32


def get_features(_examples, resources_path, embedding_size=256):
    BERT_CONFIG = os.path.join(resources_path, 'chinese_L-12_H-768_A-12', 'bert_config.json')
    INIT_CKPT = os.path.join(resources_path, 'chinese_L-12_H-768_A-12', 'bert_model.ckpt')
    VOCAB_FILE = os.path.join(resources_path, 'chinese_L-12_H-768_A-12', 'vocab.txt')

    tf.logging.set_verbosity(tf.logging.INFO)

    layer_indexes = [-1, -2, -3, -4]

    bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG)

    tokenizer = Tokenizer(vocab_file=VOCAB_FILE, do_lower_case=True)

    run_config = tf.estimator.RunConfig()

    _features = convert_examples_to_features(
        examples=_examples, seq_length=MAX_SEQ_LEN, tokenizer=tokenizer)

    unique_id_to_feature = {}
    for feature in _features:
        unique_id_to_feature[feature.unique_id] = feature

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=INIT_CKPT,
        layer_indexes=layer_indexes,
        use_one_hot_embeddings=False)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={'batch_size': BATCH_SIZE})

    input_fn = input_fn_builder(
        features=_features, seq_length=MAX_SEQ_LEN)

    sentences_vectors = []
    for result in tqdm(estimator.predict(input_fn),
                       desc='Features Loop', total=len(_examples)):
        unique_id = int(result["unique_id"])
        feature = unique_id_to_feature[unique_id]
        word_vectors = []
        for (i, token) in enumerate(feature.tokens):
            all_layers_embeddings = []
            for (j, layer_index) in enumerate(layer_indexes):
                layer_output = result["layer_output_%d" % j]
                layer_token_embedding = np.array(layer_output[i:i + 1].flat)[:embedding_size]
                all_layers_embeddings.append(layer_token_embedding)
            token_embedding = np.sum(all_layers_embeddings, axis=0).tolist()
            word_vectors.append(token_embedding)

        sentences_vectors.append(np.asarray(word_vectors))

    return sentences_vectors


if __name__ == "__main__":
    examples = read_examples('../datasets/training/pku_training_one.utf8')
    features = get_features(examples, '')
