from os.path import splitext

import tensorflow as tf
from tqdm import tqdm

from data_helpers import *
from model import create_model
from tokenizer import Tokenizer

tf.logging.set_verbosity(tf.logging.DEBUG)


def main(_):
    trn_snt_files = [
        # '../datasets/training/as_simplified_training.utf8',
        '../datasets/training/cityu_simplified_training.utf8',
        '../datasets/training/msr_training.utf8',
        '../datasets/training/pku_training.utf8'
    ]
    trn_lbl_files = [splitext(f)[0] + '.bies' for f in trn_snt_files]

    tf.logging.info('Loading training data...')
    trn_snts = read_sentences(trn_snt_files)
    y_trn = read_labels(trn_lbl_files)

    assert len(trn_snts) == len(y_trn), 'Sentences and labels must be equal'

    train_tok = False
    tokenizer = Tokenizer(trn_snts, verbose=True)
    if train_tok:
        tokenizer.fit()
        tokenizer.save()
    else:
        tokenizer.load()

    x_uni_trn, x_bi_trn = process_sentences(trn_snts, tokenizer)

    tf.logging.info('Creating model...')
    model = create_model(tokenizer.vocab_size(), stacked=False)
    model.summary()

    tf.logging.info('Training model...')
    epochs = 10
    batch_size = 32
    steps = int(len(x_uni_trn) / batch_size)
    for epoch in range(epochs):
        print('Epoch', epoch + 1)
        for uni_b, bi_b, lbl_b in tqdm(train_data_generator([x_uni_trn, x_bi_trn, y_trn], batch_size, shuffle=True),
                                       desc='Training Loop', total=steps):
            try:
                loss, acc = model.train_on_batch([uni_b, bi_b], lbl_b)
                # print('Loss:', loss, 'Acc:', acc)
            except Exception as e:
                print(e)

    model.save('unstacked_combined_model.h5')


if __name__ == '__main__':
    tf.app.run()
