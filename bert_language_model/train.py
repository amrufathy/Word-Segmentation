from os.path import splitext

import tensorflow as tf
from tqdm import tqdm

from data_helpers import *
from extract_features import get_features
from model import create_model

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
    trn_examples = read_examples(trn_snt_files)
    trn_features = np.asarray(get_features(trn_examples, ''))
    y_trn = read_labels(trn_lbl_files)

    tf.logging.info('Creating model...')
    model = create_model()
    model.summary()

    tf.logging.info('Training model...')
    epochs = 1
    batch_size = 32
    steps = int(len(trn_features) / batch_size)
    for epoch in range(epochs):
        print('Epoch', epoch + 1)
        for uni_b, lbl_b in tqdm(train_data_generator([trn_features, y_trn], batch_size, shuffle=True),
                                 desc='Training Loop', total=steps):
            try:
                loss, acc = model.train_on_batch(uni_b, lbl_b)
                # print('Loss:', loss, 'Acc:', acc)
            except Exception as e:
                print(e)

    model.save('combined_bert_model.h5')


if __name__ == '__main__':
    tf.app.run()
