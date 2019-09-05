import os.path
from argparse import ArgumentParser

from tensorflow.python.keras.models import load_model
from tqdm import tqdm

from bert_language_model.data_helpers import *
from bert_language_model.extract_features import get_features


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_path", help="The path of the input file")
    parser.add_argument("output_path", help="The path of the output file")
    parser.add_argument("resources_path", help="The path of the resources needed to load your model")

    return parser.parse_args()


def predict(input_path, output_path, resources_path):
    """
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the BIES format.

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.

    :param input_path: the path of the input file to predict.
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """

    # load test sentences
    tst_sents = read_sentences([input_path])
    tst_features = cut_and_extract_long_features(tst_sents, resources_path)

    assert len(tst_sents) == len(tst_features), 'Sentences and Features must have equal lengths'

    # load model
    model_path = os.path.join(resources_path, 'pku_bert_model.h5')
    model = load_model(model_path)
    # model.summary()

    # predict
    predictions = []
    batch_size = 32
    steps = int(len(tst_features) / batch_size)
    for uni_b in tqdm(test_data_generator(tst_features, batch_size),
                      desc='Predict Loop', total=steps):
        p = model.predict(uni_b)
        # get label for each character
        p = np.argmax(p, axis=2)

        predictions.extend(p.tolist())

    # remove padding
    predictions = remove_padding(predictions, tst_sents)
    # convert to list of strings
    predictions = [''.join(map(str, p)) for p in predictions]
    # convert to BIES format
    predictions = [p.replace('1', 'B').replace('2', 'I').
                       replace('3', 'E').replace('4', 'S').replace('0', 'S')
                   for p in predictions]
    # write predictions to output file
    with open(output_path, 'w') as f:
        f.writelines('\n'.join(predictions))


def remove_padding(predictions, sentences):
    _p = []
    for p, s in zip(predictions, sentences):
        _p.append(p[:len(s)])

    return _p


def cut_and_extract_long_features(long_sents, resources_path):
    """
    Cuts a list of sentences to max length (512), transforms to features
        and then restore to original length.
    """
    new_long_sents = []
    cut_flags = []
    new_long_features = []

    # cut long sentences
    for i in tqdm(range(len(long_sents)), desc='Cutting loop'):
        sent = long_sents[i]
        if len(sent) <= 512:
            new_long_sents.append(sent)
            cut_flags.append(False)
        else:
            num_parts = int((len(sent) / 512) + 1)
            for i in range(num_parts):
                start, end = i * 512, (i + 1) * 512
                new_long_sents.append(sent[start:end])
                cut_flags.append(True)

    long_examples = convert_sentences_to_examples(new_long_sents)
    long_features = get_features(long_examples, resources_path)

    assert len(cut_flags) == len(long_features)

    # restore lengths
    temp_feature = []
    for i in tqdm(range(len(cut_flags)), desc='Pasting loop'):
        if not cut_flags[i]:
            if len(temp_feature) != 0:
                new_long_features.append(np.asarray(temp_feature))
                temp_feature = []

            new_long_features.append(long_features[i])
        else:
            temp_feature.extend(long_features[i])

    assert len(long_sents) == len(new_long_features)
    return new_long_features


if __name__ == '__main__':
    args = parse_args()
    predict(args.input_path, args.output_path, args.resources_path)
