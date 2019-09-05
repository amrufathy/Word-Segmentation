import os.path
from argparse import ArgumentParser

from tensorflow.python.keras.models import load_model
from tqdm import tqdm

from baseline_model.data_helpers import *
from baseline_model.tokenizer import Tokenizer


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

    # load tokenizer
    tokenizer_path = os.path.join(resources_path, 'dict.pic')
    tokenizer = Tokenizer(verbose=True)
    tokenizer.load(tokenizer_path)

    # load model
    model_path = os.path.join(resources_path, 'model.h5')
    model = load_model(model_path)
    # model.summary()

    # load test sentences
    tst_snts = read_sentences([input_path])
    x_uni_tst, x_bi_tst = process_sentences(tst_snts, tokenizer)

    # predict
    predictions = []
    batch_size = 32
    steps = int(len(x_uni_tst) / batch_size)
    for uni_b, bi_b in tqdm(test_data_generator([x_uni_tst, x_bi_tst], batch_size),
                            desc='Predict Loop', total=steps):
        p = model.predict([uni_b, bi_b])
        # get label for each character
        p = np.argmax(p, axis=2)

        predictions.extend(p.tolist())

    # remove padding
    predictions = remove_padding(predictions, tst_snts)
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


if __name__ == '__main__':
    args = parse_args()
    predict(args.input_path, args.output_path, args.resources_path)
