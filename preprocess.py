"""
Preprocessing functions

"""


def convert_to_bies_format(_input_path, _output_path):
    """
    Writes a file containing the BIES format of an input file
    """
    with open(_input_path, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):  # load one line to memory
            line = line.encode('utf-8').decode('utf-8-sig')
            words = line.strip().split()

            bies_format = ''
            for word in words:
                for i in range(len(word)):
                    # if word is a single character
                    if i == 0 and i == len(word) - 1:
                        bies_format += 'S'
                    elif i == 0:
                        bies_format += 'B'
                    elif i == len(word) - 1:
                        bies_format += 'E'
                    else:
                        bies_format += 'I'

            with open(_output_path, 'a') as h:
                h.write(bies_format + '\n')


def remove_space_from_file(_input_path, _output_path):
    """
    Removes the input file from all spaces and writes the result
        to an output file
    """
    with open(_input_path, 'r') as f:
        for line in f:
            line_no_spaces = ''.join(line.split())

            with open(_output_path, 'a') as h:
                h.write(line_no_spaces + '\n')


if __name__ == '__main__':
    file_path = 'datasets/training/pku_training.utf8'
    output_path = 'datasets/training/pku_training.bies'

    convert_to_bies_format(file_path, output_path)
    # remove_space_from_file(file_path, output_path)
    pass
