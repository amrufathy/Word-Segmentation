from bert import tokenization


class Tokenizer(tokenization.FullTokenizer):
    def tokenize(self, text):
        text = ''.join(text.split())

        output = []

        for c in text:
            if c in self.vocab:
                output.append(c)
            else:
                output.append('[UNK]')

        return output


if __name__ == '__main__':
    t = Tokenizer('chinese_L-12_H-768_A-12/vocab.txt')
    _text = '迈向  充满  希望  的  新  世纪  ——  一九九八年  新年  讲话  （  附  图片  １  张  ）'

    print(len(''.join(_text.split())))
    # print(t.tokenize(_text))
    print(len(t.tokenize(_text)))
