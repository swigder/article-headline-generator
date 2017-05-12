from gensim.models import KeyedVectors

import re

import unicodedata

import torch
from torch.autograd import Variable

from data.read_data import Sample

SOS_token = 0
EOS_token = 1
OOV_token = 2


class GloveLang:
    def __init__(self):
        self.glove = KeyedVectors.load_word2vec_format('~/Documents/school/kth/dl/data/glove.6B.50d.word2vec.txt', binary=False)

    def indexes_from_text(self, text):
        return [self.glove.vocab[word].index if self.glove.__contains__(word) else 0 for word in text.split(' ')]

    def variable_from_text(self, text):
        indexes = self.indexes_from_text(text)
        indexes.append(EOS_token)
        return Variable(torch.LongTensor(indexes).view(-1, 1))

    def variables_from_sample(self, sample):
        return Sample(headline=self.variable_from_text(sample.headline), body=self.variable_from_text(sample.body))

    def index2word(self, index):
        return self.glove.index2word[index]

    def text_from_encoding(self, encoding):
        return self.glove.most_similar(positive=[encoding], topn=1)[0][0]


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize(string, max_length=None):
    string = unicode_to_ascii(string.lower().strip())
    string = re.sub(r"([.!?])", r" \1", string)
    string = re.sub(r"[^a-zA-Z\d.!?]+", r" ", string)
    # string = re.sub(r"\d\d\d+", r"#####", string)
    if max_length:
        string = ' '.join(string.split(' ')[:max_length])
    return string
