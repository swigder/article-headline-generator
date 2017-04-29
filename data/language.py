import re
from collections import namedtuple

import unicodedata


# much of the code in this file was taken / adapted from https://github.com/spro/practical-pytorch
import torch
from torch.autograd import Variable

Sample = namedtuple('Sample', ['headline', 'body'])

SOS_token = 0
EOS_token = 1
OOV_token = 2


class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "OOV"}
        self.n_words = 3  # Count SOS and EOS and OOV

    def add_sample(self, sample):
        for word in sample.headline.split(' '):
            self.add_word(word)
        for word in sample.body.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def indexes_from_text(self, text):
        return [self.word2index[word] if self.word2count[word] > 0 else OOV_token for word in text.split(' ')]

    def variable_from_text(self, text):
        indexes = self.indexes_from_text(text)
        indexes.append(EOS_token)
        return Variable(torch.LongTensor(indexes).view(-1, 1))

    def variables_from_sample(self, sample):
        return Sample(headline=self.variable_from_text(sample.headline), body=self.variable_from_text(sample.body))


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
    string = re.sub(r"\d\d\d+", r"#####", string)
    if max_length:
        string = ' '.join(string.split(' ')[:max_length])
    return string
