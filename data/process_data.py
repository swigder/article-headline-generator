import re

import unicodedata
from nltk import pos_tag

from data.read_data import Sample


def unicode_to_ascii(string):
    if len(string) == len(string.encode()):  # is ASCII
        return string
    else:
        # Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
        # slow so avoid if not necessary
        return ''.join(
            c for c in unicodedata.normalize('NFD', string)
            if unicodedata.category(c) != 'Mn'
        )


def cut_length(string, words):
    # ugly but fast
    if len(string) < words:
        return string
    spaces = 0
    for i, c in enumerate(string):
        if c.isspace():
            spaces += 1
            if spaces == words:
                return string[:i]
    return string


def normalize(string, max_length=None):
    string = unicode_to_ascii(string.lower().strip())
    string = re.sub(r"([.!?])", r" \1", string)
    string = re.sub(r"[^a-zA-Z\d.!?]+", r" ", string)
    string = re.sub(r"\d\d\d\d+", r"#", string)
    if max_length:
        string = cut_length(string, max_length)
    return string


def normalize_samples(samples, max_length=None):
    normalized_samples = []
    for sample in samples:
        normalized_samples.append(Sample(headline=normalize(sample.headline), body=normalize(sample.body, max_length)))
    return normalized_samples


def pos_tag_samples(samples, tag_tgt=False):
    pos = lambda s: " ".join(['{}{}{}'.format(w, "\N{Halfwidth Forms Light Vertical}", t) for w, t in pos_tag(s.split(' '))])

    tagged_samples = []
    for sample in samples:
        tagged_samples.append(Sample(headline=pos(sample.headline) if tag_tgt else sample.headline, body=pos(sample.body)))
    return tagged_samples
