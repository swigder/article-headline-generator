import unicodedata

import re

from nltk import pos_tag, word_tokenize

from data.language import Sample


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize(string, max_length=None):
    string = unicode_to_ascii(string.lower().strip())
    if max_length:
        string = ' '.join(word_tokenize(string)[:max_length])
    string = re.sub(r"([.!?])", r" \1", string)
    string = re.sub(r"[^a-zA-Z\d.!?]+", r" ", string)
    string = re.sub(r"\d\d\d\d+", r"#####", string)
    return string


def normalize_samples(samples, max_length=None):
    normalized_samples = []
    for sample in samples:
        normalized_samples.append(Sample(headline=normalize(sample.headline), body=normalize(sample.body, max_length)))
    return normalized_samples


def pos_tag_samples(samples):
    pos = lambda s: " ".join(['{}{}{}'.format(w, "\N{Halfwidth Forms Light Vertical}", t) for w, t in pos_tag(word_tokenize(s))])

    tagged_samples = []
    for sample in samples:
        tagged_samples.append(Sample(headline=pos(sample.headline), body=pos(sample.body)))
    return tagged_samples
