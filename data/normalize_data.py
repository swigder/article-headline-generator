import unicodedata

import re

from data.language import Sample


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize(string, max_length=None):
    string = unicode_to_ascii(string.lower().strip())
    string = re.sub(r"([.!?])", r" \1", string)
    string = re.sub(r"[^a-zA-Z\d.!?]+", r" ", string)
    string = re.sub(r"\d\d\d\d+", r"#####", string)
    if max_length:
        string = ' '.join(string.split(' ')[:max_length])
    return string


def normalize_samples(samples, max_length=None):
    normalized_samples = []
    for sample in samples:
        normalized_samples.append(Sample(headline=normalize(sample.headline), body=normalize(sample.body, max_length)))
    return normalized_samples
