import re

from nltk import pos_tag, word_tokenize

from data.read_data import Sample


def cut_length(string, words):
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
    string = string.lower().strip()
    if max_length:
        string = cut_length(string, max_length)
    string = re.sub(r"([.!?])", r" \1", string)
    string = re.sub(r"[^a-zA-Z\d.!?]+", r" ", string)
    string = re.sub(r"\d\d\d\d+", r"#", string)
    return string


def normalize_samples(samples, max_length=None):
    normalized_samples = []
    for sample in samples:
        normalized_samples.append(Sample(headline=normalize(sample.headline), body=normalize(sample.body, max_length)))
    return normalized_samples


def pos_tag_samples(samples, tag_tgt=False):
    pos = lambda s: " ".join(['{}{}{}'.format(w, "\N{Halfwidth Forms Light Vertical}", t) for w, t in pos_tag(word_tokenize(s))])

    tagged_samples = []
    for sample in samples:
        tagged_samples.append(Sample(headline=pos(sample.headline) if tag_tgt else sample.headline, body=pos(sample.body)))
    return tagged_samples
