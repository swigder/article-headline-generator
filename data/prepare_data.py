from data.embedding_language import GloveLang
from data.language import Lang, Sample, normalize
from rnn.rnn import MAX_LENGTH


def filter_data(data_set):
    return data_set


def prepare_data_index_lang(data_set):
    lang = Lang()
    print("Read %s data samples" % len(data_set))
    data_set = filter_data(data_set)
    print("Trimmed to %s data samples" % len(data_set))
    data_set = [Sample(headline=normalize(sample.headline, MAX_LENGTH), body=normalize(sample.body, MAX_LENGTH)) for sample in data_set]
    print("Counting words...")
    for sample in data_set:
        lang.add_sample(sample)
    print("Counted words: %s" % lang.n_words)
    return lang, data_set


def prepare_data_embedding_lang(data_set):
    lang = GloveLang()
    print("Read %s data samples" % len(data_set))
    data_set = filter_data(data_set)
    print("Trimmed to %s data samples" % len(data_set))
    data_set = [Sample(headline=normalize(sample.headline, MAX_LENGTH), body=normalize(sample.body, MAX_LENGTH)) for sample in data_set]
    print("Counting words...")
    return lang, data_set
