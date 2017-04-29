from data.language import Lang


def filter_data(data_set):
    return data_set


def prepare_data(data_set):
    lang = Lang()
    print("Read %s sentence pairs" % len(data_set))
    data_set = filter_data(data_set)
    print("Trimmed to %s sentence pairs" % len(data_set))
    print("Counting words...")
    for sample in data_set:
        lang.add_sample(sample)
    print("Counted words: %s" % lang.n_words)
    return lang, data_set
