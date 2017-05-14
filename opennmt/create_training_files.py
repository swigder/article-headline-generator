import argparse

from os import listdir, path
from random import shuffle, seed

from data.process_data import normalize_samples, pos_tag_samples
from data.read_data import read_event_registry_data, read_crowdflower_economic_data, read_crowdflower_wikipedia_data, \
    read_reuters_data, read_cnn_dailymail_data


def write_samples_to_opennmt_format(samples_training, samples_validation, location, samples_test=None, prefix=''):
    if prefix:
        prefix += '-'
    for name, samples in {'training': samples_training, 'validation': samples_validation, 'test': samples_test}.items():
        if samples is None:
            continue

        headlines, articles = map(list, zip(*samples))
        articles = [article.split('\n', 1)[0] for article in articles]

        with open('{}/{}{}-samples.txt'.format(location, prefix, name), mode='wt', encoding='utf-8') as myfile:
            myfile.write('\n'.join(articles))
        with open('{}/{}{}-target.txt'.format(location, prefix, name), mode='wt', encoding='utf-8') as myfile:
            myfile.write('\n'.join(headlines))


def remove_duplicate_headlines(samples):
    headlines = set()
    for corpus in samples:
        for sample in corpus:
            if sample.headline in headlines:
                corpus.remove(sample)
            headlines.add(sample.headline)


def get_all_samples():
    print('Reading samples...')
    samples = []
    dir = '../../news-retriever/data'
    samples.append(read_event_registry_data(*[path.join(dir, f) for f in listdir(dir) if f.endswith('.json')]))
    samples.append(read_crowdflower_economic_data('../../data/Full-Economic-News-DFE-839861.csv'))
    samples.append(read_crowdflower_wikipedia_data('../../data/News-article-wikipedia-DFE.csv'))
    dir = '../../data/reuters21578'
    samples.append(read_reuters_data(*[path.join(dir, f) for f in listdir(dir) if f.endswith('.sgm')]))
    dir = '/Users/xx/Files/opennmt/data/cnn/processed'
    samples.append(read_cnn_dailymail_data(*[path.join(dir, f) for f in listdir(dir) if f.endswith('.json')]))
    dir = '/Users/xx/Files/opennmt/data/dailymail/processed'
    samples.append(read_cnn_dailymail_data(*[path.join(dir, f) for f in listdir(dir) if f.endswith('.json')]))
    print('Found', sum([len(s) for s in samples]), 'samples...')
    return samples


def process_samples(samples, max_length=100, pos_tag=False, pos_tag_tgt=False):
    print('Normalizing...')
    samples = [normalize_samples(corpus, max_length=max_length) for corpus in samples]
    print('Removing duplicate headlines...')
    remove_duplicate_headlines(samples)
    print('Trimmed to', sum([len(s) for s in samples]), 'samples by removing duplicate headlines')
    if pos_tag:
        print('POS tagging...')
        samples = [pos_tag_samples(corpus, tag_tgt=pos_tag_tgt) for corpus in samples]
    return samples


def split(samples, validation_pct, test_pct):
    training_samples = []
    validation_samples = []
    test_samples = []
    for corpus_samples in samples:
        seed(448)
        shuffle(corpus_samples)
        total_samples = len(corpus_samples)
        training_boundary = int(total_samples*(1.0-validation_pct-test_pct))
        test_boundary = int(total_samples*(1.0-test_pct))
        training_samples += corpus_samples[:training_boundary]
        validation_samples += corpus_samples[training_boundary:test_boundary]
        test_samples += corpus_samples[test_boundary:]
    print('Split into {} training samples, {} validation samples, {} test samples'
          .format(len(training_samples), len(validation_samples), len(test_samples)))
    return training_samples, validation_samples, test_samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', required=True, type=str, help='output directory')
    parser.add_argument('-x', '--prefix', type=str, help='output filename prefix', default='')
    parser.add_argument('-m', '--max_length', type=int, help='max length of src', default=100)
    parser.add_argument('-p', '--pos_tag', type=bool, help='POS tag the source data', default=False)
    parser.add_argument('-pt', '--pos_tag_tgt', type=bool, help='POS tag the target', default=False)
    parser.add_argument('-v', '--val_pct', type=float, help='percent validation data', default=.15)
    parser.add_argument('-t', '--test_pct', type=float, help='percent test data', default=.1)
    args = parser.parse_args()

    print('Got the following arguments:', args)
    input('Press any key to continue...')

    all_samples = get_all_samples()
    all_samples = process_samples(all_samples, max_length=args.max_length, pos_tag=args.pos_tag, pos_tag_tgt=args.pos_tag_tgt)
    training, validation, test = split(all_samples, validation_pct=args.val_pct, test_pct=args.test_pct)
    write_samples_to_opennmt_format(samples_training=training,
                                    samples_validation=validation,
                                    samples_test=test,
                                    location=args.output, prefix=args.prefix)
