import time

from os import listdir, path
from random import shuffle, seed

from data.process_data import normalize_samples, pos_tag_samples
from data.read_data import read_event_registry_data, read_crowdflower_economic_data, read_crowdflower_wikipedia_data, \
    read_reuters_data


def write_samples_to_opennmt_format(samples_training, samples_validation, location, prefix=''):
    if prefix:
        prefix += '-'
    for name, samples in {'training': samples_training, 'validation': samples_validation}.items():
        headlines, articles = map(list, zip(*samples))
        articles = [article.split('\n', 1)[0] for article in articles]

        with open('{}/{}{}-samples.txt'.format(location, prefix, name), mode='wt', encoding='utf-8') as myfile:
            myfile.write('\n'.join(articles))
        with open('{}/{}{}-target.txt'.format(location, prefix, name), mode='wt', encoding='utf-8') as myfile:
            myfile.write('\n'.join(headlines))


def remove_duplicate_headlines(samples):
    headlines = set()
    for sample in samples:
        if sample.headline in headlines:
            samples.remove(sample)
        headlines.add(sample.headline)


if __name__ == '__main__':
    er_dir = '../../news-retriever/data'
    samples = read_event_registry_data(*[path.join(er_dir, f) for f in listdir(er_dir) if f.endswith('.json')])
    samples += read_crowdflower_economic_data('../../data/Full-Economic-News-DFE-839861.csv')
    samples += read_crowdflower_wikipedia_data('../../data/News-article-wikipedia-DFE.csv')
    reuters_dir = '../../data/reuters21578'
    samples += read_reuters_data(*[path.join(reuters_dir, f) for f in listdir(reuters_dir) if f.endswith('.sgm')])
    print('Started with', len(samples), 'samples...')
    print('Normalizing...')
    samples = normalize_samples(samples, max_length=100)
    remove_duplicate_headlines(samples)
    print('Trimmed to', len(samples), 'samples by removing duplicate headlines')
    # uncomment to pos tag
    # samples = pos_tag_samples(samples)
    seed(448)
    shuffle(samples)
    for i in range(100):
        print(samples[i].headline)
    total_samples = len(samples)
    training_samples = int(total_samples * .8)
    print('Total samples: {}, training {}, validation {}', total_samples, training_samples, total_samples - training_samples)
    write_samples_to_opennmt_format(samples_training=samples[:training_samples],
                                    samples_validation=samples[training_samples:],
                                    location='/Users/xx/Files/opennmt/data/own', prefix='pos')


