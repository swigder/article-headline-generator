import csv
import json
import random
from collections import namedtuple


Sample = namedtuple('Sample', ['headline', 'body'])


def read_event_registry_data(*files):
    articles = []
    for file in files:
        with open(file) as data_file:
            data = json.load(data_file)
            [articles.append(Sample(headline=v['info']['title'], body=v['info']['body']))
             for (k, v) in data.items()
             if 'error' not in v.keys()]
    return articles


def read_crowdflower_data(*files, headline_name, text_name):
    articles = []
    for file in files:
        with open(file) as csv_file:
            data_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
            headline, body = None, None
            for sample in data_reader:
                if headline is None:
                    headline = sample.index(headline_name)
                    body = sample.index(text_name)
                    continue
                articles.append(Sample(headline=sample[headline], body=sample[body]))

    return articles


def read_crowdflower_economic_data(*files):
    return read_crowdflower_data(*files, headline_name='headline', text_name='text')


def read_crowdflower_wikipedia_data(*files):
    return read_crowdflower_data(*files, headline_name='article', text_name='newdescp')


if __name__ == '__main__':
    samples = read_event_registry_data('../../news-retriever/data/data.json',
                                       '../../news-retriever/data/data-1000.json',
                                       '../../news-retriever/data/data-20170428-144403.json',
                                       '../../news-retriever/data/data-20170428-144520.json',
                                       '../../news-retriever/data/data-20170428-144543.json')
    print('\nEvent Registry, ', len(samples), 'samples')
    print(random.choice(samples))

    samples = read_crowdflower_economic_data('../../data/Full-Economic-News-DFE-839861.csv')
    print('\nCrowdflower economic', len(samples), 'samples')
    print(random.choice(samples))

    samples = read_crowdflower_wikipedia_data('../../data/News-article-wikipedia-DFE.csv')
    print('\nCrowdflower Wikipedia', len(samples), 'samples')
    print(random.choice(samples))

