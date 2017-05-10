import csv
import json
import random

import re
from bs4 import BeautifulSoup


from data.language import Sample


def read_event_registry_data(*files):
    # output from news-retriever project
    source_suffixes = [' \| .*', ' - Amateur Photographer', ' - Kabayan Weekly', ' - Daily Journal',
                       ' - Mobile News Online', ' - The Lethbridge Herald - News and Sports from around Lethbridge',
                       ' - Vanguard News', ' - Futurity', ' - CFN Media', ' - Kuwait Times', ' - Voice of Asia Online',
                       ' - Cyprus Mail', ' - Daily Post Nigeria', ' - MoneyWeek', ' - BBC News', ' - Reuters Africa',
                       ' - Vogue', ' - World News', ' - Ariana News']

    clean = lambda h: re.sub('|'.join(source_suffixes), '', h)

    articles = []
    processed = set()
    for file in files:
        with open(file) as data_file:
            data = json.load(data_file)
            [articles.append(Sample(headline=clean(v['info']['title']), body=v['info']['body']))
             for (k, v) in data.items()
             if 'error' not in v.keys()
             and v['info']['id'] not in processed
             and v['info']['url'] is not 'http://www.theaustralian.com.au/video']
            [processed.add(v['info']['id'])
             for (k, v) in data.items()
             if 'error' not in v.keys()]
    return articles


def read_reuters_data(*files):
    # http://www.daviddlewis.com/resources/testcollections/reuters21578/
    articles = []
    for file in files:
        try:
            soup = BeautifulSoup(open(file), 'html.parser')
            for article in soup.find_all('reuters'):
                if article.title and article.body:
                    articles.append(Sample(headline=article.title.text, body=article.body.text))
        except:
            print('Error processing file:', file)
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
    # https://www.crowdflower.com/wp-content/uploads/2016/03/Full-Economic-News-DFE-839861.csv
    return read_crowdflower_data(*files, headline_name='headline', text_name='text')


def read_crowdflower_wikipedia_data(*files):
    # https://www.crowdflower.com/wp-content/uploads/2016/03/News-article-wikipedia-DFE.csv
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

    samples = read_reuters_data('../../data/reuters21578/reut2-000.sgm')
    print('\nReuters', len(samples), 'samples')
    print(random.choice(samples))

