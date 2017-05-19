import argparse

from dateutil.parser import parse
from collections import namedtuple

import re


Translation = namedtuple('Translation', ['gold', 'predicted'])
Previous = namedtuple('Previous', ['i', 'text', 'type'])


def parse_file(file):
    '''
[05/17/17 10:20:32 INFO] GOLD 27058: katy perry s dark horse video features man with allah necklace being disintegrated mail online
[05/17/17 10:20:32 INFO] GOLD SCORE: -70.67
[05/17/17 10:20:32 INFO] PRED 27058: katy perry calls for youtube to remove katy perry s new video mail online
[05/17/17 10:20:32 INFO] PRED SCORE: -7.40
    :param file:
    :return:
    '''
    line_regex = re.compile('\[(\d\d/\d\d/\d\d \d\d:\d\d:\d\d) INFO\] (.*)')
    output_regex = re.compile('(GOLD|PRED) (\d+): (.+)')
    empty_regex = re.compile('\[(\d\d/\d\d/\d\d \d\d:\d\d:\d\d) WARNING\] Line \d+ is empty')

    previous = Previous(i=0, text=None, type='PRED')
    translations = []
    with open(file) as fp:
        for line in fp:
            match = line_regex.match(line)
            if not match:
                if empty_regex.match(line):
                    previous = Previous(i=previous.i+1, text=None, type='PRED')
                print(line)
                continue
            time, content = match.group(1, 2)
            match = output_regex.match(content)
            if not match:
                continue
            output_type, output_i, output_text = match.group(1, 2, 3)
            output_i, output_text = int(output_i), output_text.strip()
            if output_text.endswith(' daily mail online'):
                output_text = output_text[:-len(' daily mail online')]
            if output_type == previous.type:
                raise Exception('current: {}, previous: {}'.format(output_type, previous.type))
            if output_type == 'GOLD':
                if output_i != previous.i + 1:
                    raise Exception('current: {}, previous: {}'.format(output_i, previous.i))
            elif output_type == 'PRED':
                if output_i != previous.i:
                    raise Exception('current: {}, previous: {}'.format(output_i, previous.i))
                translations.append(Translation(gold=previous.text.strip(), predicted=output_text.strip()))
            previous = Previous(i=output_i, text=output_text, type=output_type)
    return translations


def output_to_meteor(translations, output_location):
    gold = [t.gold for t in translations]
    pred = [t.predicted for t in translations]

    with open(output_location + '-gold.txt', 'w') as f:
        f.write('\n'.join(gold))
    with open(output_location + '-pred.txt', 'w') as f:
        f.write('\n'.join(pred))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='file output of the translator')
    parser.add_argument('-m', '--meteor', type=str, help='output location for meteor files (directory and file prefix)')
    args = parser.parse_args()

    translations = parse_file(args.input)

    if args.meteor:
        output_to_meteor(translations, args.meteor)
