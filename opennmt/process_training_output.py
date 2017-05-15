import argparse
from statistics import mean

from dateutil.parser import parse
from recordclass import recordclass

import re

Training = recordclass('Training', ['parameters', 'epochs', 'time'])
Parameters = recordclass('Parameters', ['vocabulary_size', 'additional_features', 'max_sequence_length',
                         'training_samples', 'batches', 'encoder_structure', 'decoder_structure'])
Structure = recordclass('Structure', ['layers', 'hidden_nodes', 'dropout', 'embedding_size'])
Epoch = recordclass('Epoch', ['learning_rate', 'training_perplexities', 'validation_perplexity'])


def parse_file(file):
    line_regex = re.compile('\[(\d\d/\d\d/\d\d \d\d:\d\d:\d\d) INFO\] (.*)')
    epoch_regex = re.compile('Epoch (\d+) ; .* ; Learning rate (\d*\.\d*) ; .* ; Perplexity (\d*\.\d*)')
    validation_regex = re.compile('Validation perplexity: (\d*\.\d*)')
    epochs = []
    current_epoch = None
    start_time = None
    end_time = None
    with open(file) as fp:
        for line in fp:
            if not line_regex.match(line):
                print(line)
                continue
            time, content = line_regex.match(line).group(1, 2)
            if not start_time:
                start_time = parse(time)
            end_time = parse(time)

            epoch_match = epoch_regex.match(content)
            if epoch_match:
                epoch, learning_rate, perplexity = epoch_match.group(1, 2, 3)
                epoch, learning_rate, perplexity = int(epoch), float(learning_rate), float(perplexity)
                if epoch is not current_epoch:
                    epochs.append(Epoch(learning_rate=learning_rate, training_perplexities=[], validation_perplexity=None))
                    current_epoch = epoch
                if epochs[-1].learning_rate != learning_rate:
                    raise Exception
                epochs[-1].training_perplexities.append(perplexity)
                continue
            validation_match = validation_regex.match(content)
            if validation_match:
                validation_perplexity = validation_match.group(1)
                epochs[-1].validation_perplexity = validation_perplexity
                continue
    return Training(parameters=None, epochs=epochs, time=end_time-start_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='file output of the training')
    args = parser.parse_args()

    training = parse_file(args.input)

    for epoch in training.epochs:
        print(mean(epoch.training_perplexities), epoch.training_perplexities[-1], epoch.validation_perplexity)
