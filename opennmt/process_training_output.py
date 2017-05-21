import argparse
from statistics import mean, median

from dateutil.parser import parse
from recordclass import recordclass

import re

Training = recordclass('Training', ['parameters', 'epochs', 'time'])
Parameters = recordclass('Parameters', ['vocabulary_size', 'additional_features', 'max_sequence_length',
                         'training_samples', 'batches', 'encoder_structure', 'decoder_structure'])
Structure = recordclass('Structure', ['layers', 'hidden_nodes', 'dropout', 'embedding_size'])
Epoch = recordclass('Epoch', ['learning_rate', 'training_perplexities', 'validation_perplexity', 'time'])


def parse_file(file):
    line_regex = re.compile('\[(\d\d/\d\d/\d\d \d\d:\d\d:\d\d) INFO\] (.*)')
    epoch_regex = re.compile('Epoch (\d+) ; .* ; (?:Learning rate|Optim SGD LR) (\d*\.\d*) ; .* ; Perplexity (\d*\.\d*)')
    validation_regex = re.compile('Validation perplexity: (\d*\.\d*)')
    epochs = []
    current_epoch = None
    start_time = None
    end_time = None
    epoch_start_time = None
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
                    epochs.append(Epoch(learning_rate=learning_rate, training_perplexities=[], validation_perplexity=None, time=None))
                    current_epoch = epoch
                    epoch_start_time = parse(time)
                if epochs[-1].learning_rate != learning_rate:
                    raise Exception
                epochs[-1].training_perplexities.append(perplexity)
                continue
            validation_match = validation_regex.match(content)
            if validation_match:
                validation_perplexity = validation_match.group(1)
                epochs[-1].validation_perplexity = validation_perplexity
                epochs[-1].time = parse(time) - epoch_start_time
                continue
    return Training(parameters=None, epochs=epochs, time=end_time-start_time)


def write_mat_file():
    fileOutput = open("matfile.txt", "w")

    for epoch in training.epochs:
        mean_training_perplexity = mean(epoch.training_perplexities)
        mean_train_perplex = str(mean_training_perplexity)

        training_perplexities = (epoch.training_perplexities[-1])
        train_perplex = str(training_perplexities)

        validation_perplexities = (epoch.validation_perplexity)
        val_perplexity = str(validation_perplexities)

        fileOutput.write(mean_train_perplex)
        fileOutput.write("\t")
        fileOutput.write(train_perplex)
        fileOutput.write("\t")
        fileOutput.write(val_perplexity)
        fileOutput.write("\n")

        print(mean(epoch.training_perplexities), epoch.training_perplexities[-1], epoch.validation_perplexity)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='file output of the training')
    args = parser.parse_args()

    training = parse_file(args.input)
    [print(epoch.time) for epoch in training.epochs]
    time = median([epoch.time.seconds for epoch in training.epochs])
    print('{}:{}'.format(int(time / 60), time % 60))
    [print(epoch.validation_perplexity) for epoch in training.epochs]
    # write_mat_file()



