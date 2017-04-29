import random

from data.prepare_data import prepare_data
from data.read_data import read_crowdflower_wikipedia_data
from rnn.rnn import EncoderRNN, AttnDecoderRNN, train_epochs, evaluate

data_set = read_crowdflower_wikipedia_data('../data/News-article-wikipedia-DFE.csv')
lang, data_set = prepare_data(data_set)


def evaluate_randomly(encoder, decoder, n=10):
    for i in range(n):
        sample = random.choice(data_set)
        print('input >', sample.body)
        print('actual =', sample.headline)
        output_words, attentions = evaluate(encoder, decoder, lang, sample.body)
        output_sentence = ' '.join(output_words)
        print('predicted <', output_sentence)
        print('')


hidden_size = 256
encoder1 = EncoderRNN(lang.n_words, hidden_size)
attn_decoder1 = AttnDecoderRNN(hidden_size, lang.n_words, 1, dropout_p=0.1)
train_epochs(data_set, lang, encoder1, attn_decoder1, n_epochs=50, print_every=5)
evaluate_randomly(encoder1, attn_decoder1)
