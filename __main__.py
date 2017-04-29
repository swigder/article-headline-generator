from random import random

from data.prepare_data import prepare_data
from data.read_data import read_crowdflower_wikipedia_data

data_set = read_crowdflower_wikipedia_data('../data/News-article-wikipedia-DFE.csv')
lang, data_set = prepare_data(data_set)


# def evaluate_randomly(encoder, decoder, n=10):
#     for i in range(n):
#         pair = random.choice(pairs)
#         print('input >', pair[0])
#         print('actual =', pair[1])
#         output_words, attentions = evaluate(encoder, decoder, input_lang, output_lang, pair[0])
#         output_sentence = ' '.join(output_words)
#         print('predicted <', output_sentence)
#         print('')
#
#
# hidden_size = 256
# encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
# attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, 1, dropout_p=0.1)
# train_epochs(pairs, input_lang, output_lang, encoder1, attn_decoder1, 40000, print_every=5000)
# evaluate_randomly(encoder1, attn_decoder1)
