"""Trains and evaluates a factored-model for inflection generation, using the sigmorphon 2016 shared task data
files and evaluation script.

Usage:
  evaluate_best_factored_structured_models.py [--cnn-mem MEM][--input=INPUT] [--hidden=HIDDEN] [--epochs=EPOCHS]
  [--layers=LAYERS] [--optimization=OPTIMIZATION] TRAIN_PATH TEST_PATH RESULTS_PATH SIGMORPHON_PATH...

Arguments:
  TRAIN_PATH    destination path
  TEST_PATH     test path
  RESULTS_PATH  results file to be written
  SIGMORPHON_PATH   sigmorphon root containing data, src dirs

Options:
  -h --help                     show this help message and exit
  --cnn-mem MEM                 allocates MEM bytes for (py)cnn
  --input=INPUT                 input vector dimensions
  --hidden=HIDDEN               hidden layer dimensions
  --epochs=EPOCHS               amount of training epochs
  --layers=LAYERS               amount of layers in lstm network
  --optimization=OPTIMIZATION   chosen optimization method ADAM/SGD/ADAGRAD/MOMENTUM
"""

import time
import docopt
import task1_factored_structured_inflection
import task1_factored_inflection
import prepare_sigmorphon_data
import datetime
import common
from pycnn import *

# default values
INPUT_DIM = 100
HIDDEN_DIM = 100
EPOCHS = 1
LAYERS = 2
MAX_PREDICTION_LEN = 50
OPTIMIZATION = 'ADAM'
EARLY_STOPPING = True
MAX_PATIENCE = 50
REGULARIZATION = 0.0001
LEARNING_RATE = 0.001  # 0.1

NULL = '%'
UNK = '#'
EPSILON = '*'
BEGIN_WORD = '<'
END_WORD = '>'


def main(train_path, test_path, results_file_path, sigmorphon_root_dir, input_dim, hidden_dim, epochs, layers,
         optimization):
    hyper_params = {'INPUT_DIM': input_dim, 'HIDDEN_DIM': hidden_dim, 'EPOCHS': epochs, 'LAYERS': layers,
                    'MAX_PREDICTION_LEN': MAX_PREDICTION_LEN, 'OPTIMIZATION': optimization}

    print 'train path = ' + str(train_path)
    print 'test path = ' + str(test_path)
    for param in hyper_params:
        print param + '=' + str(hyper_params[param])

    # load data
    (train_words, train_lemmas, train_feat_dicts) = prepare_sigmorphon_data.load_data(train_path)
    (test_words, test_lemmas, test_feat_dicts) = prepare_sigmorphon_data.load_data(test_path)
    alphabet, feats = prepare_sigmorphon_data.get_alphabet(train_words, train_lemmas, train_feat_dicts)

    # used for character dropout
    alphabet.append(NULL)
    alphabet.append(UNK)

    # used during decoding
    alphabet.append(EPSILON)
    alphabet.append(BEGIN_WORD)
    alphabet.append(END_WORD)

    # add indices to alphabet - used to indicate when copying from lemma to word
    for marker in [str(i) for i in xrange(MAX_PREDICTION_LEN)]:
        alphabet.append(marker)

    # char 2 int
    alphabet_index = dict(zip(alphabet, range(0, len(alphabet))))
    inverse_alphabet_index = {index: char for char, index in alphabet_index.items()}

    # cluster the data by inflection type (features)
    train_morph_to_data_indices = common.cluster_data_by_morph_type(train_feat_dicts, feats)
    test_morph_to_data_indices = common.cluster_data_by_morph_type(test_feat_dicts, feats)

    accuracies = []
    final_results = {}

    # factored model: new model per inflection type
    for morph_index, morph_type in enumerate(train_morph_to_data_indices):

        # get the inflection-specific data
        train_morph_words = [train_words[i] for i in train_morph_to_data_indices[morph_type]]
        if len(train_morph_words) < 1:
            print 'only ' + str(len(train_morph_words)) + ' samples for this inflection type. skipping'
            continue
        else:
            print 'now evaluating model for morph ' + str(morph_index) + '/' + str(len(train_morph_to_data_indices)) + \
                  ': ' + morph_type + ' with ' + str(len(train_morph_words)) + ' examples'

        # test best model
        try:
            test_morph_lemmas = [test_lemmas[i] for i in test_morph_to_data_indices[morph_type]]
            test_morph_words = [test_words[i] for i in test_morph_to_data_indices[morph_type]]

            # load best model
            best_model, encoder_frnn, encoder_rrnn, decoder_rnn = load_best_model(str(morph_index), alphabet,
                                                                                  results_file_path, input_dim,
                                                                                  hidden_dim, layers)

            predicted_templates = task1_factored_structured_inflection.predict_templates(best_model, decoder_rnn,
                                                                                         encoder_frnn, encoder_rrnn,
                                                                                         alphabet_index,
                                                                                         inverse_alphabet_index,
                                                                                         test_morph_lemmas)

            accuracy = task1_factored_structured_inflection.evaluate_model(predicted_templates, test_morph_lemmas,
                                                                           test_morph_words)
            accuracies.append(accuracy)

            # get predicted_templates in the same order they appeared in the original file
            # iterate through them and foreach concat morph, lemma, features in order to print later in the task format
            for i in test_morph_to_data_indices[morph_type]:
                inflection = task1_factored_structured_inflection.instantiate_template(
                    predicted_templates[test_lemmas[i]], test_lemmas[i])
                final_results[i] = (test_lemmas[i], inflection, morph_type)

        except KeyError:
            print 'could not find relevant examples in test data for morph: ' + morph_type

    accuracy_vals = [accuracies[i][1] for i in xrange(len(accuracies))]
    macro_avg_accuracy = sum(accuracy_vals) / len(accuracies)
    print 'macro avg accuracy: ' + str(macro_avg_accuracy)

    mic_nom = sum([accuracies[i][0] * accuracies[i][1] for i in xrange(len(accuracies))])
    mic_denom = sum([accuracies[i][0] for i in xrange(len(accuracies))])
    micro_average_accuracy = mic_nom / mic_denom
    print 'micro avg accuracy: ' + str(micro_average_accuracy)

    task1_factored_inflection.write_results_file(hyper_params, macro_avg_accuracy, micro_average_accuracy,
                                                 train_path, test_path, results_file_path + '.best',
                                                 sigmorphon_root_dir, final_results)


def load_best_model(morph_index, alphabet, results_file_path, input_dim, hidden_dim, layers):
    tmp_model_path = results_file_path + '_' + morph_index + '_bestmodel.txt'

    new_model = Model()

    # character embeddings
    new_model.add_lookup_parameters("lookup", (len(alphabet), input_dim))

    # used in softmax output
    new_model.add_parameters("R", (len(alphabet), hidden_dim))
    new_model.add_parameters("bias", len(alphabet))

    # rnn's
    encoder_frnn = LSTMBuilder(layers, input_dim, hidden_dim, new_model)
    encoder_rrnn = LSTMBuilder(layers, input_dim, hidden_dim, new_model)

    # 2 * HIDDEN_DIM + 3 * INPUT_DIM, as it gets a concatenation of frnn, rrnn, previous output char,
    # current lemma char, current index
    decoder_rnn = LSTMBuilder(layers, 2 * hidden_dim + 3 * input_dim, hidden_dim, new_model)

    new_model.load(tmp_model_path)
    return new_model, encoder_frnn, encoder_rrnn, decoder_rnn


if __name__ == '__main__':
    arguments = docopt.docopt(__doc__)

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

    # default values
    if arguments['TRAIN_PATH']:
        train_path = arguments['TRAIN_PATH']
    else:
        train_path = '/Users/roeeaharoni/research_data/sigmorphon2016-master/data/turkish-task1-train'
    if arguments['TEST_PATH']:
        test_path = arguments['TEST_PATH']
    else:
        test_path = '/Users/roeeaharoni/research_data/sigmorphon2016-master/data/turkish-task1-dev'
    if arguments['RESULTS_PATH']:
        results_file_path = arguments['RESULTS_PATH']
    else:
        results_file_path = '/Users/roeeaharoni/Dropbox/phd/research/morphology/inflection_generation/results/results_'\
                     + st + '.txt'
    if arguments['SIGMORPHON_PATH']:
        sigmorphon_root_dir = arguments['SIGMORPHON_PATH'][0]
    else:
        sigmorphon_root_dir = '/Users/roeeaharoni/research_data/sigmorphon2016-master/'
    if arguments['--input']:
        input_dim = int(arguments['--input'])
    else:
        input_dim = INPUT_DIM
    if arguments['--hidden']:
        hidden_dim = int(arguments['--hidden'])
    else:
        hidden_dim = HIDDEN_DIM
    if arguments['--epochs']:
        epochs = int(arguments['--epochs'])
    else:
        epochs = EPOCHS
    if arguments['--layers']:
        layers = int(arguments['--layers'])
    else:
        layers = LAYERS
    if arguments['--optimization']:
        optimization = arguments['--optimization']
    else:
        optimization = OPTIMIZATION

    print arguments

    main(train_path, test_path, results_file_path, sigmorphon_root_dir, input_dim, hidden_dim, epochs, layers,
         optimization)
