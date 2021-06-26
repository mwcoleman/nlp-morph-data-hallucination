"""Trains and evaluates a joint-structured-model for inflection generation, using the sigmorphon 2016 shared task data
files and evaluation script.

Usage:
  evaluate_best_nfst_models.py [--cnn-mem MEM][--input=INPUT] [--feat-input=FEAT][--hidden=HIDDEN]
  [--epochs=EPOCHS] [--layers=LAYERS] [--optimization=OPTIMIZATION] TRAIN_PATH TEST_PATH RESULTS_PATH SIGMORPHON_PATH...

Arguments:
  TRAIN_PATH    train data path
  TEST_PATH     test data path
  RESULTS_PATH  results file to load the models from
  SIGMORPHON_PATH   sigmorphon root containing data, src dirs

Options:
  -h --help                     show this help message and exit
  --cnn-mem MEM                 allocates MEM bytes for (py)cnn
  --input=INPUT                 input vector dimensions
  --feat-input=FEAT             feature input vector dimension
  --hidden=HIDDEN               hidden layer dimensions
  --epochs=EPOCHS               amount of training epochs
  --layers=LAYERS               amount of layers in lstm network
  --optimization=OPTIMIZATION   chosen optimization method ADAM/SGD/ADAGRAD/MOMENTUM

"""

import time
import docopt
import task1_ndst_wo_ij_feats
import prepare_sigmorphon_data
import datetime
import common
from pycnn import *

# default values
INPUT_DIM = 150
FEAT_INPUT_DIM = 20
HIDDEN_DIM = 150
EPOCHS = 1
LAYERS = 2
MAX_PREDICTION_LEN = 50
OPTIMIZATION = 'ADAM'

NULL = '%'
UNK = '#'
EPSILON = '*'
BEGIN_WORD = '<'
END_WORD = '>'
UNK_FEAT = '@'
STEP = '^'
ALIGN_SYMBOL = '~'


def main(train_path, test_path, results_file_path, sigmorphon_root_dir, input_dim, hidden_dim, epochs, layers,
         optimization, feat_input_dim):
    hyper_params = {'INPUT_DIM': input_dim, 'HIDDEN_DIM': hidden_dim, 'FEAT_INPUT_DIM': feat_input_dim,
                    'EPOCHS': epochs, 'LAYERS': layers, 'MAX_PREDICTION_LEN': MAX_PREDICTION_LEN,
                    'OPTIMIZATION': optimization}


    print 'train path = ' + str(train_path)
    print 'test path =' + str(test_path)
    for param in hyper_params:
        print param + '=' + str(hyper_params[param])

    # load train and test data
    (train_words, train_lemmas, train_feat_dicts) = prepare_sigmorphon_data.load_data(train_path)
    (test_words, test_lemmas, test_feat_dicts) = prepare_sigmorphon_data.load_data(test_path)
    alphabet, feature_types = prepare_sigmorphon_data.get_alphabet(train_words, train_lemmas, train_feat_dicts)

    # used for character dropout
    alphabet.append(NULL)
    alphabet.append(UNK)

    # used during decoding
    alphabet.append(EPSILON)
    alphabet.append(BEGIN_WORD)
    alphabet.append(END_WORD)

    # add indices to alphabet - used to indicate when copying from lemma to word
    for marker in [str(i) for i in xrange(3 * MAX_PREDICTION_LEN)]:
        alphabet.append(marker)

    # indicates the FST to step forward in the input
    alphabet.append(STEP)

    # char 2 int
    alphabet_index = dict(zip(alphabet, range(0, len(alphabet))))
    inverse_alphabet_index = {index: char for char, index in alphabet_index.items()}

    # feat 2 int
    feature_alphabet = common.get_feature_alphabet(train_feat_dicts)
    feature_alphabet.append(UNK_FEAT)
    feat_index = dict(zip(feature_alphabet, range(0, len(feature_alphabet))))

    # cluster the data by POS type (features)
    train_cluster_to_data_indices = common.cluster_data_by_pos(train_feat_dicts)
    test_cluster_to_data_indices = common.cluster_data_by_pos(test_feat_dicts)

    # cluster the data by inflection type (features)
    # train_cluster_to_data_indices = common.cluster_data_by_morph_type(train_feat_dicts, feature_types)
    # test_cluster_to_data_indices = common.cluster_data_by_morph_type(test_feat_dicts, feature_types)

    accuracies = []
    final_results = {}

    # factored model: new model per inflection type
    for cluster_index, cluster_type in enumerate(train_cluster_to_data_indices):

        # get the inflection-specific data
        train_cluster_words = [train_words[i] for i in train_cluster_to_data_indices[cluster_type]]
        if len(train_cluster_words) < 1:
            print 'only ' + str(len(train_cluster_words)) + ' samples for this inflection type. skipping'
            continue
        else:
            print 'now evaluating model for cluster ' + str(cluster_index + 1) + '/' + \
                  str(len(train_cluster_to_data_indices)) + ': ' + cluster_type + ' with ' + \
                  str(len(train_cluster_words)) + ' examples'

        # test best model
        try:
            test_cluster_lemmas = [test_lemmas[i] for i in test_cluster_to_data_indices[cluster_type]]
            test_cluster_words = [test_words[i] for i in test_cluster_to_data_indices[cluster_type]]
            test_cluster_feat_dicts = [test_feat_dicts[i] for i in test_cluster_to_data_indices[cluster_type]]

            # load best model
            best_model, encoder_frnn, encoder_rrnn, decoder_rnn = load_best_model(str(cluster_index), alphabet,
                                                                                  results_file_path, input_dim,
                                                                                  hidden_dim, layers,
                                                                                  feature_alphabet, feat_input_dim,
                                                                                  feature_types)

            predicted_templates = task1_ndst_wo_ij_feats.predict_templates(best_model,
                                                               decoder_rnn,
                                                               encoder_frnn,
                                                               encoder_rrnn,
                                                               alphabet_index,
                                                               inverse_alphabet_index,
                                                               test_cluster_lemmas,
                                                               test_cluster_feat_dicts,
                                                               feat_index,
                                                               feature_types)

            accuracy = task1_ndst_wo_ij_feats.evaluate_model(predicted_templates,
                                                 test_cluster_lemmas,
                                                 test_cluster_feat_dicts,
                                                 test_cluster_words,
                                                 feature_types,
                                                 print_results=True)
            accuracies.append(accuracy)

            # get predicted_templates in the same order they appeared in the original file
            # iterate through them and foreach concat morph, lemma, features in order to print later in the task format
            for i in test_cluster_to_data_indices[cluster_type]:
                joint_index = test_lemmas[i] + ':' + common.get_morph_string(test_feat_dicts[i], feature_types)
                inflection = task1_ndst_wo_ij_feats.instantiate_template(
                    predicted_templates[joint_index], test_lemmas[i])
                final_results[i] = (test_lemmas[i], test_feat_dicts[i], inflection)

        except KeyError:
            print 'could not find relevant examples in test data for cluster: ' + cluster_type

    accuracy_vals = [accuracies[i][1] for i in xrange(len(accuracies))]
    macro_avg_accuracy = sum(accuracy_vals) / len(accuracies)
    print 'macro avg accuracy: ' + str(macro_avg_accuracy)

    mic_nom = sum([accuracies[i][0] * accuracies[i][1] for i in xrange(len(accuracies))])
    mic_denom = sum([accuracies[i][0] for i in xrange(len(accuracies))])
    micro_average_accuracy = mic_nom / mic_denom
    print 'micro avg accuracy: ' + str(micro_average_accuracy)

    if 'test' in test_path:
        suffix = '.best.test'
    else:
        suffix = '.best'
    common.write_results_file_and_evaluate_externally(hyper_params, micro_average_accuracy, train_path,
                                                      test_path, results_file_path + suffix, sigmorphon_root_dir,
                                                      final_results)


def load_best_model(morph_index, alphabet, results_file_path, input_dim, hidden_dim, layers, feature_alphabet,
                    feat_input_dim, feature_types):

    tmp_model_path = results_file_path + '_' + morph_index + '_bestmodel.txt'
    model, encoder_frnn, encoder_rrnn, decoder_rnn = task1_ndst_wo_ij_feats.build_model(alphabet, input_dim, hidden_dim,
                                                                                        layers, feature_types,
                                                                                        feat_input_dim,
                                                                                        feature_alphabet)
    print 'trying to load model from: {}'.format(tmp_model_path)
    model.load(tmp_model_path)
    return model, encoder_frnn, encoder_rrnn, decoder_rnn


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
        results_file_path = '/Users/roeeaharoni/Dropbox/phd/research/morphology/inflection_generation/results/results_' \
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
    if arguments['--feat-input']:
        feat_input_dim = int(arguments['--feat-input'])
    else:
        feat_input_dim = FEAT_INPUT_DIM
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
         optimization, feat_input_dim)
