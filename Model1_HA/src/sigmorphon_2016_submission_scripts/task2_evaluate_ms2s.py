"""Trains and evaluates a joint-structured-model for inflection generation, using the sigmorphon 2016 shared task data
files and evaluation script.

Usage:
  task2_evaluate_ms2s.py [--dynet-mem MEM][--input=INPUT] [--feat-input=FEAT][--hidden=HIDDEN]
  [--epochs=EPOCHS] [--layers=LAYERS] [--optimization=OPTIMIZATION] [--nbest=NBEST]
  TRAIN_PATH TEST_PATH RESULTS_PATH SIGMORPHON_PATH...

Arguments:
  TRAIN_PATH    train data path
  TEST_PATH     test data path
  RESULTS_PATH  results file to load the models from
  SIGMORPHON_PATH   sigmorphon root containing data, src dirs

Options:
  -h --help                     show this help message and exit
  --dynet-mem MEM               allocates MEM bytes for dynet
  --input=INPUT                 input vector dimensions
  --feat-input=FEAT             feature input vector dimension
  --hidden=HIDDEN               hidden layer dimensions
  --epochs=EPOCHS               amount of training epochs
  --layers=LAYERS               amount of layers in lstm network
  --optimization=OPTIMIZATION   chosen optimization method ADAM/SGD/ADAGRAD/MOMENTUM
  --nbest=NBEST                 amount of nbest results

"""

import time
import docopt
import task2_ms2s
import task2_joint_structured_inflection
import src.prepare_sigmorphon_data as prepare_sigmorphon_data
import datetime
import src.common as common
from dynet import *

# default values
INPUT_DIM = 150
FEAT_INPUT_DIM = 20
HIDDEN_DIM = 150
EPOCHS = 1
LAYERS = 2
MAX_PREDICTION_LEN = 50
OPTIMIZATION = 'ADAM'
NBEST = 1

NULL = '%'
UNK = '#'
UNK_FEAT = '@'
EPSILON = '*'
BEGIN_WORD = '<'
END_WORD = '>'


def main(train_path, test_path, results_file_path, sigmorphon_root_dir, input_dim, hidden_dim, epochs, layers,
         optimization, feat_input_dim, nbest):
    hyper_params = {'INPUT_DIM': input_dim, 'HIDDEN_DIM': hidden_dim, 'EPOCHS': epochs, 'LAYERS': layers,
                    'MAX_PREDICTION_LEN': MAX_PREDICTION_LEN, 'OPTIMIZATION': optimization, 'NBEST': nbest}

    print 'train path = ' + str(train_path)
    print 'test path =' + str(test_path)
    for param in hyper_params:
        print param + '=' + str(hyper_params[param])

    # load data
    (train_target_words, train_source_words, train_target_feat_dicts,
     train_source_feat_dicts) = prepare_sigmorphon_data.load_data(
        train_path, 2)
    (test_target_words, test_source_words, test_target_feat_dicts,
     test_source_feat_dicts) = prepare_sigmorphon_data.load_data(
        test_path, 2)
    alphabet, feature_types = prepare_sigmorphon_data.get_alphabet(train_target_words, train_source_words,
                                                                   train_target_feat_dicts, train_source_feat_dicts)

    # used for character dropout
    alphabet.append(NULL)
    alphabet.append(UNK)

    # used during decoding
    alphabet.append(EPSILON)
    alphabet.append(BEGIN_WORD)
    alphabet.append(END_WORD)

    feature_alphabet = common.get_feature_alphabet(train_source_feat_dicts + train_target_feat_dicts)
    feature_alphabet.append(UNK_FEAT)

    # add indices to alphabet - used to indicate when copying from lemma to word
    for marker in [str(i) for i in xrange(MAX_PREDICTION_LEN)]:
        alphabet.append(marker)

    # feat 2 int
    feat_index = dict(zip(feature_alphabet, range(0, len(feature_alphabet))))

    # char 2 int
    alphabet_index = dict(zip(alphabet, range(0, len(alphabet))))
    inverse_alphabet_index = {index: char for char, index in alphabet_index.items()}

    # cluster the data by POS type (features)
    # TODO: do we need to cluster on both source and target feats? 
    #       probably enough to cluster on source here becasue pos will be same
    #       (no derivational morphology in this task)
    train_cluster_to_data_indices = common.cluster_data_by_pos(train_source_feat_dicts)
    test_cluster_to_data_indices = common.cluster_data_by_pos(test_source_feat_dicts)

    # cluster the data by inflection type (features)
    # train_cluster_to_data_indices = common.cluster_data_by_morph_type(train_feat_dicts, feature_types)
    # test_cluster_to_data_indices = common.cluster_data_by_morph_type(test_feat_dicts, feature_types)

    accuracies = []
    final_results = {}

    # factored model: new model per inflection type
    for cluster_index, cluster_type in enumerate(train_cluster_to_data_indices):

        # get the inflection-specific data
        train_cluster_target_words = [train_target_words[i] for i in train_cluster_to_data_indices[cluster_type]]
        if len(train_cluster_target_words) < 1:
            print 'only ' + str(len(train_cluster_target_words)) + ' samples for this inflection type. skipping'
            continue
        else:
            print 'now evaluating model for cluster ' + str(cluster_index + 1) + '/' + \
                  str(len(train_cluster_to_data_indices)) + ': ' + cluster_type + ' with ' + \
                  str(len(train_cluster_target_words)) + ' examples'

        # test best model

        test_cluster_source_words = [test_source_words[i] for i in test_cluster_to_data_indices[cluster_type]]
        test_cluster_target_words = [test_target_words[i] for i in test_cluster_to_data_indices[cluster_type]]
        test_cluster_source_feat_dicts = [test_source_feat_dicts[i] for i in test_cluster_to_data_indices[cluster_type]]
        test_cluster_target_feat_dicts = [test_target_feat_dicts[i] for i in test_cluster_to_data_indices[cluster_type]]

        # load best model
        best_model, params = load_best_model(str(cluster_index), alphabet,
                                                                              results_file_path, input_dim,
                                                                              hidden_dim, layers,
                                                                              feature_alphabet, feat_input_dim,
                                                                              feature_types)

        lang = train_path.split('/')[-1].replace('-task{0}-train'.format('1'), '')

        # handle greedy prediction
        if nbest == 1:
            is_nbest = False
            predicted_templates = task2_ms2s.predict_templates(
                best_model,
                params,
                alphabet_index,
                inverse_alphabet_index,
                test_cluster_source_words,
                test_cluster_source_feat_dicts,
                test_cluster_target_feat_dicts,
                feat_index,
                feature_types)

            accuracy = task2_ms2s.evaluate_model(predicted_templates,
                                                 test_cluster_source_words,
                                                 test_cluster_source_feat_dicts,
                                                 test_cluster_target_words,
                                                 test_cluster_target_feat_dicts,
                                                 feature_types,
                                                 print_results=False)
            accuracies.append(accuracy)
            print '{0} {1} accuracy: {2}'.format(lang, cluster_type, accuracy[1])

            # get predicted_templates in the same order they appeared in the original file
            # iterate through them and foreach concat morph, lemma, features in order to print later in the task format
            for i in test_cluster_to_data_indices[cluster_type]:
                joint_index = test_source_words[i] + ':' + common.get_morph_string(test_source_feat_dicts[i],
                                                                                   feature_types) \
                              + ':' + common.get_morph_string(test_target_feat_dicts[i], feature_types)
                inflection = task2_ms2s.instantiate_template(
                    predicted_templates[joint_index], test_source_words[i])
                final_results[i] = (
                test_source_words[i], test_source_feat_dicts[i], inflection, test_target_feat_dicts[i])

            micro_average_accuracy = accuracy[1]

        # handle n-best prediction
        else:
            is_nbest = True

            predicted_nbset_templates = task2_ms2s.predict_nbest_templates(
                best_model,
                params,
                alphabet_index,
                inverse_alphabet_index,
                test_cluster_source_words,
                test_cluster_source_feat_dicts,
                test_cluster_target_feat_dicts,
                feat_index,
                feature_types,
                nbest,
                test_cluster_target_words)

            # get predicted_templates in the same order they appeared in the original file
            # iterate through them and foreach concat morph, lemma, features in order to print later in the task format
            for i in test_cluster_to_data_indices[cluster_type]:
                joint_index = test_source_words[i] + ':' + common.get_morph_string(test_source_feat_dicts[i],
                                                                                   feature_types) \
                              + ':' + common.get_morph_string(test_target_feat_dicts[i], feature_types)

                nbest_inflections = []
                templates = [t for (t, p) in predicted_nbset_templates[joint_index]]
                for template in templates:
                    nbest_inflections.append(
                        task2_ms2s.instantiate_template(
                            template,
                            test_source_words[i]))
                final_results[i] = (
                test_source_words[i], test_source_feat_dicts[i], nbest_inflections, test_target_feat_dicts[i])

            micro_average_accuracy = -1

    if 'test' in test_path:
        suffix = '.best.test'
    else:
        suffix = '.best'

    task2_joint_structured_inflection.write_results_file(hyper_params,
                                                         micro_average_accuracy,
                                                         train_path,
                                                         test_path,
                                                         results_file_path + suffix,
                                                         sigmorphon_root_dir,
                                                         final_results,
                                                         is_nbest)


def load_best_model(morph_index, alphabet, results_file_path, input_dim, hidden_dim, layers, feature_alphabet,
                    feat_input_dim, feature_types):
    tmp_model_path = results_file_path + '_' + morph_index + '_bestmodel.txt'
    print 'trying to open ' + tmp_model_path

    params = {}

    model = Model()

    # character embeddings
    params["char_lookup"] = model.add_lookup_parameters((len(alphabet), input_dim))

    # feature embeddings
    # TODO: add another input dim for features?
    params["feat_lookup"] = model.add_lookup_parameters((len(feature_alphabet), feat_input_dim))

    # used in softmax output
    params["R"] = model.add_parameters((len(alphabet), hidden_dim))
    params["bias"] = model.add_parameters(len(alphabet))

    # rnn's
    params["encoder_frnn"] = LSTMBuilder(layers, input_dim, hidden_dim, model)
    params["encoder_rrnn"] = LSTMBuilder(layers, input_dim, hidden_dim, model)

    # TODO: inspect carefully, as dims may be sub-optimal in some cases (many feature types?)
    # 2 * HIDDEN_DIM + 3 * INPUT_DIM + len(feats) * FEAT_INPUT_DIM, as it gets a concatenation of frnn, rrnn
    # (both of HIDDEN_DIM size), previous output char, current lemma char (of INPUT_DIM size) current index char
    # and feats * FEAT_INPUT_DIM
    # 2 * len(feature_types) * feat_input_dim, as it gets both source and target feature embeddings
    params["decoder_rnn"] = LSTMBuilder(layers, 2 * hidden_dim + 3 * input_dim + 2 * len(feature_types) * feat_input_dim,
                              hidden_dim,
                              model)

    model.populate(tmp_model_path)
    return model, params


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
    if arguments['--nbest']:
        nbest = int(arguments['--nbest'])
    else:
        nbest = NBEST

    print arguments

    main(train_path, test_path, results_file_path, sigmorphon_root_dir, input_dim, hidden_dim, epochs, layers,
         optimization, feat_input_dim, nbest)
