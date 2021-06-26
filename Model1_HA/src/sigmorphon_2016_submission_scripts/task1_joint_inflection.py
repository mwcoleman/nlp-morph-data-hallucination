"""Trains and evaluates a joint-model for inflection generation, using the sigmorphon 2016 shared task data files
and evaluation script.

Usage:
  pycnn_joint_inflection.py [--cnn-mem MEM][--input=INPUT] [--hidden=HIDDEN] [--feat-input=FEAT] [--epochs=EPOCHS]
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
  --feat-input=FEAT             feature input vector dimension
  --hidden=HIDDEN               hidden layer dimensions
  --epochs=EPOCHS               amount of training epochs
  --layers=LAYERS               amount of layers in lstm network
  --optimization=OPTIMIZATION   chosen optimization method ADAM/SGD/ADAGRAD/MOMENTUM/ADADELTA
"""

import numpy as np
import random
import prepare_sigmorphon_data
import progressbar
import datetime
import time
import codecs
import os
import common
from multiprocessing import Pool
# from matplotlib import pyplot as plt
from docopt import docopt
from pycnn import *

# default values
INPUT_DIM = 150
FEAT_INPUT_DIM = 20
HIDDEN_DIM = 150
EPOCHS = 1
LAYERS = 2
MAX_PREDICTION_LEN = 50
OPTIMIZATION = 'ADAM'
EARLY_STOPPING = True
MAX_PATIENCE = 100
REGULARIZATION = 0.0001
LEARNING_RATE = 0.001  # 0.1

NULL = '%'
UNK = '#'
UNK_FEAT = '@'
EPSILON = '*'
BEGIN_WORD = '<'
END_WORD = '>'


def main(train_path, test_path, results_file_path, sigmorphon_root_dir, input_dim, hidden_dim, feat_input_dim, epochs,
         layers, optimization):
    hyper_params = {'INPUT_DIM': input_dim, 'HIDDEN_DIM': hidden_dim, 'FEAT_INPUT_DIM': feat_input_dim,
                    'EPOCHS': epochs, 'LAYERS': layers, 'MAX_PREDICTION_LEN': MAX_PREDICTION_LEN,
                    'OPTIMIZATION': optimization, 'PATIENCE': MAX_PATIENCE, 'REGULARIZATION': REGULARIZATION,
                    'LEARNING_RATE': LEARNING_RATE}
    parallelize_training = True
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

    # char 2 int
    alphabet_index = dict(zip(alphabet, range(0, len(alphabet))))
    inverse_alphabet_index = {index: char for char, index in alphabet_index.items()}

    # feat 2 int
    feature_alphabet = common.get_feature_alphabet(train_feat_dicts)
    feature_alphabet.append(UNK_FEAT)
    feat_index = dict(zip(feature_alphabet, range(0, len(feature_alphabet))))

    # cluster the data by POS type (features)
    train_pos_to_data_indices = common.cluster_data_by_pos(train_feat_dicts)
    test_pos_to_data_indices = common.cluster_data_by_pos(test_feat_dicts)
    train_cluster_to_data_indices = train_pos_to_data_indices
    test_cluster_to_data_indices = test_pos_to_data_indices

    # cluster the data by inflection type (features) - used for sanity check
    # train_morph_to_data_indices = common.cluster_data_by_morph_type(train_feat_dicts, feature_types)
    # test_morph_to_data_indices = common.cluster_data_by_morph_type(test_feat_dicts, feature_types)
    # train_cluster_to_data_indices = train_morph_to_data_indices
    # test_cluster_to_data_indices = test_morph_to_data_indices

    # generate params for each model
    params = []
    for cluster_index, cluster_type in enumerate(train_cluster_to_data_indices):
        params.append([input_dim, hidden_dim, layers, cluster_index, cluster_type, train_lemmas, train_feat_dicts,
                       train_words, test_lemmas, test_feat_dicts, train_cluster_to_data_indices, test_words,
                       test_cluster_to_data_indices, alphabet, alphabet_index, inverse_alphabet_index, epochs,
                       optimization, results_file_path, feat_index, feature_types, feat_input_dim, feature_alphabet])

    # train models in parallel or in loop
    if parallelize_training:
        p = Pool(4, maxtasksperchild=1)
        print 'now training {0} models in parallel'.format(len(train_cluster_to_data_indices))
        p.map(train_cluster_model_wrapper, params)
    else:
        print 'now training {0} models in loop'.format(len(train_cluster_to_data_indices))
        for p in params:
            train_cluster_model(*p)
    print 'finished training all models'

    # evaluate best models
    os.system('python task1_evaluate_best_joint_models.py --cnn-mem 4096 --input={0} --hidden={1} --input-feat {2} \
              --epochs={3} --layers={4} --optimization={5} {6} {7} {8} {9}'.format(input_dim, hidden_dim,
                                                                                   feat_input_dim, epochs, layers,
                                                                                   optimization, train_path, test_path,
                                                                                   results_file_path,
                                                                                   sigmorphon_root_dir))
    return


def train_cluster_model_wrapper(params):
    from matplotlib import pyplot as plt
    return train_cluster_model(*params)


def train_cluster_model(input_dim, hidden_dim, layers, cluster_index, cluster_type, train_lemmas, train_feat_dicts,
                        train_words, test_lemmas, test_feat_dicts, train_cluster_to_data_indices, test_words,
                        test_cluster_to_data_indices, alphabet, alphabet_index, inverse_alphabet_index, epochs,
                        optimization, results_file_path, feat_index, feature_types, feat_input_dim, feature_alphabet):

    # get the cluster-specific train data
    train_cluster_words = [train_words[i] for i in train_cluster_to_data_indices[cluster_type]]
    train_cluster_lemmas = [train_lemmas[i] for i in train_cluster_to_data_indices[cluster_type]]
    train_cluster_feat_dicts = [train_feat_dicts[i] for i in train_cluster_to_data_indices[cluster_type]]
    if len(train_cluster_words) < 1:
        print 'only ' + str(len(train_cluster_words)) + ' samples for this inflection type. skipping'
        return
    else:
        print 'now training model for cluster ' + str(cluster_index + 1) + '/' + \
              str(len(train_cluster_to_data_indices)) + ': ' + cluster_type + ' with ' + \
              str(len(train_cluster_words)) + ' examples'

    # get the cluster-specific dev data
    # TODO: now dev and test are the same - change later when test sets are available
    try:
        # get dev lemmas for early stopping
        dev_cluster_lemmas = [test_lemmas[i] for i in test_cluster_to_data_indices[cluster_type]]
        dev_cluster_words = [test_words[i] for i in test_cluster_to_data_indices[cluster_type]]
        dev_cluster_feat_dicts = [test_feat_dicts[i] for i in test_cluster_to_data_indices[cluster_type]]
    except KeyError:
        dev_cluster_lemmas = []
        dev_cluster_words = []
        dev_cluster_feat_dicts = []
        print 'could not find relevant examples in dev data for cluster: ' + cluster_type

    # build model
    initial_model, encoder_frnn, encoder_rrnn, decoder_rnn = build_model(alphabet, feature_alphabet, feature_types,
                                                                         input_dim, hidden_dim, feat_input_dim,
                                                                         layers)

    # TODO: change so dev will be dev and not test when getting the actual data
    dev_lemmas = dev_cluster_lemmas
    dev_feat_dicts = dev_cluster_feat_dicts
    dev_words = dev_cluster_words
    # dev_lemmas = test_lemmas
    # dev_feat_dicts = test_feat_dicts
    # dev_words = test_words

    # train model
    trained_model = train_model(initial_model, encoder_frnn, encoder_rrnn, decoder_rnn, train_cluster_words,
                                train_cluster_lemmas, train_cluster_feat_dicts, dev_words, dev_lemmas,
                                dev_feat_dicts, alphabet_index, inverse_alphabet_index, feat_index, feature_types,
                                epochs, optimization, results_file_path + '_morph_{0}'.format(cluster_index))

    # test model
    test_cluster_lemmas = [test_lemmas[i] for i in test_cluster_to_data_indices[cluster_type]]
    test_cluster_words = [test_words[i] for i in test_cluster_to_data_indices[cluster_type]]
    test_cluster_feat_dicts = [test_feat_dicts[i] for i in test_cluster_to_data_indices[cluster_type]]

    predictions = predict(trained_model, decoder_rnn, encoder_frnn, encoder_rrnn, alphabet_index,
                          inverse_alphabet_index, feat_index, feature_types, test_cluster_lemmas,
                          test_cluster_feat_dicts)

    evaluate_predictions(predictions, test_cluster_lemmas, test_cluster_feat_dicts, test_cluster_words,
                         feature_types, True)
    return trained_model

def build_model(alphabet, feature_alphabet, feature_types, input_dim, hidden_dim, feat_input_dim, layers):
    print 'creating model...'

    model = Model()

    # character embeddings
    model.add_lookup_parameters("char_lookup", (len(alphabet), input_dim))

    # feature embeddings
    # TODO: add another input dim for features?
    model.add_lookup_parameters("feat_lookup", (len(feature_alphabet), feat_input_dim))

    # used in softmax output
    model.add_parameters("R", (len(alphabet), hidden_dim))
    model.add_parameters("bias", len(alphabet))

    # rnn's
    encoder_frnn = LSTMBuilder(layers, input_dim, hidden_dim, model)
    encoder_rrnn = LSTMBuilder(layers, input_dim, hidden_dim, model)

    # TODO: inspect carefully, as dims may be sub-optimal in some cases (many feature types?)
    # 2 * HIDDEN_DIM + 2 * INPUT_DIM + len(feats) * FEAT_INPUT_DIM, as it gets a concatenation of frnn, rrnn
    # (both of HIDDEN_DIM size), previous output char, current lemma char (of INPUT_DIM size) and feats * FEAT_INPUT_DIM
    decoder_rnn = LSTMBuilder(layers, 2 * hidden_dim + 2 * input_dim + len(feature_types) * feat_input_dim, hidden_dim,
                              model)

    print 'finished creating model'

    return model, encoder_frnn, encoder_rrnn, decoder_rnn


def save_pycnn_model(model, results_file_path):
    tmp_model_path = results_file_path + '_bestmodel.txt'
    print 'saving to ' + tmp_model_path
    model.save(tmp_model_path)
    print 'saved to {0}'.format(tmp_model_path)


def evaluate_predictions(predictions, lemmas, feature_dicts, words, feature_types, print_res=False):
    if print_res:
        print 'evaluating model...'

    test_data = zip(lemmas, feature_dicts, words)
    c = 0
    for i, (lemma, feat_dict, word) in enumerate(test_data):
        joint_index = lemma + ':' + common.get_morph_string(feat_dict, feature_types)
        if predictions[joint_index] == word:
            c += 1
            sign = 'V'
        else:
            sign = 'X'
        if print_res:
            print 'lemma: ' + lemma + ' gold: ' + word + ' prediction: ' + predictions[joint_index] + ' ' + sign
    accuracy = float(c) / len(predictions)

    if print_res:
        print 'finished evaluating model. accuracy: ' + str(c) + '/' + str(len(predictions)) + '=' + str(accuracy) + \
              '\n\n'

    return len(predictions), accuracy


def train_model(model, encoder_frnn, encoder_rrnn, decoder_rnn, train_words, train_lemmas, train_feat_dicts,
                dev_words, dev_lemmas, dev_feat_dicts, alphabet_index, inverse_alphabet_index, feat_index,
                feature_types, epochs, optimization, results_file_path):
    print 'training...'

    np.random.seed(17)
    random.seed(17)

    if optimization == 'ADAM':
        trainer = AdamTrainer(model, lam=REGULARIZATION, alpha=LEARNING_RATE, beta_1=0.9, beta_2=0.999, eps=1e-8)
    elif optimization == 'MOMENTUM':
        trainer = MomentumSGDTrainer(model)
    elif optimization == 'SGD':
        trainer = SimpleSGDTrainer(model)
    elif optimization == 'ADAGRAD':
        trainer = AdagradTrainer(model)
    else:
        trainer = SimpleSGDTrainer(model)

    total_loss = 0
    best_avg_dev_loss = 999
    best_dev_accuracy = -1
    best_train_accuracy = -1
    patience = 0
    train_len = len(train_words)
    epochs_x = []
    train_loss_y = []
    dev_loss_y = []
    train_accuracy_y = []
    dev_accuracy_y = []

    # progress bar init
    widgets = [progressbar.Bar('>'), ' ', progressbar.ETA()]
    train_progress_bar = progressbar.ProgressBar(widgets=widgets, maxval=epochs).start()
    avg_loss = -1

    for e in xrange(epochs):

        # randomize the training set
        indices = range(train_len)
        random.shuffle(indices)
        train_set = zip(train_lemmas, train_feat_dicts, train_words)
        train_set = [train_set[i] for i in indices]

        # compute loss for each example and update
        for i, example in enumerate(train_set):
            lemma, feats, word = example
            loss = one_word_loss(model, encoder_frnn, encoder_rrnn, decoder_rnn, lemma, feats, word, alphabet_index,
                                 feat_index, feature_types)
            loss_value = loss.value()
            total_loss += loss_value
            loss.backward()
            trainer.update()
            if i > 0:
                # print 'avg. loss at ' + str(i) + ': ' + str(total_loss / float(i + e*train_len)) + '\n'
                avg_loss = total_loss / float(i + e * train_len)
            else:
                avg_loss = total_loss

        # TODO: handle when no dev set is available - do best on train set...
        if EARLY_STOPPING:

            if len(dev_lemmas) > 0:

                # get train accuracy
                train_predictions = predict(model, decoder_rnn, encoder_frnn, encoder_rrnn, alphabet_index,
                                            inverse_alphabet_index, feat_index, feature_types, train_lemmas,
                                            train_feat_dicts)

                train_accuracy = evaluate_predictions(train_predictions, train_lemmas, train_feat_dicts, train_words,
                                                      feature_types, False)[1]

                if train_accuracy > best_train_accuracy:
                    best_train_accuracy = train_accuracy

                # get dev accuracy
                dev_predictions = predict(model, decoder_rnn, encoder_frnn, encoder_rrnn, alphabet_index,
                                          inverse_alphabet_index, feat_index, feature_types, dev_lemmas, dev_feat_dicts)

                # get dev accuracy
                dev_accuracy = evaluate_predictions(dev_predictions, dev_lemmas, dev_feat_dicts, dev_words,
                                                    feature_types, False)[1]

                if dev_accuracy > best_dev_accuracy:
                    best_dev_accuracy = dev_accuracy

                    # save best model to disk
                    save_pycnn_model(model, results_file_path)
                    print 'saved new best model'
                    patience = 0
                else:
                    patience += 1

                # found "perfect" model
                if dev_accuracy == 1:
                    train_progress_bar.finish()
                    # plt.cla()
                    return model

                # get dev loss
                total_dev_loss = 0
                dev_data = zip(dev_lemmas, dev_feat_dicts, dev_words)
                for lemma, feats, word in dev_data:
                    total_dev_loss += one_word_loss(model, encoder_frnn, encoder_rrnn, decoder_rnn, lemma, feats, word,
                                                    alphabet_index, feat_index, feature_types).value()

                avg_dev_loss = total_dev_loss / float(len(dev_lemmas))
                if avg_dev_loss < best_avg_dev_loss:
                    best_avg_dev_loss = avg_dev_loss

                print 'epoch: {0} train loss: {1:.2f} dev loss: {2:.2f} dev accuracy: {3:.2f} train accuracy = {4:.2f} \
 best dev accuracy {5:.2f} best train accuracy: {6:.2f} patience = {7}'.format(e, avg_loss, avg_dev_loss, dev_accuracy,
                                                                               train_accuracy, best_dev_accuracy,
                                                                               best_train_accuracy, patience)

                if patience == MAX_PATIENCE:
                    print 'out of patience after {0} epochs'.format(str(e))
                    # TODO: would like to return best model but pycnn has a bug with save and load. Maybe copy via code?
                    # return best_model[0]
                    train_progress_bar.finish()
                    # plt.cla()
                    return model

                # update lists for plotting
                train_accuracy_y.append(train_accuracy)
                epochs_x.append(e)
                train_loss_y.append(avg_loss)
                dev_loss_y.append(avg_dev_loss)
                dev_accuracy_y.append(dev_accuracy)
            else:
                print 'no dev set for early stopping, running all epochs'

        # finished epoch
        train_progress_bar.update(e)
        # with plt.style.context('fivethirtyeight'):
        #     p1, = plt.plot(epochs_x, dev_loss_y, label='dev loss')
        #     p2, = plt.plot(epochs_x, train_loss_y, label='train loss')
        #     p3, = plt.plot(epochs_x, dev_accuracy_y, label='dev acc.')
        #     p4, = plt.plot(epochs_x, train_accuracy_y, label='train acc.')
        #     plt.legend(loc='upper left', handles=[p1, p2, p3, p4])
        #     plt.savefig(results_file_path + '_learning_curves.png')
    train_progress_bar.finish()
    # plt.cla()
    print 'finished training. average loss: ' + str(avg_loss)
    return model


# noinspection PyPep8Naming
def one_word_loss(model, encoder_frnn, encoder_rrnn, decoder_rnn, lemma, feats, word, alphabet_index, feat_index,
                  feature_types):
    renew_cg()

    # read the parameters
    char_lookup = model["char_lookup"]
    feat_lookup = model["feat_lookup"]
    R = parameter(model["R"])
    bias = parameter(model["bias"])

    # convert characters to matching embeddings, if UNK handle properly
    lemma = BEGIN_WORD + lemma + END_WORD
    lemma_char_vecs = []
    for char in lemma:
        try:
            lemma_char_vecs.append(char_lookup[alphabet_index[char]])
        except KeyError:
            # handle UNK or dropout
            lemma_char_vecs.append(char_lookup[alphabet_index[UNK]])

    # convert features to matching embeddings, if UNK handle properly
    feat_vecs = []
    for feat in sorted(feature_types):
        # TODO: is it OK to use same UNK for all feature types? and for unseen feats as well?
        # if this feature has a value, take it from the lookup. otherwise use UNK
        if feat in feats:
            feat_str = feat + ':' + feats[feat]
            try:
                feat_vecs.append(feat_lookup[feat_index[feat_str]])
            except KeyError:
                # handle UNK or dropout
                feat_vecs.append(feat_lookup[feat_index[UNK_FEAT]])
        else:
            feat_vecs.append(feat_lookup[feat_index[UNK_FEAT]])
    feats_input = concatenate(feat_vecs)

    # bilstm forward pass
    s_0 = encoder_frnn.initial_state()
    s = s_0
    for c in lemma_char_vecs:
        s = s.add_input(c)
    encoder_frnn_h = s.h()

    # bilstm backward pass
    s_0 = encoder_rrnn.initial_state()
    s = s_0
    for c in reversed(lemma_char_vecs):
        s = s.add_input(c)
    encoder_rrnn_h = s.h()

    # concatenate BILSTM final hidden states
    if len(encoder_rrnn_h) == 1 and len(encoder_frnn_h) == 1:
        encoded = concatenate([encoder_frnn_h[0], encoder_rrnn_h[0]])
    else:
        # if there's more than one hidden layer in the rnn's, take the last one
        encoded = concatenate([encoder_frnn_h[-1], encoder_rrnn_h[-1]])

    # initialize the decoder rnn
    s_0 = decoder_rnn.initial_state()
    s = s_0

    # set prev_output_vec for first lstm step as BEGIN_WORD
    # TODO: change this so it'll be possible to use different dims for input and hidden
    prev_output_vec = char_lookup[alphabet_index[BEGIN_WORD]]

    loss = []
    word = BEGIN_WORD + word + END_WORD

    # run the decoder through the sequence and aggregate loss
    for i, word_char in enumerate(word):

        # if the lemma is finished, pad with epsilon chars
        if i < len(lemma):
            lemma_input_char_vec = char_lookup[alphabet_index[lemma[i]]]
        else:
            lemma_input_char_vec = char_lookup[alphabet_index[EPSILON]]

        decoder_input = concatenate([encoded, prev_output_vec, lemma_input_char_vec, feats_input])
        s = s.add_input(decoder_input)
        decoder_rnn_output = s.output()
        probs = softmax(R * decoder_rnn_output + bias)
        loss.append(-log(pick(probs, alphabet_index[word_char])))

        # prepare for the next iteration
        prev_output_vec = decoder_rnn_output

    # TODO: maybe here a "special" loss function is appropriate?
    # loss = esum(loss)
    loss = average(loss)

    return loss


# noinspection PyPep8Naming
def predict_inflection(model, encoder_frnn, encoder_rrnn, decoder_rnn, lemma, feat_dict, alphabet_index,
                       inverse_alphabet_index, feat_index, feature_types):
    renew_cg()

    # read the parameters
    char_lookup = model["char_lookup"]
    feat_lookup = model["feat_lookup"]
    R = parameter(model["R"])
    bias = parameter(model["bias"])

    # convert characters to matching embeddings, if UNK handle properly
    lemma = BEGIN_WORD + lemma + END_WORD
    lemma_char_vecs = []
    for char in lemma:
        try:
            lemma_char_vecs.append(char_lookup[alphabet_index[char]])
        except KeyError:
            # handle UNK or dropout
            lemma_char_vecs.append(char_lookup[alphabet_index[UNK]])

    # convert features to matching embeddings, if UNK handle properly
    feat_vecs = []
    for feat in sorted(feature_types):
        # TODO: is it OK to use same UNK for all feature types? and for unseen feats as well?
        # if this feature has a value, take it from the lookup. otherwise use UNK
        if feat in feat_dict:
            feat_str = feat + ':' + feat_dict[feat]
            try:
                feat_vecs.append(feat_lookup[feat_index[feat_str]])
            except KeyError:
                # handle UNK or dropout
                feat_vecs.append(feat_lookup[feat_index[UNK_FEAT]])
        else:
            feat_vecs.append(feat_lookup[feat_index[UNK_FEAT]])
    feats_input = concatenate(feat_vecs)

    # bilstm forward pass
    s_0 = encoder_frnn.initial_state()
    s = s_0
    for c in lemma_char_vecs:
        s = s.add_input(c)
    encoder_frnn_h = s.h()

    # bilstm backward pass
    s_0 = encoder_rrnn.initial_state()
    s = s_0
    for c in reversed(lemma_char_vecs):
        s = s.add_input(c)
    encoder_rrnn_h = s.h()

    # concatenate BILSTM final hidden states
    if len(encoder_rrnn_h) == 1 and len(encoder_frnn_h) == 1:
        encoded = concatenate([encoder_frnn_h[0], encoder_rrnn_h[0]])
    else:
        # if there's more than one hidden layer in the rnn's, take the last one
        encoded = concatenate([encoder_frnn_h[-1], encoder_rrnn_h[-1]])

    # initialize the decoder rnn
    s_0 = decoder_rnn.initial_state()
    s = s_0

    # set prev_output_vec for first lstm step as BEGIN_WORD
    prev_output_vec = char_lookup[alphabet_index[BEGIN_WORD]]
    i = 0
    predicted = ''

    # run the decoder through the sequence and predict characters
    while i < MAX_PREDICTION_LEN:

        # if the lemma is finished or unknown character, pad with epsilon chars
        if i < len(lemma) and lemma[i] in alphabet_index:
            lemma_input_char_vec = char_lookup[alphabet_index[lemma[i]]]
        else:
            lemma_input_char_vec = char_lookup[alphabet_index[EPSILON]]

        # prepare input vector and perform LSTM step
        decoder_input = concatenate([encoded, prev_output_vec, lemma_input_char_vec, feats_input])
        s = s.add_input(decoder_input)

        # compute softmax probs and predict
        decoder_rnn_output = s.output()
        probs = softmax(R * decoder_rnn_output + bias)
        probs = probs.vec_value()
        next_predicted_char_index = common.argmax(probs)
        predicted = predicted + inverse_alphabet_index[next_predicted_char_index]

        # check if reached end of word
        if predicted[-1] == END_WORD:
            break

        # prepare for the next iteration
        # prev_output_vec = char_lookup[next_predicted_char_index]
        prev_output_vec = decoder_rnn_output
        i += 1

    # remove the begin and end word symbols
    return predicted[1:-1]


def predict(model, decoder_rnn, encoder_frnn, encoder_rrnn, alphabet_index, inverse_alphabet_index, feat_index,
            feature_types, lemmas, feature_dicts):
    test_data = zip(lemmas, feature_dicts)
    predictions = {}
    for lemma, feat_dict in test_data:
        predicted_word = predict_inflection(model, encoder_frnn, encoder_rrnn, decoder_rnn, lemma, feat_dict,
                                            alphabet_index, inverse_alphabet_index, feat_index, feature_types)
        joint_index = lemma + ':' + common.get_morph_string(feat_dict, feature_types)
        predictions[joint_index] = predicted_word

    return predictions


if __name__ == '__main__':
    arguments = docopt(__doc__)
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

    # default values
    if arguments['TRAIN_PATH']:
        train_path_param = arguments['TRAIN_PATH']
    else:
        train_path_param = '/Users/roeeaharoni/research_data/sigmorphon2016-master/data/turkish-task1-train'
    if arguments['TEST_PATH']:
        test_path_param = arguments['TEST_PATH']
    else:
        test_path_param = '/Users/roeeaharoni/research_data/sigmorphon2016-master/data/turkish-task1-dev'
    if arguments['RESULTS_PATH']:
        results_file_path_param = arguments['RESULTS_PATH']
    else:
        results_file_path_param = \
            '/Users/roeeaharoni/Dropbox/phd/research/morphology/inflection_generation/results/results_' + st + '.txt'
    if arguments['SIGMORPHON_PATH']:
        sigmorphon_root_dir_param = arguments['SIGMORPHON_PATH'][0]
    else:
        sigmorphon_root_dir_param = '/Users/roeeaharoni/research_data/sigmorphon2016-master/'
    if arguments['--input']:
        input_dim_param = int(arguments['--input'])
    else:
        input_dim_param = INPUT_DIM
    if arguments['--hidden']:
        hidden_dim_param = int(arguments['--hidden'])
    else:
        hidden_dim_param = HIDDEN_DIM
    if arguments['--feat-input']:
        feat_input_dim_param = int(arguments['--feat-input'])
    else:
        feat_input_dim_param = FEAT_INPUT_DIM
    if arguments['--epochs']:
        epochs_param = int(arguments['--epochs'])
    else:
        epochs_param = EPOCHS
    if arguments['--layers']:
        layers_param = int(arguments['--layers'])
    else:
        layers_param = LAYERS
    if arguments['--optimization']:
        optimization_param = arguments['--optimization']
    else:
        optimization_param = OPTIMIZATION

    print arguments

    main(train_path_param, test_path_param, results_file_path_param, sigmorphon_root_dir_param, input_dim_param,
         hidden_dim_param, feat_input_dim_param, epochs_param, layers_param, optimization_param)
