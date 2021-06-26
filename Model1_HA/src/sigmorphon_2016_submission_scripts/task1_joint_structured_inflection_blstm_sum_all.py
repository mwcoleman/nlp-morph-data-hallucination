"""Trains and evaluates a joint-structured model for inflection generation, using the sigmorphon 2016 shared task data
files and evaluation script.

Usage:
  task1_joint_structured_inflection_blstm_feedback_fix.py [--cnn-mem MEM][--input=INPUT] [--hidden=HIDDEN]
  [--feat-input=FEAT] [--epochs=EPOCHS] [--layers=LAYERS] [--optimization=OPTIMIZATION] [--reg=REGULARIZATION]
  [--learning=LEARNING] [--plot] TRAIN_PATH TEST_PATH RESULTS_PATH SIGMORPHON_PATH...

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
  --feat-input=FEAT             feature input vector dimension
  --epochs=EPOCHS               amount of training epochs
  --layers=LAYERS               amount of layers in lstm network
  --optimization=OPTIMIZATION   chosen optimization method ADAM/SGD/ADAGRAD/MOMENTUM/ADADELTA
  --reg=REGULARIZATION          regularization parameter for optimization
  --learning=LEARNING           learning rate parameter for optimization
  --plot                        draw a learning curve plot while training each model
"""

import numpy as np
import random
import prepare_sigmorphon_data
import progressbar
import datetime
import time
import os
import common
from multiprocessing import Pool
from matplotlib import pyplot as plt
from docopt import docopt
from pycnn import *

# default values
INPUT_DIM = 100
FEAT_INPUT_DIM = 20
HIDDEN_DIM = 100
EPOCHS = 1
LAYERS = 2
MAX_PREDICTION_LEN = 50
OPTIMIZATION = 'ADAM'
EARLY_STOPPING = True
MAX_PATIENCE = 100
REGULARIZATION = 0.0
LEARNING_RATE = 0.0001  # 0.1
PARALLELIZE = True

NULL = '%'
UNK = '#'
EPSILON = '*'
BEGIN_WORD = '<'
END_WORD = '>'
UNK_FEAT = '@'


# TODO: add numbered epsilons to vocabulary?
# TODO: add attention mechanism?
# TODO: try sutskever trick - predict inverse
def main(train_path, test_path, results_file_path, sigmorphon_root_dir, input_dim, hidden_dim, feat_input_dim, epochs,
         layers, optimization, regularization, learning_rate, plot):
    if plot:
        parallelize_training = False
        print 'plotting, parallelization is disabled!!!'
    else:
        parallelize_training = PARALLELIZE

    hyper_params = {'INPUT_DIM': input_dim, 'HIDDEN_DIM': hidden_dim, 'FEAT_INPUT_DIM': feat_input_dim,
                    'EPOCHS': epochs, 'LAYERS': layers, 'MAX_PREDICTION_LEN': MAX_PREDICTION_LEN,
                    'OPTIMIZATION': optimization, 'PATIENCE': MAX_PATIENCE, 'REGULARIZATION': regularization,
                    'LEARNING_RATE': learning_rate}

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
    for marker in [str(i) for i in xrange(MAX_PREDICTION_LEN)]:
        alphabet.append(marker)

    # char 2 int
    alphabet_index = dict(zip(alphabet, range(0, len(alphabet))))
    inverse_alphabet_index = {index: char for char, index in alphabet_index.items()}

    # feat 2 int
    feature_alphabet = common.get_feature_alphabet(train_feat_dicts)
    feature_alphabet.append(UNK_FEAT)
    feat_index = dict(zip(feature_alphabet, range(0, len(feature_alphabet))))

    # align the words to the inflections, the alignment will later be used by the model
    print 'started aligning'
    train_word_pairs = zip(train_lemmas, train_words)
    test_word_pairs = zip(test_lemmas, test_words)
    align_symbol = '~'

    # train_aligned_pairs = dumb_align(train_word_pairs, align_symbol)
    train_aligned_pairs = common.mcmc_align(train_word_pairs, align_symbol)

    # TODO: align together?
    test_aligned_pairs = common.mcmc_align(test_word_pairs, align_symbol)
    # random.shuffle(train_aligned_pairs)
    # for p in train_aligned_pairs[:100]:
    #    generate_template(p)
    print 'finished aligning'

    # joint model: cluster the data by POS type (features)
    train_pos_to_data_indices = common.cluster_data_by_pos(train_feat_dicts)
    test_pos_to_data_indices = common.cluster_data_by_pos(test_feat_dicts)
    train_cluster_to_data_indices = train_pos_to_data_indices
    test_cluster_to_data_indices = test_pos_to_data_indices

    # factored model: cluster the data by inflection type (features)
    # train_morph_to_data_indices = common.cluster_data_by_morph_type(train_feat_dicts, feature_types)
    # test_morph_to_data_indices = common.cluster_data_by_morph_type(test_feat_dicts, feature_types)
    # train_cluster_to_data_indices = train_morph_to_data_indices
    # test_cluster_to_data_indices = test_morph_to_data_indices

    # create input for each model and then parallelize or run in loop.
    params = []
    for cluster_index, cluster_type in enumerate(train_cluster_to_data_indices):
        params.append([input_dim, hidden_dim, layers, cluster_index, cluster_type, train_lemmas, train_feat_dicts,
                       train_words, test_lemmas, test_feat_dicts, train_cluster_to_data_indices, test_words,
                       test_cluster_to_data_indices, alphabet, alphabet_index, inverse_alphabet_index, epochs,
                       optimization, results_file_path, train_aligned_pairs, test_aligned_pairs, feat_index,
                       feature_types, feat_input_dim, feature_alphabet, plot])

    if parallelize_training:
        # set maxtasksperchild=1 to free finished processes
        p = Pool(4, maxtasksperchild=1)
        print 'now training {0} models in parallel'.format(len(train_cluster_to_data_indices))
        models = p.map(train_cluster_model_wrapper, params)
    else:
        print 'now training {0} models in loop'.format(len(train_cluster_to_data_indices))
        for p in params:
            trained_model, last_epoch = train_cluster_model(*p)
    print 'finished training all models'

    # evaluate best models
    os.system('python task1_evaluate_best_joint_structured_models_blstm_feed_fix.py --cnn-mem 6096 --input={0} --hidden={1} --feat-input={2} \
                 --epochs={3} --layers={4} --optimization={5} {6} {7} {8} {9}'.format(input_dim, hidden_dim,
                                                                                      feat_input_dim, epochs,
                                                                                      layers, optimization, train_path,
                                                                                      test_path,
                                                                                      results_file_path,
                                                                                      sigmorphon_root_dir))
    return


def train_cluster_model_wrapper(params):
    # from matplotlib import pyplot as plt
    return train_cluster_model(*params)


def train_cluster_model(input_dim, hidden_dim, layers, cluster_index, cluster_type, train_lemmas, train_feat_dicts,
                        train_words, test_lemmas, test_feat_dicts, train_cluster_to_data_indices, test_words,
                        test_cluster_to_data_indices, alphabet, alphabet_index, inverse_alphabet_index, epochs,
                        optimization, results_file_path, train_aligned_pairs, test_aligned_pairs, feat_index,
                        feature_types, feat_input_dim, feature_alphabet, plot):
    # get the inflection-specific data
    train_cluster_words = [train_words[i] for i in train_cluster_to_data_indices[cluster_type]]
    train_cluster_lemmas = [train_lemmas[i] for i in train_cluster_to_data_indices[cluster_type]]
    train_cluster_alignments = [train_aligned_pairs[i] for i in train_cluster_to_data_indices[cluster_type]]
    train_cluster_feat_dicts = [train_feat_dicts[i] for i in train_cluster_to_data_indices[cluster_type]]

    if len(train_cluster_words) < 1:
        print 'only ' + str(len(train_cluster_words)) + ' samples for this inflection type. skipping'
        # continue
    else:
        print 'now training model for cluster ' + str(cluster_index + 1) + '/' + \
              str(len(train_cluster_to_data_indices)) + ': ' + cluster_type + ' with ' + \
              str(len(train_cluster_words)) + ' examples'

    # build model
    initial_model, encoder_frnn, encoder_rrnn, decoder_rnn = build_model(alphabet, input_dim, hidden_dim, layers,
                                                                         feature_types, feat_input_dim,
                                                                         feature_alphabet)

    # TODO: now dev and test are the same - change later when test sets are available
    # get dev lemmas for early stopping
    try:
        dev_cluster_lemmas = [test_lemmas[i] for i in test_cluster_to_data_indices[cluster_type]]
        dev_cluster_words = [test_words[i] for i in test_cluster_to_data_indices[cluster_type]]
        dev_cluster_alignments = [test_aligned_pairs[i] for i in test_cluster_to_data_indices[cluster_type]]
        dev_cluster_feat_dicts = [test_feat_dicts[i] for i in test_cluster_to_data_indices[cluster_type]]

    except KeyError:
        dev_cluster_lemmas = []
        dev_cluster_words = []
        dev_cluster_alignments = []
        dev_cluster_feat_dicts = []
        print 'could not find relevant examples in dev data for cluster: ' + cluster_type

    # train model
    trained_model, last_epoch = train_model(initial_model, encoder_frnn, encoder_rrnn, decoder_rnn, train_cluster_lemmas,
                                train_cluster_feat_dicts, train_cluster_words, dev_cluster_lemmas,
                                dev_cluster_feat_dicts, dev_cluster_words, alphabet_index, inverse_alphabet_index,
                                epochs, optimization, results_file_path, str(cluster_index),
                                train_cluster_alignments, dev_cluster_alignments, feat_index, feature_types, plot)

    # evaluate last model on dev
    predicted_templates = predict_templates(trained_model, decoder_rnn, encoder_frnn, encoder_rrnn, alphabet_index,
                                            inverse_alphabet_index, dev_cluster_lemmas, dev_cluster_feat_dicts,
                                            feat_index,
                                            feature_types)
    if len(predicted_templates) > 0:
        evaluate_model(predicted_templates, dev_cluster_lemmas, dev_cluster_feat_dicts, dev_cluster_words,
                       feature_types, print_results=False)
    else:
        print 'no examples in dev set to evaluate'

    return trained_model, last_epoch


def build_model(alphabet, input_dim, hidden_dim, layers, feature_types, feat_input_dim, feature_alphabet):
    print 'creating model...'

    model = Model()

    # character embeddings
    model.add_lookup_parameters("char_lookup", (len(alphabet), input_dim))

    # feature embeddings
    model.add_lookup_parameters("feat_lookup", (len(feature_alphabet), feat_input_dim))

    # used in softmax output
    model.add_parameters("R", (len(alphabet), hidden_dim))
    model.add_parameters("bias", len(alphabet))

    # rnn's
    encoder_frnn = LSTMBuilder(layers, input_dim, hidden_dim, model)
    encoder_rrnn = LSTMBuilder(layers, input_dim, hidden_dim, model)

    # 2 * HIDDEN_DIM + 3 * INPUT_DIM, as it gets a concatenation of frnn, rrnn, previous output char,
    # current lemma char, current index marker
    decoder_rnn = LSTMBuilder(layers, 2 * hidden_dim + 3 * input_dim + len(feature_types) * feat_input_dim, hidden_dim,
                              model)
    print 'finished creating model'

    return model, encoder_frnn, encoder_rrnn, decoder_rnn


def train_model(model, encoder_frnn, encoder_rrnn, decoder_rnn, train_lemmas, train_feat_dicts, train_words, dev_lemmas,
                dev_feat_dicts, dev_words, alphabet_index, inverse_alphabet_index, epochs, optimization,
                results_file_path, morph_index, train_aligned_pairs, dev_aligned_pairs, feat_index, feature_types, plot):
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
    elif optimization == 'ADADELTA':
        trainer = AdadeltaTrainer(model)
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
        train_set = zip(train_lemmas, train_feat_dicts, train_words, train_aligned_pairs)
        train_set = [train_set[i] for i in indices]

        # compute loss for each example and update
        for i, example in enumerate(train_set):
            lemma, feats, word, alignment = example
            loss = one_word_loss(model, encoder_frnn, encoder_rrnn, decoder_rnn, lemma, feats, word,
                                 alphabet_index, alignment, feat_index, feature_types)
            loss_value = loss.value()
            total_loss += loss_value
            loss.backward()
            trainer.update()
            if i > 0:
                avg_loss = total_loss / float(i + e * train_len)
            else:
                avg_loss = total_loss

        if EARLY_STOPPING:

            # get train accuracy
            train_predictions = predict_templates(model, decoder_rnn, encoder_frnn, encoder_rrnn, alphabet_index,
                                                  inverse_alphabet_index, train_lemmas, train_feat_dicts, feat_index,
                                                  feature_types)
            print 'train:'
            train_accuracy = evaluate_model(train_predictions, train_lemmas, train_feat_dicts, train_words,
                                            feature_types, False)[1]

            if train_accuracy > best_train_accuracy:
                best_train_accuracy = train_accuracy

            dev_accuracy = 0
            avg_dev_loss = 0

            if len(dev_lemmas) > 0:

                # get dev accuracy
                dev_predictions = predict_templates(model, decoder_rnn, encoder_frnn, encoder_rrnn, alphabet_index,
                                                    inverse_alphabet_index, dev_lemmas, dev_feat_dicts, feat_index,
                                                    feature_types)
                print 'dev:'
                # get dev accuracy
                dev_accuracy = evaluate_model(dev_predictions, dev_lemmas, dev_feat_dicts, dev_words, feature_types,
                                              False)[1]

                if dev_accuracy > best_dev_accuracy:
                    best_dev_accuracy = dev_accuracy

                    # save best model to disk
                    save_pycnn_model(model, results_file_path, morph_index)
                    print 'saved new best model'
                    patience = 0
                else:
                    patience += 1

                # found "perfect" model
                if dev_accuracy == 1:
                    train_progress_bar.finish()
                    if plot:
                        plt.cla()
                    return model, e

                # get dev loss
                total_dev_loss = 0
                for i in xrange(len(dev_lemmas)):
                    total_dev_loss += one_word_loss(model, encoder_frnn, encoder_rrnn, decoder_rnn, dev_lemmas[i],
                                                    dev_feat_dicts[i], dev_words[i], alphabet_index,
                                                    dev_aligned_pairs[i], feat_index, feature_types).value()

                avg_dev_loss = total_dev_loss / float(len(dev_lemmas))
                if avg_dev_loss < best_avg_dev_loss:
                    best_avg_dev_loss = avg_dev_loss

                print 'epoch: {0} train loss: {1:.4f} dev loss: {2:.4f} dev accuracy: {3:.4f} train accuracy = {4:.4f} \
 best dev accuracy {5:.4f} best train accuracy: {6:.4f} patience = {7}'.format(e, avg_loss, avg_dev_loss, dev_accuracy,
                                                                               train_accuracy, best_dev_accuracy,
                                                                               best_train_accuracy, patience)

                if patience == MAX_PATIENCE:
                    print 'out of patience after {0} epochs'.format(str(e))
                    # TODO: would like to return best model but pycnn has a bug with save and load. Maybe copy via code?
                    # return best_model[0]
                    train_progress_bar.finish()
                    if plot:
                        plt.cla()
                    return model, e
            else:

                # if no dev set is present, optimize on train set
                print 'no dev set for early stopping, running all epochs until perfectly fitting or patience was \
                reached on the train set'

                if train_accuracy > best_train_accuracy:
                    best_train_accuracy = train_accuracy

                    # save best model to disk
                    save_pycnn_model(model, results_file_path, morph_index)
                    print 'saved new best model'
                    patience = 0
                else:
                    patience += 1

                print 'epoch: {0} train loss: {1:.4f} train accuracy = {2:.4f} best train accuracy: {3:.4f} \
                patience = {4}'.format(e, avg_loss, train_accuracy, best_train_accuracy, patience)

                # found "perfect" model on train set or patience has reached
                if train_accuracy == 1 or patience == MAX_PATIENCE:
                    train_progress_bar.finish()
                    if plot:
                        plt.cla()
                    return model, e

            # update lists for plotting
            train_accuracy_y.append(train_accuracy)
            epochs_x.append(e)
            train_loss_y.append(avg_loss)
            dev_loss_y.append(avg_dev_loss)
            dev_accuracy_y.append(dev_accuracy)

        # finished epoch
        train_progress_bar.update(e)
        if plot:
            with plt.style.context('fivethirtyeight'):
                p1, = plt.plot(epochs_x, dev_loss_y, label='dev loss')
                p2, = plt.plot(epochs_x, train_loss_y, label='train loss')
                p3, = plt.plot(epochs_x, dev_accuracy_y, label='dev acc.')
                p4, = plt.plot(epochs_x, train_accuracy_y, label='train acc.')
                plt.legend(loc='upper left', handles=[p1, p2, p3, p4])
            plt.savefig(results_file_path + '_' + morph_index + '.png')
    train_progress_bar.finish()
    if plot:
        plt.cla()
    print 'finished training. average loss: ' + str(avg_loss)
    return model, e


def save_pycnn_model(model, results_file_path, model_index):
    tmp_model_path = results_file_path + '_' + model_index + '_bestmodel.txt'
    print 'saving to ' + tmp_model_path
    model.save(tmp_model_path)
    print 'saved to {0}'.format(tmp_model_path)


# noinspection PyPep8Naming
def one_word_loss(model, encoder_frnn, encoder_rrnn, decoder_rnn, lemma, feats, word, alphabet_index, aligned_pair,
                  feat_index, feature_types):
    renew_cg()

    # read the parameters
    char_lookup = model["char_lookup"]
    feat_lookup = model["feat_lookup"]
    R = parameter(model["R"])
    bias = parameter(model["bias"])

    # convert characters to matching embeddings, if UNK handle properly
    template = generate_template_from_alignment(aligned_pair)

    # sanity check
    instantiated = instantiate_template(template, lemma)
    if not instantiated == word:
        print 'bad train instantiation:'
        print lemma
        print word
        print template
        print instantiated
        raise Exception()

    lemma = BEGIN_WORD + lemma + END_WORD

    # convert characters to matching embeddings
    lemma_char_vecs = []
    for char in lemma:
        try:
            lemma_char_vecs.append(char_lookup[alphabet_index[char]])
        except KeyError:
            # handle UNK
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

    # BiLSTM forward pass
    s_0 = encoder_frnn.initial_state()
    s = s_0
    frnn_outputs = []
    for c in lemma_char_vecs:
        s = s.add_input(c)
        frnn_outputs.append(s.output())

    # BiLSTM backward pass
    s_0 = encoder_rrnn.initial_state()
    s = s_0
    rrnn_outputs = []
    for c in reversed(lemma_char_vecs):
        s = s.add_input(c)
        rrnn_outputs.append(s.output())

    # BiLTSM outputs
    blstm_outputs = []
    lemma_char_vecs_len = len(lemma_char_vecs)
    for i in xrange(lemma_char_vecs_len):
        blstm_outputs.append(concatenate([frnn_outputs[i], rrnn_outputs[lemma_char_vecs_len - i - 1]]))

    # initialize the decoder rnn
    s_0 = decoder_rnn.initial_state()
    s = s_0

    # set prev_output_vec for first lstm step as BEGIN_WORD
    prev_output_vec = char_lookup[alphabet_index[BEGIN_WORD]]
    loss = []

    # Using the alignments, replace characters in word with lemma indices (if copied), otherwise leave as is.
    # Then compute loss normally (or by instantiating accordingly in prediction time)
    # TODO: try sutskever flip trick?
    # TODO: attention on the lemma chars could help here?
    template.append(END_WORD)

    # run the decoder through the sequence and aggregate loss
    for i, options in enumerate(template):

        # if the lemma is finished, pad with epsilon chars and use last blstm output as encoded
        if i < len(lemma):
            blstm_output = blstm_outputs[i]
            try:
                lemma_input_char_vec = char_lookup[alphabet_index[lemma[i]]]
            except KeyError:
                # handle UNK
                lemma_input_char_vec = char_lookup[alphabet_index[UNK]]
        else:
            lemma_input_char_vec = char_lookup[alphabet_index[EPSILON]]
            blstm_output = blstm_outputs[lemma_char_vecs_len - 1]

        # TODO: check if template index char helps, maybe redundant
        # encoded lemma, previous output (hidden) vector, lemma input char, template index char, features
        decoder_input = concatenate([blstm_output, prev_output_vec, lemma_input_char_vec,
                                     char_lookup[alphabet_index[str(i)]], feats_input])

        s = s.add_input(decoder_input)
        decoder_rnn_output = s.output()
        probs = softmax(R * decoder_rnn_output + bias)

        local_loss = scalarInput(0)
        max_likelihood_output = options[0]
        max_output_loss = -log(pick(probs, alphabet_index[options[0]]))

        for o in options:
            neg_log_likelihood = -log(pick(probs, alphabet_index[o]))
            local_loss += neg_log_likelihood
            if neg_log_likelihood < max_output_loss:
                max_likelihood_output = o
                max_output_loss = neg_log_likelihood
        loss.append(local_loss)

        # prepare for the next iteration - "feedback"
        prev_output_vec = char_lookup[alphabet_index[max_likelihood_output]]

    # TODO: maybe here a "special" loss function is appropriate?
    # loss = esum(loss)
    loss = average(loss)

    return loss


# noinspection PyPep8Naming
def predict_inflection_template(model, encoder_frnn, encoder_rrnn, decoder_rnn, lemma, feats, alphabet_index,
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
            # handle UNK
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

    # BiLSTM forward pass
    s_0 = encoder_frnn.initial_state()
    s = s_0
    frnn_outputs = []
    for c in lemma_char_vecs:
        s = s.add_input(c)
        frnn_outputs.append(s.output())

    # BiLSTM backward pass
    s_0 = encoder_rrnn.initial_state()
    s = s_0
    rrnn_outputs = []
    for c in reversed(lemma_char_vecs):
        s = s.add_input(c)
        rrnn_outputs.append(s.output())

    # BiLTSM outputs
    blstm_outputs = []
    lemma_char_vecs_len = len(lemma_char_vecs)
    for i in xrange(lemma_char_vecs_len):
        blstm_outputs.append(concatenate([frnn_outputs[i], rrnn_outputs[lemma_char_vecs_len - i - 1]]))

    # initialize the decoder rnn
    s_0 = decoder_rnn.initial_state()
    s = s_0

    # set prev_output_vec for first lstm step as BEGIN_WORD
    prev_output_vec = char_lookup[alphabet_index[BEGIN_WORD]]
    i = 0
    predicted_template = []

    # run the decoder through the sequence and predict characters
    while i < MAX_PREDICTION_LEN:

        # if the lemma is finished, pad with epsilon chars
        if i < len(lemma):
            blstm_output = blstm_outputs[i]
            try:
                lemma_input_char_vec = char_lookup[alphabet_index[lemma[i]]]
            except KeyError:
                # handle unseen characters
                lemma_input_char_vec = char_lookup[alphabet_index[UNK]]
        else:
            lemma_input_char_vec = char_lookup[alphabet_index[EPSILON]]
            blstm_output = blstm_outputs[lemma_char_vecs_len - 1]

        decoder_input = concatenate([blstm_output,
                                     prev_output_vec,
                                     lemma_input_char_vec,
                                     char_lookup[alphabet_index[str(i)]],
                                     feats_input])

        # prepare input vector and perform LSTM step
        # decoder_input = concatenate([encoded, prev_output_vec])
        s = s.add_input(decoder_input)

        # compute softmax probs and predict
        decoder_rnn_output = s.output()
        probs = softmax(R * decoder_rnn_output + bias)
        probs = probs.vec_value()
        next_char_index = common.argmax(probs)
        predicted_template.append([inverse_alphabet_index[next_char_index]])

        # check if reached end of word
        if predicted_template[-1] == END_WORD:
            break

        # prepare for the next iteration - "feedback"
        prev_output_vec = char_lookup[next_char_index]
        i += 1

    # remove the end word symbol
    return predicted_template[0:-1]


def predict_templates(model, decoder_rnn, encoder_frnn, encoder_rrnn, alphabet_index, inverse_alphabet_index, lemmas,
                      feats, feat_index, feature_types):
    predictions = {}
    for i, (lemma, feat_dict) in enumerate(zip(lemmas, feats)):
        predicted_template = predict_inflection_template(model, encoder_frnn, encoder_rrnn, decoder_rnn, lemma,
                                                         feat_dict, alphabet_index, inverse_alphabet_index, feat_index,
                                                         feature_types)

        joint_index = lemma + ':' + common.get_morph_string(feat_dict, feature_types)
        predictions[joint_index] = predicted_template

    return predictions


def instantiate_template(template, lemma):
    word = ''
    for place in template:
        for t in place:
            if represents_int(t):
                try:
                    word = word + lemma[int(t)]
                    break
                except IndexError:
                    continue
            else:
                word = word + t
                break

    return word


def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def evaluate_model(predicted_templates, lemmas, feature_dicts, words, feature_types, print_results=True):
    if print_results:
        print 'evaluating model...'

    # 2 possible approaches: one - predict template, instantiate, check if equal to word
    # TODO: other option - predict template, generate template using the correct word, check if templates are equal
    test_data = zip(lemmas, feature_dicts, words)
    c = 0
    for i, (lemma, feat_dict, word) in enumerate(test_data):
        joint_index = lemma + ':' + common.get_morph_string(feat_dict, feature_types)
        predicted_word = instantiate_template(predicted_templates[joint_index], lemma)
        if predicted_word == word:
            c += 1
            sign = 'V'
        else:
            sign = 'X'
        if print_results:
            print 'lemma: ' + lemma + ' gold: ' + words[i] + ' template:' + ''.join(predicted_templates[joint_index]) \
                  + ' prediction: ' + predicted_word + ' ' + sign

    accuracy = float(c) / len(predicted_templates)

    if print_results:
        print 'finished evaluating model. accuracy: ' + str(c) + '/' + str(len(predicted_templates)) + '=' + \
              str(accuracy) + '\n\n'

    return len(predicted_templates), accuracy


# noinspection PyPep8Naming
def generate_template_from_alignment(aligned_pair):
    # go through alignment
    # if lemma and inflection are equal, output copy index of lemma
    # if they are not equal - output the inflection char
    template = []
    lemma_index = 0
    aligned_lemma, aligned_word = aligned_pair
    for i in xrange(len(aligned_word)):

        # first see if the chars may be copied from somewhere in the lemma
        lemma = aligned_lemma.replace('~', '')
        possible_outputs = [str(k) for k,l in enumerate(lemma) if l==aligned_word[i]]

        # if added prefix, add it to template
        if aligned_lemma[i] == '~':
            possible_outputs.append(aligned_word[i])
            template.append(possible_outputs)
            continue
        # if deleted prefix, promote lemma index and continue
        elif aligned_word[i] == '~':
            lemma_index += 1
            continue
        # if both are not ~, enable copy from anywhere or word char
        else:
            possible_outputs.append(aligned_word[i])
            template.append(possible_outputs)

        # promote lemma index
        lemma_index += 1

    return template


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
    if arguments['--reg']:
        regularization_param = float(arguments['--reg'])
    else:
        regularization_param = REGULARIZATION
    if arguments['--learning']:
        learning_rate_param = float(arguments['--learning'])
    else:
        learning_rate_param = LEARNING_RATE
    if arguments['--plot']:
        plot_param = True
    else:
        plot_param = False

    print arguments

    main(train_path_param, test_path_param, results_file_path_param, sigmorphon_root_dir_param, input_dim_param,
         hidden_dim_param, feat_input_dim_param, epochs_param, layers_param, optimization_param, regularization_param,
         learning_rate_param, plot_param)
