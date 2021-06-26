"""Trains and evaluates a factored-structured model for inflection generation, using the sigmorphon 2016 shared task
data files and evaluation script.

Usage:
  pycnn_factored_structured_inflection.py [--cnn-mem MEM][--input=INPUT] [--hidden=HIDDEN] [--epochs=EPOCHS]
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
  --optimization=OPTIMIZATION   chosen optimization method ADAM/SGD/ADAGRAD/MOMENTUM/ADADELTA
"""

import numpy as np
import random
import prepare_sigmorphon_data
import progressbar
import datetime
import time
import os
import align
import common
from multiprocessing import Pool
from matplotlib import pyplot as plt
from docopt import docopt
from pycnn import *

# default values
INPUT_DIM = 100
HIDDEN_DIM = 100
EPOCHS = 1
LAYERS = 2
CHAR_DROPOUT_PROB = 0
MAX_PREDICTION_LEN = 50
OPTIMIZATION = 'ADAM'
EARLY_STOPPING = True
MAX_PATIENCE = 100
REGULARIZATION = 0.0
LEARNING_RATE = 0.0001  # 0.1

NULL = '%'
UNK = '#'
EPSILON = '*'
BEGIN_WORD = '<'
END_WORD = '>'


# TODO: add numbered epsilons to vocabulary?
# TODO: try to add attention mechanism?
# TODO: try sutskever trick - predict inverse
def main(train_path, test_path, results_file_path, sigmorphon_root_dir, input_dim, hidden_dim, epochs, layers,
         optimization):
    parallelize_training = False
    hyper_params = {'INPUT_DIM': input_dim, 'HIDDEN_DIM': hidden_dim, 'EPOCHS': epochs, 'LAYERS': layers,
                    'CHAR_DROPOUT_PROB': CHAR_DROPOUT_PROB, 'MAX_PREDICTION_LEN': MAX_PREDICTION_LEN,
                    'OPTIMIZATION': optimization, 'PATIENCE': MAX_PATIENCE, 'REGULARIZATION': REGULARIZATION,
                    'LEARNING_RATE': LEARNING_RATE}

    print 'train path = ' + str(train_path)
    print 'test path =' + str(test_path)
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

    # align the words to the inflections, the alignment will later be used by the model
    print 'started aligning'
    train_word_pairs = zip(train_lemmas, train_words)
    test_word_pairs = zip(test_lemmas, test_words)
    align_symbol = '~'

    # train_aligned_pairs = dumb_align(train_word_pairs, align_symbol)
    train_aligned_pairs = mcmc_align(train_word_pairs, align_symbol)

    # TODO: align together?
    test_aligned_pairs = mcmc_align(test_word_pairs, align_symbol)
    # random.shuffle(train_aligned_pairs)
    # for p in train_aligned_pairs[:100]:
    #    generate_template(p)
    print 'finished aligning'

    # factored model: new model per inflection type. create input for each model and then parallelize or run in loop.
    params = []
    for morph_index, morph_type in enumerate(train_morph_to_data_indices):
        params.append([input_dim, hidden_dim, layers, morph_index, morph_type, train_lemmas, train_words, test_lemmas,
                       train_morph_to_data_indices, test_words, test_morph_to_data_indices, alphabet, alphabet_index,
                       inverse_alphabet_index, epochs, optimization, results_file_path, train_aligned_pairs,
                       test_aligned_pairs])

    if parallelize_training:
        p = Pool(4, maxtasksperchild=1)
        p.map(train_morph_model, params)
        print 'finished training all models'
    else:
        for p in params:
            if not check_if_exists(p[-3], p[3]):
                train_morph_model(*p)
            else:
                print 'model ' + str(p[3]) + ' exists, skipping...'

    # evaluate best models
    os.system('python task1_evaluate_best_factored_structured_models.py --cnn-mem 8192 --input={0} --hidden={1} --epochs={2} \
              --layers={3} --optimization={4} {5} {6} {7} {8}'.format(input_dim, hidden_dim, epochs, layers,
                                                                      optimization, train_path, test_path,
                                                                      results_file_path,
                                                                      sigmorphon_root_dir))
    return


def check_if_exists(results_file_path, morph_index):
    path = results_file_path + '_' + str(morph_index) + '_bestmodel.txt'
    return os.path.isfile(path)


def train_morph_model(input_dim, hidden_dim, layers, morph_index, morph_type, train_lemmas, train_words, test_lemmas,
                      train_morph_to_data_indices, test_words, test_morph_to_data_indices, alphabet, alphabet_index,
                      inverse_alphabet_index, epochs, optimization, results_file_path, train_aligned_pairs,
                      test_aligned_pairs):
    # get the inflection-specific data
    train_morph_words = [train_words[i] for i in train_morph_to_data_indices[morph_type]]
    train_morph_lemmas = [train_lemmas[i] for i in train_morph_to_data_indices[morph_type]]
    train_morph_alignments = [train_aligned_pairs[i] for i in train_morph_to_data_indices[morph_type]]
    if len(train_morph_words) < 1:
        print 'only ' + str(len(train_morph_words)) + ' samples for this inflection type. skipping'
        # continue
    else:
        print 'now training model for morph ' + str(morph_index) + '/' + str(len(train_morph_to_data_indices)) + \
              ': ' + morph_type + ' with ' + str(len(train_morph_words)) + ' examples'

    # build model
    initial_model, encoder_frnn, encoder_rrnn, decoder_rnn = build_model(alphabet, input_dim, hidden_dim, layers)

    # TODO: now dev and test are the same - change later when test sets are available
    # get dev lemmas for early stopping
    try:
        dev_morph_lemmas = [test_lemmas[i] for i in test_morph_to_data_indices[morph_type]]
        dev_morph_words = [test_words[i] for i in test_morph_to_data_indices[morph_type]]
        dev_morph_alignments = [test_aligned_pairs[i] for i in test_morph_to_data_indices[morph_type]]
    except KeyError:
        dev_morph_lemmas = []
        dev_morph_words = []
        dev_morph_alignments = []
        print 'could not find relevant examples in dev data for morph: ' + morph_type

    # train model
    trained_model = train_model(initial_model, encoder_frnn, encoder_rrnn, decoder_rnn, train_morph_words,
                                train_morph_lemmas, dev_morph_words, dev_morph_lemmas, alphabet_index,
                                inverse_alphabet_index, epochs, optimization, results_file_path, str(morph_index),
                                train_morph_alignments, dev_morph_alignments)

    # evaluate last model on dev
    predicted = predict_templates(trained_model, decoder_rnn, encoder_frnn, encoder_rrnn, alphabet_index,
                                  inverse_alphabet_index, dev_morph_lemmas)
    if len(predicted) > 0:
        evaluate_model(predicted, dev_morph_lemmas, dev_morph_words)
    else:
        print 'no examples in dev set to evaluate'

    return trained_model


def build_model(alphabet, input_dim, hidden_dim, layers):
    print 'creating model...'

    model = Model()

    # character embeddings
    model.add_lookup_parameters("lookup", (len(alphabet), input_dim))

    # used in softmax output
    model.add_parameters("R", (len(alphabet), hidden_dim))
    model.add_parameters("bias", len(alphabet))

    # rnn's
    encoder_frnn = LSTMBuilder(layers, input_dim, hidden_dim, model)
    encoder_rrnn = LSTMBuilder(layers, input_dim, hidden_dim, model)

    # 2 * HIDDEN_DIM + 3 * INPUT_DIM, as it gets a concatenation of frnn, rrnn, previous output char,
    # current lemma char, current index
    decoder_rnn = LSTMBuilder(layers, 2 * hidden_dim + 3 * input_dim, hidden_dim, model)
    print 'finished creating model'

    return model, encoder_frnn, encoder_rrnn, decoder_rnn


def predict_inflection_template(model, encoder_frnn, encoder_rrnn, decoder_rnn, lemma, alphabet_index,
                                inverse_alphabet_index):
    renew_cg()

    # read the parameters
    lookup = model["lookup"]
    # noinspection PyPep8Naming
    R = parameter(model["R"])
    bias = parameter(model["bias"])

    # convert characters to matching embeddings, if UNK handle properly
    lemma = BEGIN_WORD + lemma + END_WORD
    lemma_char_vecs = []
    for char in lemma:
        try:
            lemma_char_vecs.append(lookup[alphabet_index[char]])
        except KeyError:
            # handle UNK
            lemma_char_vecs.append(lookup[alphabet_index[UNK]])

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
        # if there's more than one layer, take the last one
        encoded = concatenate([encoder_frnn_h[-1], encoder_rrnn_h[-1]])

    # initialize the decoder rnn
    s_0 = decoder_rnn.initial_state()
    s = s_0

    # set prev_output_vec for first lstm step as BEGIN_WORD
    prev_output_vec = lookup[alphabet_index[BEGIN_WORD]]
    i = 0
    predicted_template = []

    # run the decoder through the sequence and predict characters
    while i < MAX_PREDICTION_LEN:

        # if the lemma is finished, pad with epsilon chars
        if i < len(lemma):
            lemma_input_char_vec = lookup[alphabet_index[lemma[i]]]
        else:
            lemma_input_char_vec = lookup[alphabet_index[EPSILON]]

        decoder_input = concatenate([encoded, prev_output_vec, lemma_input_char_vec, lookup[alphabet_index[str(i)]]])

        # prepare input vector and perform LSTM step
        # decoder_input = concatenate([encoded, prev_output_vec])
        s = s.add_input(decoder_input)

        # compute softmax probs and predict
        decoder_rnn_output = s.output()
        probs = softmax(R * decoder_rnn_output + bias)
        probs = probs.vec_value()
        next_char_index = common.argmax(probs)
        predicted_template.append(inverse_alphabet_index[next_char_index])

        # check if reached end of word
        if predicted_template[-1] == END_WORD:
            break

        # prepare for the next iteration
        # prev_output_vec = lookup[next_char_index]
        prev_output_vec = decoder_rnn_output
        i += 1

    # remove the begin and end word symbols
    return predicted_template[0:-1]


def predict_templates(model, decoder_rnn, encoder_frnn, encoder_rrnn, alphabet_index, inverse_alphabet_index, lemmas):
    predictions = {}
    for i, lemma in enumerate(lemmas):
        predicted_template = predict_inflection_template(model, encoder_frnn, encoder_rrnn, decoder_rnn, lemma,
                                                         alphabet_index, inverse_alphabet_index)
        predictions[lemma] = predicted_template

    return predictions


def instantiate_template(template, lemma):
    word = ''
    for t in template:
        if represents_int(t):
            # noinspection PyBroadException
            try:
                word = word + lemma[int(t)]
            except:
                continue
        else:
            word = word + t

    # print 'instantiating'
    # print template
    # print lemma
    # print word
    return word


def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def evaluate_model(predicted_templates, lemmas, words, print_results=True):
    if print_results:
        print 'evaluating model...'

    # TODO: 2 possible approaches: one - predict template, instantiate, check if equal to word
    # TODO: two - predict template, generate template using the correct word, check if templates are equal
    # TODO: for now, go with one, maybe try two later
    c = 0
    for i, lemma in enumerate(lemmas):

        # will not work for joint! assumes every lemma is distinct in the model/cluster
        predicted_template = predicted_templates[lemma]
        predicted_word = instantiate_template(predicted_template, lemma)
        if predicted_word == words[i]:
            c += 1
            sign = 'V'
        else:
            sign = 'X'
        if print_results:
            print 'lemma: ' + lemma + ' gold: ' + words[i] + ' template:' + ''.join(predicted_template) \
                  + ' prediction: ' + predicted_word + ' ' + sign
    accuracy = float(c) / len(predicted_templates)

    if print_results:
        print 'finished evaluating model. accuracy: ' + str(c) + '/' + str(len(predicted_templates)) + '=' + str(
            accuracy) + \
              '\n\n'

    return len(predicted_templates), accuracy


def train_model(model, encoder_frnn, encoder_rrnn, decoder_rnn, train_morph_words, train_morph_lemmas, dev_morph_words,
                dev_morph_lemmas, alphabet_index, inverse_alphabet_index, epochs, optimization, results_file_path,
                morph_index, train_aligned_pairs, dev_aligned_pairs):
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
    train_len = len(train_morph_words)
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
        train_morph_lemmas = [train_morph_lemmas[i] for i in indices]
        train_morph_words = [train_morph_words[i] for i in indices]
        train_set = zip(train_morph_lemmas, train_morph_words)
        train_aligned_pairs = [train_aligned_pairs[i] for i in indices]

        # compute loss for each example and update
        for i, example in enumerate(train_set):
            loss = one_word_loss(model, encoder_frnn, encoder_rrnn, decoder_rnn, example[0], example[1], alphabet_index,
                                 train_aligned_pairs[i])
            loss_value = loss.value()
            total_loss += loss_value
            loss.backward()
            trainer.update()
            if i > 0:
                # print 'avg. loss at ' + str(i) + ': ' + str(total_loss / float(i + e*train_len)) + '\n'
                avg_loss = total_loss / float(i + e * train_len)
            else:
                avg_loss = total_loss

        if EARLY_STOPPING:

            # get train accuracy
            train_predictions = predict_templates(model, decoder_rnn, encoder_frnn, encoder_rrnn, alphabet_index,
                                                  inverse_alphabet_index, train_morph_lemmas)
            print 'train:'
            train_accuracy = evaluate_model(train_predictions, train_morph_lemmas, train_morph_words, True)[1]

            if train_accuracy > best_train_accuracy:
                best_train_accuracy = train_accuracy

            dev_accuracy = 0
            avg_dev_loss = 0

            if len(dev_morph_lemmas) > 0:

                # get dev accuracy
                dev_predictions = predict_templates(model, decoder_rnn, encoder_frnn, encoder_rrnn, alphabet_index,
                                                    inverse_alphabet_index, dev_morph_lemmas)
                print 'dev:'
                # get dev accuracy
                dev_accuracy = evaluate_model(dev_predictions, dev_morph_lemmas, dev_morph_words, True)[1]

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
                    plt.cla()
                    return model

                # get dev loss
                total_dev_loss = 0
                for i in xrange(len(dev_morph_lemmas)):
                    total_dev_loss += one_word_loss(model, encoder_frnn, encoder_rrnn, decoder_rnn, dev_morph_lemmas[i],
                                                    dev_morph_words[i], alphabet_index, dev_aligned_pairs[i]).value()

                avg_dev_loss = total_dev_loss / float(len(dev_morph_lemmas))
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
                    plt.cla()
                    return model
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

                print 'epoch: {0} train loss: {1:.2f} train accuracy = {2:.2f} best train accuracy: {3:.2f} \
                patience = {4}'.format(e, avg_loss, train_accuracy, best_train_accuracy, patience)

                # found "perfect" model on train set or patience has reached
                if train_accuracy == 1 or patience == MAX_PATIENCE:
                    train_progress_bar.finish()
                    plt.cla()
                    return model

            # update lists for plotting
            train_accuracy_y.append(train_accuracy)
            epochs_x.append(e)
            train_loss_y.append(avg_loss)
            dev_loss_y.append(avg_dev_loss)
            dev_accuracy_y.append(dev_accuracy)

        # finished epoch
        train_progress_bar.update(e)
        with plt.style.context('fivethirtyeight'):
            p1, = plt.plot(epochs_x, dev_loss_y, label='dev loss')
            p2, = plt.plot(epochs_x, train_loss_y, label='train loss')
            p3, = plt.plot(epochs_x, dev_accuracy_y, label='dev acc.')
            p4, = plt.plot(epochs_x, train_accuracy_y, label='train acc.')
            plt.legend(loc='upper left', handles=[p1, p2, p3, p4])
        plt.savefig(results_file_path + '_' + morph_index + '.png')
    train_progress_bar.finish()
    plt.cla()
    print 'finished training. average loss: ' + str(avg_loss)
    return model


def save_pycnn_model(model, results_file_path, morph_index):
    tmp_model_path = results_file_path + '_' + morph_index + '_bestmodel.txt'
    print 'saving to ' + tmp_model_path
    model.save(tmp_model_path)
    print 'saved to {0}'.format(tmp_model_path)


# noinspection PyPep8Naming
def generate_template(aligned_pair):
    # go through alignment
    # if lemma and inflection are equal, output copy index of lemma
    # if they are not equal - output the inflection char
    template = []
    lemma_index = 0
    aligned_lemma, aligned_word = aligned_pair
    for i in xrange(len(aligned_lemma)):
        # if added prefix, add it to template
        if aligned_lemma[i] == '~':
            template.append(aligned_word[i])
            continue
        # if deleted prefix, promote lemma index and continue
        elif aligned_word[i] == '~':
            lemma_index += 1
            continue
        # if both are not ~, check if equal. if they are, add lemma index. else, add word char.
        elif aligned_lemma[i] == aligned_word[i]:
            template.append(str(lemma_index))
        else:
            template.append(aligned_word[i])

        # promote lemma index
        lemma_index += 1

    # print aligned_lemma
    # print aligned_word
    # print template
    return template


def one_word_loss(model, encoder_frnn, encoder_rrnn, decoder_rnn, lemma, word, alphabet_index, aligned_pair):
    renew_cg()

    # read the parameters
    lookup = model["lookup"]

    # noinspection PyPep8Naming
    R = parameter(model["R"])
    bias = parameter(model["bias"])

    # convert characters to matching embeddings, if UNK handle properly
    template = generate_template(aligned_pair)

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
    lemma_char_vecs = []
    for char in lemma:
        try:
            lemma_char_vecs.append(lookup[alphabet_index[char]])
        except KeyError:
            # handle UNK
            lemma_char_vecs.append(lookup[alphabet_index[UNK]])

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
    prev_output_vec = lookup[alphabet_index[BEGIN_WORD]]
    loss = []

    # TODO: now for the fun part: using the alignments, replace characters in word with lemma indices (if copied),
    # TODO: otherwise leave as is. Then compute loss normally (or by instantiating accordingly in prediction time)
    # TODO: try sutskever flip trick?
    # TODO: attention on the lemma chars could help here?
    # TODO: think about the best heuristic to create a template from the aligned pair with respect to the network loss

    # template.insert(0, BEGIN_WORD)
    template.append(END_WORD)

    # run the decoder through the sequence and aggregate loss
    for i, template_char in enumerate(template):

        # if the lemma is finished, pad with epsilon chars
        if i < len(lemma):
            lemma_input_char_vec = lookup[alphabet_index[lemma[i]]]
        else:
            lemma_input_char_vec = lookup[alphabet_index[EPSILON]]

        decoder_input = concatenate([encoded, prev_output_vec, lemma_input_char_vec, lookup[alphabet_index[str(i)]]])
        # decoder_input = concatenate([encoded, prev_output_vec])
        s = s.add_input(decoder_input)
        decoder_rnn_output = s.output()
        probs = softmax(R * decoder_rnn_output + bias)
        loss.append(-log(pick(probs, alphabet_index[template_char])))

        # prepare for the next iteration
        prev_output_vec = decoder_rnn_output

    # TODO: maybe here a "special" loss function is appropriate?
    # loss = esum(loss)
    loss = average(loss)

    return loss


def dumb_align(wordpairs, align_symbol):
    alignedpairs = []
    for idx, pair in enumerate(wordpairs):
        ins = pair[0]
        outs = pair[1]
        if len(ins) > len(outs):
            outs += align_symbol * (len(ins) - len(outs))
        elif len(outs) > len(ins):
            ins += + align_symbol * (len(outs) - len(ins))
            alignedpairs.append((ins, outs))
    return alignedpairs


def mcmc_align(wordpairs, align_symbol):
    a = align.Aligner(wordpairs, align_symbol=align_symbol)
    return a.alignedpairs


def med_align(wordpairs, align_symbol):
    a = align.Aligner(wordpairs, align_symbol=align_symbol, mode='med')
    return a.alignedpairs


if __name__ == '__main__':
    arguments = docopt(__doc__)
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
