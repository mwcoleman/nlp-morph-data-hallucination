"""Trains and evaluates a factored-model for inflection generation, using the sigmorphon 2016 shared task data files
and evaluation script.

Usage:
  pycnn_factored_inflection.py [--cnn-mem MEM][--input=INPUT] [--hidden=HIDDEN] [--epochs=EPOCHS] [--layers=LAYERS]
  [--optimization=OPTIMIZATION] TRAIN_PATH TEST_PATH RESULTS_PATH SIGMORPHON_PATH...

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
import codecs
import os
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
MAX_PREDICTION_LEN = 50
OPTIMIZATION = 'ADAM'
EARLY_STOPPING = True
MAX_PATIENCE = 100
REGULARIZATION = 0.0001
LEARNING_RATE = 0.001  # 0.1

NULL = '%'
UNK = '#'
EPSILON = '*'
BEGIN_WORD = '<'
END_WORD = '>'


# TODO: run this (and baseline - done) on all available languages - in progress
# TODO: print comparison with gold standard in the end of every iteration
# TODO: try running on GPU
# TODO: consider different begin, end chars for lemma and word
# TODO: consider different lookup table for lemma and word
# TODO: implement (character?) dropout
# TODO: make different input and hidden dims work (problem with first input in decoder)
# TODO: try different learning algorithms (ADAGRAD, ADAM...)
# TODO: refactor so less code repetition in train, predict (both have bilstm etc.)
# TODO: think how to give more emphasis on suffix generalization/learning
# TODO: handle unk chars better


def main(train_path, test_path, results_file_path, sigmorphon_root_dir, input_dim, hidden_dim, epochs, layers,
         optimization):
    parallelize_training = True
    hyper_params = {'INPUT_DIM': input_dim, 'HIDDEN_DIM': hidden_dim, 'EPOCHS': epochs, 'LAYERS': layers,
                    'MAX_PREDICTION_LEN': MAX_PREDICTION_LEN, 'OPTIMIZATION': optimization, 'PATIENCE': MAX_PATIENCE,
                    'REGULARIZATION': REGULARIZATION, 'LEARNING_RATE': LEARNING_RATE}

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

    # char 2 int
    alphabet_index = dict(zip(alphabet, range(0, len(alphabet))))
    inverse_alphabet_index = {index: char for char, index in alphabet_index.items()}

    # cluster the data by inflection type (features)
    train_morph_to_data_indices = common.cluster_data_by_morph_type(train_feat_dicts, feats)
    test_morph_to_data_indices = common.cluster_data_by_morph_type(test_feat_dicts, feats)

    # factored model: new model per inflection type
    params = []
    for morph_index, morph_type in enumerate(train_morph_to_data_indices):
        params.append([input_dim, hidden_dim, layers, morph_index, morph_type, train_lemmas, train_words, test_lemmas,
                       train_morph_to_data_indices, test_words, test_morph_to_data_indices, alphabet, alphabet_index,
                       inverse_alphabet_index, epochs, optimization, results_file_path])

    if parallelize_training:
        p = Pool(4, maxtasksperchild=1)
        p.map(train_morph_model_wrapper, params)
        print 'finished training all models'
    else:
        for p in params:
            train_morph_model(*p)

    # evaluate best models
    os.system('python task1_evaluate_best_factored_models.py --cnn-mem 4096 --input={0} --hidden={1} --epochs={2} --layers={3}\
  --optimization={4} {5} {6} {7} {8}'.format(input_dim, hidden_dim, epochs, layers, optimization, train_path, test_path,
                                             results_file_path, sigmorphon_root_dir))
    return

def train_morph_model_wrapper(params):
    # from matplotlib import pyplot as plt
    return train_morph_model(*params)

def train_morph_model(input_dim, hidden_dim, layers, morph_index, morph_type, train_lemmas, train_words, test_lemmas,
                      train_morph_to_data_indices, test_words, test_morph_to_data_indices, alphabet, alphabet_index,
                      inverse_alphabet_index, epochs, optimization, results_file_path):
    # get the inflection-specific data
    train_morph_words = [train_words[i] for i in train_morph_to_data_indices[morph_type]]
    train_morph_lemmas = [train_lemmas[i] for i in train_morph_to_data_indices[morph_type]]
    if len(train_morph_words) < 1:
        print 'only ' + str(len(train_morph_words)) + ' samples for this inflection type. skipping'
        # continue
    else:
        print 'now training model for morph ' + str(morph_index) + '/' + str(len(train_morph_to_data_indices)) + \
              ': ' + morph_type + ' with ' + str(len(train_morph_words)) + ' examples'

    # TODO: remove this later, it's a temporary fix to avoid re-training already trained models
    tmp_model_path = results_file_path + '_' + morph_index + '_bestmodel.txt'
    if os.path.isfile(tmp_model_path):
        print '\n\n\n**********************************SKIPPING {}*******************************\n\n\n'.format(
            tmp_model_path)
        return


    # build model
    initial_model, encoder_frnn, encoder_rrnn, decoder_rnn = build_model(alphabet, input_dim, hidden_dim, layers)

    # TODO: now dev and test are the same - change later when test sets are available
    # get dev lemmas for early stopping
    try:
        dev_morph_lemmas = [test_lemmas[i] for i in test_morph_to_data_indices[morph_type]]
        dev_morph_words = [test_words[i] for i in test_morph_to_data_indices[morph_type]]
    except KeyError:
        dev_morph_lemmas = []
        dev_morph_words = []
        print 'could not find relevant examples in dev data for morph: ' + morph_type

    # train model
    trained_model = train_model(initial_model, encoder_frnn, encoder_rrnn, decoder_rnn, train_morph_words,
                                train_morph_lemmas, dev_morph_words, dev_morph_lemmas, alphabet_index,
                                inverse_alphabet_index, epochs, optimization, results_file_path, str(morph_index))

    # evaluate last model on dev
    if len(dev_morph_lemmas) > 0:
        predicted = predict(trained_model, decoder_rnn, encoder_frnn, encoder_rrnn, alphabet_index,
                        inverse_alphabet_index, dev_morph_lemmas, dev_morph_words)
        evaluate_model(predicted, zip(dev_morph_lemmas, dev_morph_words))

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

    # 2 * HIDDEN_DIM + 2 * INPUT_DIM, as it gets a concatenation of frnn, rrnn, previous output char, current lemma char
    decoder_rnn = LSTMBuilder(layers, 2 * hidden_dim + 2 * input_dim, hidden_dim, model)

    print 'finished creating model'

    return model, encoder_frnn, encoder_rrnn, decoder_rnn


# noinspection PyPep8Naming
def one_word_loss(model, encoder_frnn, encoder_rrnn, decoder_rnn, lemma, word, alphabet_index):
    renew_cg()

    # read the parameters
    lookup = model["lookup"]
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
        # if there's more than one hidden layer in the rnn's, take the last one
        encoded = concatenate([encoder_frnn_h[-1], encoder_rrnn_h[-1]])

    # initialize the decoder rnn
    s_0 = decoder_rnn.initial_state()
    s = s_0

    # set prev_output_vec for first lstm step as BEGIN_WORD
    # TODO: change this so it'll be possible to use different dims for input and hidden
    prev_output_vec = lookup[alphabet_index[BEGIN_WORD]]
    loss = []
    word = BEGIN_WORD + word + END_WORD

    # run the decoder through the sequence and aggregate loss
    for i, word_char in enumerate(word):

        # if the lemma is finished, pad with epsilon chars
        if i < len(lemma):
            lemma_input_char_vec = lookup[alphabet_index[lemma[i]]]
        else:
            lemma_input_char_vec = lookup[alphabet_index[EPSILON]]

        decoder_input = concatenate([encoded, prev_output_vec, lemma_input_char_vec])
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


def save_pycnn_model(model, results_file_path, morph_index):
    tmp_model_path = results_file_path + '_' + morph_index + '_bestmodel.txt'
    print 'saving to ' + tmp_model_path
    model.save(tmp_model_path)
    print 'saved to {0}'.format(tmp_model_path)


def train_model(model, encoder_frnn, encoder_rrnn, decoder_rnn, train_morph_words, train_morph_lemmas, dev_morph_words,
                dev_morph_lemmas, alphabet_index, inverse_alphabet_index, epochs, optimization, results_file_path,
                morph_index):
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
        train_set = zip(train_morph_lemmas, train_morph_words)
        train_set = [train_set[i] for i in indices]

        # compute loss for each example and update
        for i, example in enumerate(train_set):
            loss = one_word_loss(model, encoder_frnn, encoder_rrnn, decoder_rnn, example[0], example[1], alphabet_index)
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
            train_predictions = predict(model, decoder_rnn, encoder_frnn, encoder_rrnn, alphabet_index,
                                        inverse_alphabet_index, train_morph_lemmas, train_morph_words)
            train_accuracy = evaluate_model(train_predictions, train_set, False)[1]

            if train_accuracy > best_train_accuracy:
                best_train_accuracy = train_accuracy

            dev_accuracy = 0
            avg_dev_loss = 0

            if len(dev_morph_lemmas) > 0:

                # get dev accuracy
                dev_predictions = predict(model, decoder_rnn, encoder_frnn, encoder_rrnn, alphabet_index,
                                          inverse_alphabet_index, dev_morph_lemmas, dev_morph_words)

                # get dev accuracy
                dev_data = zip(dev_morph_lemmas, dev_morph_words)
                dev_accuracy = evaluate_model(dev_predictions, dev_data, False)[1]

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
                    # plt.cla()
                    return model

                # get dev loss
                total_dev_loss = 0
                for word, lemma in dev_data:
                    total_dev_loss += one_word_loss(model, encoder_frnn, encoder_rrnn, decoder_rnn, lemma, word,
                                                    alphabet_index).value()
                if len(dev_morph_lemmas) > 0:
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
                    # plt.cla()
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
                    # plt.cla()
                    return model

            # update lists for plotting
            train_accuracy_y.append(train_accuracy)
            epochs_x.append(e)
            train_loss_y.append(avg_loss)
            dev_loss_y.append(avg_dev_loss)
            dev_accuracy_y.append(dev_accuracy)

        # finished epoch
        train_progress_bar.update(e)
        # with plt.style.context('fivethirtyeight'):
        #     p1, = plt.plot(epochs_x, dev_loss_y, label='dev loss')
        #     p2, = plt.plot(epochs_x, train_loss_y, label='train loss')
        #     p3, = plt.plot(epochs_x, dev_accuracy_y, label='dev acc.')
        #     p4, = plt.plot(epochs_x, train_accuracy_y, label='train acc.')
        #     plt.legend(loc='upper left', handles=[p1, p2, p3, p4])
        # plt.savefig(results_file_path + '_' + morph_index + '.png')
    train_progress_bar.finish()
    # plt.cla()
    print 'finished training. average loss: ' + str(avg_loss)
    return model


def predict(model, decoder_rnn, encoder_frnn, encoder_rrnn, alphabet_index, inverse_alphabet_index, lemmas,
            words):
    test_data = zip(lemmas, words)
    predictions = {}
    for lemma, word in test_data:
        predicted_word = predict_inflection(model, encoder_frnn, encoder_rrnn, decoder_rnn, lemma, alphabet_index,
                                            inverse_alphabet_index)
        predictions[lemma] = predicted_word

    return predictions


# noinspection PyPep8Naming
def predict_inflection(model, encoder_frnn, encoder_rrnn, decoder_rnn, lemma, alphabet_index, inverse_alphabet_index):
    renew_cg()

    # read the parameters
    lookup = model["lookup"]
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
    predicted = ''

    # run the decoder through the sequence and predict characters
    while i < MAX_PREDICTION_LEN:

        # if the lemma is finished or unknown character, pad with epsilon chars
        if i < len(lemma) and lemma[i] in alphabet_index:
            lemma_input_char_vec = lookup[alphabet_index[lemma[i]]]
        else:
            lemma_input_char_vec = lookup[alphabet_index[EPSILON]]

        # prepare input vector and perform LSTM step
        decoder_input = concatenate([encoded, prev_output_vec, lemma_input_char_vec])
        s = s.add_input(decoder_input)

        # compute softmax probs and predict
        decoder_rnn_output = s.output()
        probs = softmax(R * decoder_rnn_output + bias)
        probs = probs.vec_value()
        next_char_index = common.argmax(probs)
        predicted = predicted + inverse_alphabet_index[next_char_index]

        # check if reached end of word
        if predicted[-1] == END_WORD:
            break

        # prepare for the next iteration
        # prev_output_vec = lookup[next_char_index]
        prev_output_vec = decoder_rnn_output
        i += 1

    # remove the begin and end word symbols
    return predicted[1:-1]


def evaluate_model(predictions, test_data, print_res=True):
    if print_res:
        print 'evaluating model...'

    c = 0
    for i, lemma in enumerate(predictions.keys()):
        (lemma, word) = test_data[i]
        predicted_word = predictions[lemma]
        if predicted_word == word:
            c += 1
            sign = 'V'
        else:
            sign = 'X'
        if print_res:
            print 'lemma: ' + lemma + ' gold: ' + word + ' prediction: ' + predicted_word + ' ' + sign
    accuracy = float(c) / len(predictions)

    if print_res:
        print 'finished evaluating model. accuracy: ' + str(c) + '/' + str(len(predictions)) + '=' + str(accuracy) + \
              '\n\n'

    return len(predictions), accuracy


def write_results_file(hyper_params, macro_avg_accuracy, micro_average_accuracy, train_path, test_path,
                       output_file_path, sigmorphon_root_dir, final_results):

    if 'test' in test_path:
        output_file_path += '.test'

    if 'dev' in test_path:
        output_file_path += '.dev'

    # write hyperparams, micro + macro avg. accuracy
    with codecs.open(output_file_path, 'w', encoding='utf8') as f:
        f.write('train path = ' + str(train_path) + '\n')
        f.write('test path = ' + str(test_path) + '\n')

        for param in hyper_params:
            f.write(param + ' = ' + str(hyper_params[param]) + '\n')

        f.write('Prediction Accuracy = ' + str(micro_average_accuracy) + '\n')
        f.write('Macro-Average Accuracy = ' + str(macro_avg_accuracy) + '\n')

    # write predictions in sigmorphon format
    predictions_path = output_file_path + '.predictions'
    with codecs.open(test_path, 'r', encoding='utf8') as test_file:
        lines = test_file.readlines()
        with codecs.open(predictions_path, 'w', encoding='utf8') as predictions:
            for i, line in enumerate(lines):
                lemma, morph, word = line.split()
                if i in final_results:
                    predictions.write(u'{0}\t{1}\t{2}\n'.format(lemma, morph, final_results[i][1]))
                else:
                    # TODO: handle unseen morphs?
                    print u'could not find prediction for {0} {1}'.format(lemma, morph)
                    predictions.write(u'{0}\t{1}\t{2}\n'.format(lemma, morph, 'ERROR'))

    # evaluate with sigmorphon script
    evaluation_path = output_file_path + '.evaluation'
    os.chdir(sigmorphon_root_dir)
    os.system('python ' + sigmorphon_root_dir + '/src/evalm.py --gold ' + test_path + ' --guesses ' + predictions_path +
              ' > ' + evaluation_path)
    os.system('python ' + sigmorphon_root_dir + '/src/evalm.py --gold ' + test_path + ' --guesses ' + predictions_path)

    print 'wrote results to: ' + output_file_path + '\n' + evaluation_path + '\n' + predictions_path
    return


def print_data_stats(alphabet, feats, morph_types, test_morph_types, train_feat_dicts, train_lemmas, train_words):
    print '\nalphabet' + str(sorted([f for f in alphabet if all(ord(c) < 128 for c in f)]))
    print 'features' + str(feats)
    print 'train_words: ' + str(len(train_words)) + ' ' + str(train_words[:10])
    print 'train_lemmas: ' + str(len(train_lemmas)) + ' ' + str(train_lemmas[:10])
    print 'train_feat_dicts: ' + str(len(train_feat_dicts)) + ' ' + str(train_feat_dicts[:1])
    print 'morph types: ' + str(len(morph_types)) + ' ' + str(morph_types.keys()[0])
    print 'verb morph types: ' + str(len([m for m in morph_types if 'pos=V' in m]))
    print 'noun morph types: ' + str(len([m for m in morph_types if 'pos=N' in m]))
    print 'test morph types: ' + str(len(test_morph_types)) + ' ' + str(test_morph_types.keys()[0])
    # for morph in morph_types:
    #    print morph


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
