"""Trains and evaluates a joint-structured model for inflection generation, using the sigmorphon 2016 shared task data
files and evaluation script.

Usage:
  task1_attention.py [--cnn-mem MEM][--input=INPUT] [--hidden=HIDDEN]
  [--feat-input=FEAT] [--epochs=EPOCHS] [--layers=LAYERS] [--optimization=OPTIMIZATION] [--reg=REGULARIZATION]
  [--learning=LEARNING] [--plot] [--dev=DEV] TRAIN_PATH TEST_PATH RESULTS_PATH SIGMORPHON_PATH...

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
  --dev=DEV                     development set file path
"""

import sys
sys.path.insert(0, './machine_translation/')
import prepare_data
import subprocess
import numpy as np
import random
import progressbar
import datetime
import time
import os
from multiprocessing import Pool
from matplotlib import pyplot as plt
import codecs
from docopt import docopt

import prepare_sigmorphon_data
import common


def main(train_path, dev_path, test_path, results_path):
    # read morph input files (train+dev)
    (train_words, train_lemmas, train_feat_dicts) = prepare_sigmorphon_data.load_data(train_path)
    (test_words, test_lemmas, test_feat_dicts) = prepare_sigmorphon_data.load_data(test_path)
    (dev_words, dev_lemmas, dev_feat_dicts) = prepare_sigmorphon_data.load_data(dev_path)

    merged_train_dev_lemmas = []
    merged_train_dev_words = []
    merged_train_dev_feat_dicts = []

    if dev_path != 'NONE':
        # merge the train and dev files, if dev exists
        merged_train_dev_lemmas += train_lemmas
        merged_train_dev_lemmas += dev_lemmas

        merged_train_dev_words += train_words
        merged_train_dev_words += dev_words

        merged_train_dev_feat_dicts += train_feat_dicts
        merged_train_dev_feat_dicts += dev_feat_dicts

    # TODO: optional - implement data augmentation

    # concatenate feats and characters for input
    tokenized_test_inputs, tokenized_test_outputs = convert_sigmorphon_to_MED_format(test_feat_dicts, test_lemmas, test_words)

    tokenized_train_inputs, tokenized_train_outputs = convert_sigmorphon_to_MED_format(train_feat_dicts, train_lemmas, train_words)

    tokenized_dev_inputs, tokenized_dev_outputs = convert_sigmorphon_to_MED_format(dev_feat_dicts, dev_lemmas, dev_words)

    tokenized_merged_inputs, tokenized_merged_outputs = convert_sigmorphon_to_MED_format(merged_train_dev_feat_dicts,
                                                                                         merged_train_dev_lemmas,
                                                                                         merged_train_dev_words)

    parallel_data = zip(tokenized_train_inputs, tokenized_train_outputs)

    # write input and output files
    train_inputs_file_path, train_outputs_file_path = write_converted_file(results_path,
                                                               tokenized_train_inputs,
                                                               tokenized_train_outputs,
                                                               'train.in',
                                                               'train.out')

    train_inputs_file_path, train_outputs_file_path = write_converted_file(results_path,
                                                                           tokenized_train_inputs,
                                                                           tokenized_train_outputs,
                                                                           'train.in.tok',
                                                                           'train.out.tok')

    test_inputs_file_path, test_outputs_file_path = write_converted_file(results_path,
                                                                           tokenized_test_inputs,
                                                                           tokenized_test_outputs,
                                                                           'test.in',
                                                                           'test.out')

    test_inputs_file_path, test_outputs_file_path = write_converted_file(results_path,
                                                                         tokenized_test_inputs,
                                                                         tokenized_test_outputs,
                                                                         'test.in.tok',
                                                                         'test.out.tok')

    merged_inputs_file_path, merged_outputs_file_path = write_converted_file(results_path,
                                                                         tokenized_merged_inputs,
                                                                         tokenized_merged_outputs,
                                                                         'merged.in',
                                                                         'merged.out')


    merged_inputs_file_path, merged_outputs_file_path = write_converted_file(results_path,
                                                                     tokenized_merged_inputs,
                                                                     tokenized_merged_outputs,
                                                                     'merged.in.tok',
                                                                     'merged.out.tok')

    dev_inputs_file_path, dev_outputs_file_path = write_converted_file(results_path,
                                                                             tokenized_dev_inputs,
                                                                             tokenized_dev_outputs,
                                                                             'dev.in',
                                                                             'dev.out')


    dev_inputs_file_path, dev_outputs_file_path = write_converted_file(results_path,
                                                                       tokenized_dev_inputs,
                                                                       tokenized_dev_outputs,
                                                                       'dev.in.tok',
                                                                       'dev.out.tok')


    # after the above files are created, hacky preprocess them by instantiating the args variables in prepare_data.py to
    # point the created files. only changes in original prepare_data.py code required for that are:

    # args.source = 'train.in'
    # args.target = 'train.out'
    # args.source_dev = 'test.in'
    # args.target_dev = 'test.out'

    # tr_files = ['/Users/roeeaharoni/GitHub/morphological-reinflection/src/machine_translation/data/train.in',
    #             '/Users/roeeaharoni/GitHub/morphological-reinflection/src/machine_translation/data/train.out']

    # change shuf to gshuf on mac

    # blocks search.py - line 102 - add on_unused_input='ignore'

    # eventually, run training script on the preprocessed files by changing those values in configuration.py:
    # bleu_val_freq, val_burn_in, val_set, val_set_grndtruth

    # and then run:
    # python -m machine_translation

    # finally run the script that converts the validation_out.txt file into the sigmorphon format and run evaluation
    sigmorphon_dev_file_path = dev_path
    MED_validation_file_path = './search_model_morph/validation_out.txt'
    output_file_path = './search_model_morph/validation_out.sigmorphon.txt'
    convert_MED_output_to_sigmorphon_format(sigmorphon_dev_file_path, MED_validation_file_path, output_file_path)

    return


def write_converted_file(results_path, tokenized_train_inputs, tokenized_train_outputs, in_suffix, out_suffix):
    inputs_file_path = results_path + in_suffix
    with codecs.open(inputs_file_path, 'w', encoding='utf8') as inputs:
        for input in tokenized_train_inputs:
            inputs.write(u'{}\n'.format(input))
    outputs_file_path = results_path + out_suffix
    with codecs.open(outputs_file_path, 'w', encoding='utf8') as outputs:
        for output in tokenized_train_outputs:
            outputs.write(u'{}\n'.format(output))
    print 'created source file: {} \n and target file:{}\n'.format(inputs_file_path, outputs_file_path)
    return inputs_file_path, outputs_file_path


def convert_sigmorphon_to_MED_format(train_feat_dicts, train_lemmas, train_words):
    tokenized_inputs = []
    tokenized_outputs = []
    train_set = zip(train_lemmas, train_feat_dicts, train_words)
    for i, example in enumerate(train_set):
        lemma, feats, word = example
        concatenated_input = ''
        for feat in feats:
            concatenated_input += feat + '=' + feats[feat] + ' '
        for char in lemma:
            concatenated_input += char + ' '

        # remove redundant space in the end
        tokenized_inputs.append(concatenated_input[:-1])

        # tokenize output
        tokenized_output = ''
        for char in word:
            tokenized_output += char + ' '

        # remove redundant space in the end
        tokenized_outputs.append(tokenized_output[:-1])
    return tokenized_inputs, tokenized_outputs


def convert_MED_output_to_sigmorphon_format(sigmorphon_dev_file_path, MED_validation_file_path, output_file_path):
    with codecs.open(sigmorphon_dev_file_path, 'r', encoding='utf8') as test_file:
        sig_lines = test_file.readlines()

        with codecs.open(MED_validation_file_path, 'r', encoding='utf8') as MED_file:
            med_lines = MED_file.readlines()

            with codecs.open(output_file_path, 'w', encoding='utf8') as predictions:
                for i, line in enumerate(sig_lines):
                    input = line.split('\t')[0]
                    feats = line.split('\t')[1]
                    prediction = med_lines[i].replace(' ','').replace('</S>\n','')
                    predictions.write(u'{0}\t{1}\t{2}\n'.format(input, feats, prediction))


if __name__ == '__main__':
    arguments = docopt(__doc__)
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

    # default values
    if arguments['TRAIN_PATH']:
        train_path_param = arguments['TRAIN_PATH']
    else:
        train_path_param = '/Users/roeeaharoni/research_data/sigmorphon2016-master/data/turkish-task1-train'
    if arguments['--dev']:
        dev_path_param = arguments['--dev']
    else:
        dev_path_param = 'NONE'
    if arguments['TEST_PATH']:
        test_path_param = arguments['TEST_PATH']
    else:
        test_path_param = 'NONE'
    if arguments['RESULTS_PATH']:
        results_file_path_param = arguments['RESULTS_PATH']
    else:
        results_file_path_param = \
            '/Users/roeeaharoni/Dropbox/phd/research/morphology/inflection_generation/results/results_' + st + '.txt'
    if arguments['SIGMORPHON_PATH']:
        sigmorphon_root_dir_param = arguments['SIGMORPHON_PATH'][0]
    else:
        sigmorphon_root_dir_param = '/Users/roeeaharoni/research_data/sigmorphon2016-master/'
    main(train_path_param, dev_path_param, test_path_param, results_file_path_param)