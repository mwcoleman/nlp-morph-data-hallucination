"""Runs a script on selected languages in parallel, supports ensemble training

Usage:
  run_all_langs_generic.py [--dynet-mem MEM][--input=INPUT] [--feat-input=FEAT][--hidden=HIDDEN] [--epochs=EPOCHS]
  [--layers=LAYERS] [--optimization=OPTIMIZATION] [--pool=POOL] [--langs=LANGS] [--script=SCRIPT] [--prefix=PREFIX]
  [--augment] [--merged] [--task=TASK] [--ensemble=ENSEMBLE] [--eval]
  SRC_PATH RESULTS_PATH SIGMORPHON_PATH...

Arguments:
  SRC_PATH  source files directory path
  RESULTS_PATH  results file to be written
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
  --pool=POOL                   amount of processes in pool
  --langs=LANGS                 languages separated by comma
  --script=SCRIPT               the training script to run
  --prefix=PREFIX               the output files prefix
  --augment                     whether to perform data augmentation
  --merged                      whether to train on train+dev merged
  --task=TASK                   the current task to train
  --ensemble=ENSEMBLE           the amount of ensemble models to train, 1 if not mentioned
  --eval                        run only evaluation without training
"""

import os
import time
import datetime
import docopt
from multiprocessing import Pool


# default values
INPUT_DIM = 200
FEAT_INPUT_DIM = 20
HIDDEN_DIM = 200
EPOCHS = 1
LAYERS = 2
OPTIMIZATION = 'ADAM'
POOL = 4
LANGS = ['russian', 'georgian', 'finnish', 'arabic', 'navajo', 'spanish', 'turkish', 'german', 'hungarian', 'maltese',
         'celex0', 'celex1', 'celex2' , 'celex3', 'celex4',
         'german500', 'german1000', 'german3000', 'german5000', 'german7000', 'german9000', 'german12000',
         'arabic500', 'arabic1000', 'arabic3000', 'arabic5000', 'arabic7000', 'arabic9000', 'arabic12000',
         'finnish500', 'finnish1000', 'finnish3000', 'finnish5000', 'finnish7000', 'finnish9000', 'finnish12000',
         'turkish500', 'turkish1000', 'turkish3000', 'turkish5000', 'turkish7000', 'turkish9000', 'turkish12000',
         'de_noun', 'de_verb', 'es_verb', 'fi_verb', 'fi_nounadj', 'french', 'dutch']
CNN_MEM = 2000


def main(src_dir, results_dir, sigmorphon_root_dir, input_dim, hidden_dim, epochs, layers,
         optimization, feat_input_dim, pool_size, langs, script, prefix, task, augment, merged, ensemble, eval_only):
    parallelize_training = True
    params = []
    print ('now training langs: ' + str(langs))
    for lang in langs:

        # check if an ensemble was requested
        ensemble_paths = []
        if ensemble > 1:
            for e in xrange(ensemble):

                # create prefix for ensemble
                ens_prefix = prefix + '_ens_{}'.format(e)

                if not eval_only:
                    # should train ensemble model: add params set for parallel model training execution
                    params.append([CNN_MEM, epochs, feat_input_dim, hidden_dim, input_dim, lang, layers, optimization,
                                   results_dir, sigmorphon_root_dir, src_dir, script, ens_prefix, task, augment,
                                   merged, '', eval_only])
                else:
                    # eval ensemble by generating a list of existing ensemble model paths and then passing it to
                    # the script as a single concatenated string parameter
                    ensemble_paths.append('{}/{}_{}-results.txt'.format(results_dir, ens_prefix, lang))
            if eval_only:
                concat_paths = ','.join(ensemble_paths)
                params.append([CNN_MEM, epochs, feat_input_dim, hidden_dim, input_dim, lang, layers, optimization,
                               results_dir, sigmorphon_root_dir, src_dir, script, prefix, task, augment,
                               merged, concat_paths, eval_only])
        else:
            # train a single model
            params.append([CNN_MEM, epochs, feat_input_dim, hidden_dim, input_dim, lang, layers, optimization,
                           results_dir, sigmorphon_root_dir, src_dir, script, prefix, task, augment, merged,
                           ensemble_paths, eval_only])

    # train models for each lang/ensemble in parallel or in loop
    if parallelize_training:
        pool = Pool(int(pool_size) * ensemble, maxtasksperchild=1)
        print ('now training {} langs in parallel, {} ensemble models per lang'.format(len(langs), ensemble))
        pool.map(train_language_wrapper, params)
    else:
        print ('now training {0} langs in loop, {} ensemble models per lang'.format(len(langs), ensemble))
        for p in params:
            train_language(*p)

    print ('finished training all models')


def train_language_wrapper(params):
    train_language(*params)


def train_language(cnn_mem, epochs, feat_input_dim, hidden_dim, input_dim, lang, layers, optimization, results_dir,
                   sigmorphon_root_dir, src_dir, script, prefix, task, augment, merged, ensemble_paths, eval_only):

    if augment:
        augment_str = '--augment'
    else:
        augment_str = ''

    if eval_param:
        eval_str = '--eval'
    else:
        eval_str = ''

    if len(ensemble_paths) > 0:
        ensemble_str = '--ensemble={}'.format(ensemble_paths)
    else:
        ensemble_str = ''

    start = time.time()
    os.chdir(src_dir)

    root_dir = src_dir.replace('/src/', '')


    # sampled train sets support (arabic500 etc.)
    sampled = ['german', 'arabic', 'turkish', 'finnish']
    found = [s for s in sampled if s in lang and len(lang) > len(s)]
    if len(found) > 0:
        l = found[0]
        train_path = '{}/data/sigmorphon_sampled/{}'.format(root_dir, lang)
        dev_path = '{}/data/{}-task{}-dev'.format(sigmorphon_root_dir, l, task)
        test_path = '{}/biu/gold/{}-task{}-test'.format(root_dir, l, task)
    else:
        # celex support
        if 'celex' in lang:
            if lang == 'celex':
                # backwards compatibillity
                fold = '0'
            else:
                fold = lang[-1]
            train_path = '{}/data/celex/13SIA-13SKE_2PIE-13PKE_2PKE-z_rP-pA_{}.train.txt'.format(root_dir, fold)
            dev_path = '{}/data/celex/13SIA-13SKE_2PIE-13PKE_2PKE-z_rP-pA_{}.dev.txt'.format(root_dir, fold)
            test_path = '{}/data/celex/13SIA-13SKE_2PIE-13PKE_2PKE-z_rP-pA_{}.test.txt'.format(root_dir, fold)
        else:
            if lang in ['de_noun', 'de_verb', 'es_verb', 'fi_verb', 'fi_nounadj']:
                train_path = '{}/data/ddn13/base_forms_{}_train.txt.sigmorphon_format.txt'.format(root_dir, lang)
                dev_path = '{}/data/ddn13/base_forms_{}_dev.txt.sigmorphon_format.txt'.format(root_dir, lang)
                test_path = '{}/data/ddn13/base_forms_{}_test.txt.sigmorphon_format.txt'.format(root_dir, lang)
            else:
                if lang in['dutch', 'french']:
                    train_path = '{}/data/nck15/{}_train.txt.sigmorphon_format.txt'.format(root_dir, lang)
                    dev_path = '{}/data/nck15/{}_dev.txt.sigmorphon_format.txt'.format(root_dir, lang)
                    test_path = '{}/data/nck15/{}_test.txt.sigmorphon_format.txt'.format(root_dir, lang)
                else:
                    # default sigmorphon format
                    train_path = '{}/data/{}-task{}-train'.format(sigmorphon_root_dir, lang, task)
                    dev_path = '{}/data/{}-task{}-dev'.format(sigmorphon_root_dir, lang, task)
                    test_path = '{}/biu/gold/{}-task{}-test'.format(root_dir, lang, task)

    # same for all
    results_path = '{}/{}_{}-results.txt'.format(results_dir, prefix, lang)

    if merged:
        train_path = '../data/sigmorphon_train_dev_merged/{}-task{}-merged'.format(lang, task)

    if 'attention' or 'ndst_twin_2' in script:
        # train on train, evaluate on dev for early stopping, finally eval on train
        command = 'python {0} --dynet-mem {1} --input={2} --hidden={3} --feat-input={4} --epochs={5} --layers={6} \
        --optimization={7} {8} {9} {10}\
        {11} \
        {12} \
        {13} \
        {14} \
        {15}'.format(script, cnn_mem, input_dim, hidden_dim, feat_input_dim, epochs, layers, optimization,
                     eval_str, augment_str, ensemble_str, train_path, dev_path, test_path, results_path,
                     sigmorphon_root_dir)
        print('\n' + command +'\n')
        os.system(command)
    else:
        # train on train+dev, evaluate on dev for early stopping
        os.system('python {0} --dynet-mem {1} --input={2} --hidden={3} --feat-input={4} --epochs={5} --layers={6} \
        --optimization={7} {8}\
        {9} \
        {10} \
        {11} \
        {12}'.format(script, cnn_mem, input_dim, hidden_dim, feat_input_dim, epochs, layers, optimization,
                     augment_str, train_path, dev_path, results_path, sigmorphon_root_dir))

    end = time.time()
    print('finished {} in {}'.format(lang, str(end - start)))


def evaluate_baseline(lang, results_dir, sig_root):
    os.chdir(sig_root + '/src/baseline')

    # run baseline system
    os.system('./baseline.py --task=1 --language={0} \
        --path={1}/data/ > {2}/baseline_{0}_task1_predictions.txt'.format(lang, sig_root, results_dir))
    os.chdir(sig_root + '/src')

    # eval baseline system
    os.system('python evalm.py --gold ../data/{0}-task1-dev --guesses \
        {1}/baseline_{0}_task1_predictions.txt'.format(lang, results_dir))


if __name__ == '__main__':
    arguments = docopt.docopt(__doc__)

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

    # default values
    if arguments['SRC_PATH']:
        src_dir_param = arguments['SRC_PATH']
    else:
        src_dir_param = '/Users/roeeaharoni/GitHub/morphological-reinflection/src/'
    if arguments['RESULTS_PATH']:
        results_dir_param = arguments['RESULTS_PATH']
    else:
        results_dir_param = '/Users/roeeaharoni/Dropbox/phd/research/morphology/inflection_generation/results/'
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
    if arguments['--pool']:
        pool_size_param = arguments['--pool']
    else:
        pool_size_param = POOL
    if arguments['--langs']:
        langs_param = [l.strip() for l in arguments['--langs'].split(',')]
    else:
        langs_param = LANGS
    if arguments['--script']:
        script_param = arguments['--script']
    else:
        print ('script is mandatory')
        raise ValueError
    if arguments['--prefix']:
        prefix_param = arguments['--prefix']
    else:
        print ('prefix is mandatory')
        raise ValueError
    if arguments['--task']:
        task_param = arguments['--task']
    else:
        task_param = '1'
    if arguments['--augment']:
        augment_param = True
    else:
        augment_param = False
    if arguments['--merged']:
        merged_param = True
    else:
        merged_param = False
    if arguments['--ensemble']:
        ensemble_param = int(arguments['--ensemble'])
    else:
        ensemble_param = 1
    if arguments['--eval']:
        eval_param = True
    else:
        eval_param = False

    print(arguments)

    main(src_dir_param, results_dir_param, sigmorphon_root_dir_param, input_dim_param, hidden_dim_param, epochs_param,
         layers_param, optimization_param, feat_input_dim_param, pool_size_param, langs_param, script_param,
         prefix_param, task_param, augment_param, merged_param, ensemble_param, eval_param)
