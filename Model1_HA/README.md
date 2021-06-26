# morphological-reinflection

Source code for the paper: [Morphological Inflection Generation with Hard Monotonic Attention](http://www.aclweb.org/anthology/P17-1183).


Requires [dynet](https://github.com/clab/dynet). You should compile the aligner before running `hard_attention.py` by 
running `make` while in the `src` directory.


A minor fix from Kat:
	-- added canonicalization (for the SM data after 2017 where grammatical categories are not presented);
	edit and run: preprocessdata.sh (it uses [Kyle Gorman's canonicalization script](https://github.com/unimorph/um-canonicalize))
	


Usage:

    hard_attention.py [--dynet-mem MEM][--input=INPUT] [--hidden=HIDDEN] [--feat-input=FEAT] [--epochs=EPOCHS] [--layers=LAYERS] [--optimization=OPTIMIZATION] [--reg=REGULARIZATION][--learning=LEARNING] [--plot] [--eval] [--ensemble=ENSEMBLE] TRAIN_PATH DEV_PATH TEST_PATH RESULTS_PATH SIGMORPHON_PATH...

Arguments:
* TRAIN_PATH    train set path
* DEV_PATH      development set path
* TEST_PATH     test set path
* RESULTS_PATH  results file (to be written)
* SIGMORPHON_PATH   sigmorphon repository root containing data, src dirs (available here: https://github.com/ryancotterell/sigmorphon2016)

Options:
* -h --help                     show this help message and exit
* --dynet-mem MEM               allocates MEM bytes for dynet
* --input=INPUT                 input embeddings dimension
* --hidden=HIDDEN               lstm hidden layer dimension
* --feat-input=FEAT             feature embeddings dimension
* --epochs=EPOCHS               number of training epochs
* --layers=LAYERS               number of layers in lstm
* --optimization=OPTIMIZATION   chosen optimization method ADAM/SGD/ADAGRAD/MOMENTUM/ADADELTA
* --reg=REGULARIZATION          regularization parameter for optimization
* --learning=LEARNING           learning rate parameter for optimization
* --plot                        draw a learning curve plot while training each model
* --eval                        run evaluation on existing model (without training)
* --ensemble=ENSEMBLE           ensemble model paths, separated by comma

For example:

    python hard_attention.py --dynet-mem 4096 --input=100 --hidden=100 --feat-input=20 --epochs=100 --layers=2 --optimization=ADADELTA  /Users/roeeaharoni/research_data/sigmorphon2016-master/data/navajo-task1-train /Users/roeeaharoni/research_data/sigmorphon2016-master/data/navajo-task1-dev /Users/roeeaharoni/research_data/sigmorphon2016-master/data/navajo-task1-test /Users/roeeaharoni/Dropbox/phd/research/morphology/inflection_generation/results/navajo_results.txt /Users/roeeaharoni/research_data/sigmorphon2016-master/
    
If you use this code for research purposes, please use the following citation:

    @InProceedings{aharoni-goldberg:2017:Long,
      author    = {Aharoni, Roee  and  Goldberg, Yoav},
      title     = {Morphological Inflection Generation with Hard Monotonic Attention},
      booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
      month     = {July},
      year      = {2017},
      address   = {Vancouver, Canada},
      publisher = {Association for Computational Linguistics},
      pages     = {2004--2015},
      abstract  = {We present a neural model for morphological inflection generation which employs
        a hard attention mechanism, inspired by the nearly-monotonic alignment commonly
        found between the characters in a word and the characters in its inflection. We
        evaluate the model on three previously studied morphological inflection
        generation datasets and show that it provides state of the art results in
        various setups compared to previous neural and non-neural approaches. Finally
        we present an analysis of the continuous representations learned by both the
        hard and soft (Bahdanau, 2014) attention models for the task, shedding some
        light on the features such models extract.},
      url       = {http://aclweb.org/anthology/P17-1183}
    }
