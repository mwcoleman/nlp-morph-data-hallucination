Note: a detailed project report is [here](https://github.com/mwcoleman/nlp-morph-data-hallucination/raw/main/Report.pdf)

## Morphological inflection using transformers and n-gram data hallucination

Morphological reinflection is the task of generating a word inflection from a lemma (source word). Given the diversity of
morphological systems, it presents many hurdles to computational approaches. The SIGMORPHON shared task series is devoted
to this challenge. In recent years, neural networks have been without a doubt the top performer in NLP tasks and have repeatedly
achieved state of the art performance on a wide variety of challenges, including morphological reinflection.

### Baseline systems
(non neural naive baselines: ~30% accuracy './non-neural alignment models/'

1. Hard-Attention model (Aharoni & Goldberg, 2017) - With and without data augmentation
2. Character level transduction with the Transformer model (Wu et al, 2020) - With and without data augmentation

### Data augmentation

The contribution in this project is the development of an n-gram model data hallucination approach to augment low resource 
(<1000 inflection pairs) training data sets. Average performance (accuracy) increased by ~2%.

Sampling of sequences are without replacement from a categorical distribution utilising a markov model (over
observed sequences in the original training data), where the probability of a replacement is conditioned on the preceding
n-characters (i.e. ngram)

Replacement sequences can have the following flexible characterstics:
- They can be k +/- m characters long, where k is the original sequence length, and m <= n
- They can be ‘fixed’ or ‘compositional’; the former is where the sequence must have been observed in the training
data in its entirety, the latter is where the sequence can be made up of recursive sampling
- (providing it does not conflict with the above choices) a maximum sequence length may be specified (e.g. when
compositional, we may implement a strict character-level unigram model if desired)
- Multiple ‘stems’ can be replaced, it is possible to specify a minimum length criteria for stem replacement (default is 2)


### Usage (data augmentation)

`python ngram_hall.py [lang] [output_version_name] [target dataset size]`  
(There are additional optional parameters, see the report for info)
