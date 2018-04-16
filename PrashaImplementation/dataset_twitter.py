from __future__ import absolute_import
from __future__ import print_function

import glob
import numpy as np
np.random.seed(0)
import sys
sys.setrecursionlimit(3000)
import codecs
import itertools
import os
import re
from collections import Counter
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.utils.np_utils import accuracy, to_categorical, categorical_probas_to_classes
from keras.models import Sequential, Graph, model_from_json
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import ShuffleSplit
from six.moves import cPickle


def load_authors(dataset_dir, authors_file):
    with codecs.open(authors_file, 'r', encoding='utf-8', errors='ignore') as f:
        list_authors = f.read().splitlines()
    return list_authors

#authors = list_authors[:50] #Nice trick

def process_listfiles(list_files, aut_to_ix, author, sample):
    corpus = []
    target = []
    for filename in list_files:
        with codecs.open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().splitlines()[:sample]
        target += [aut_to_ix[author]] * len(content)
        corpus += content
    return corpus, target

def aggregate_content(corpus, n_examples, aut_to_ix, author, oversampling):
    agg_corpus = []
    target = []
    if oversampling > 1 and n_examples > 1:
        ranging = np.arange(0, len(corpus), n_examples/oversampling)
    else:
        ranging = np.arange(0, len(corpus), n_examples)
    for i in ranging:
        agg_corpus.append('EB'.join(corpus[i:i+n_examples]))
    target += [aut_to_ix[author]] * len(agg_corpus)
    return agg_corpus, target

def get_dataset_partitions(xauthor, xauthor2, sampling_size, dataset_dir, authors_file, testing_digit, n_examples=1, oversampling=1, random=True, ngram=1):
    '''
    Parameters
    xauthor: Index position of the author list, if -1 given takes the complete authors_file.
    sampling_size: Number of tweets to be taken from each author.
    dataset_dir: Folder where the authors' tweets are located.
    authors_file: File specifying the authors' list.
    testing_digit: Fold to be used for test.
    n_examples: Number of training samples to be concatenated.
    oversampling: Oversample the training set if n_examples > 1, 2 of oversample means a 50%(1/2) oversampling
    random: Whether to shuffle the samples or not

    Returns
    X_train, X_test, y_train, y_test, maxlen, nb_classes, vocab_size, char_to_ix, ix_to_char, train(raw text), test(raw text)
    '''

    list_authors = load_authors(dataset_dir, authors_file) #Check if it's a good practice
    if xauthor >= 0:
        authors = list_authors[xauthor:xauthor2]
        print('From author_{a1} to author_{a2}'.format(a1=xauthor,a2=xauthor2))
    else:
        authors = list_authors

    train = []
    test = []
    y_train = []
    y_test = []
    aut_to_ix = {ch: i for i, ch in enumerate(authors)}
    ix_to_aut = {i: ch for i, ch in enumerate(authors)}
    for author in authors:
        #list_test = glob.glob(dataset_dir + author + "/*_[{td}].txt".format(td=testing_digit))
        #list_train = glob.glob(dataset_dir + author + "/*_[!{td}].txt".format(td=testing_digit))
        list_test = glob.glob(dataset_dir + author + "/*_[{td}].txt.segmented".format(td=testing_digit))
        list_train = glob.glob(dataset_dir + author + "/*_[!{td}].txt.segmented".format(td=testing_digit))
        corpus_train, _ = process_listfiles(list_train, aut_to_ix, author, sampling_size)
        corpus_test, _ = process_listfiles(list_test, aut_to_ix, author, sampling_size)
        #Aggregation function for varying text length
        agg_corpus_train, target_train = aggregate_content(corpus_train, n_examples, aut_to_ix, author, oversampling)
        agg_corpus_test, target_test = aggregate_content(corpus_test, n_examples, aut_to_ix, author, 1)#No oversampling on test
        train += agg_corpus_train
        test += agg_corpus_test
        y_train += target_train
        y_test += target_test
    
    #ngrams = [x[i:i+ngram]  for x in train for i in range(len(x)-ngram+1)]
    ngrams = []
    for x in train:
        ngrams += re.sub(r'(\\n|\\)','',str(x)).split()

    print(train[:10])
    print(Counter(ngrams))
    chars = list(set(ngrams))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars, 1)} #Enumerating from 1 to avoid confusion with zeropadding
    ix_to_char = {i: ch for i, ch in enumerate(chars, 1)}

    inputs_train = []
    inputs_test = []
    for line in train:
        inputs_train.append([char_to_ix[w] for w in re.sub(r'(\\n|\\)','',str(line)).split() if w in char_to_ix])
    for line in test:
        inputs_test.append([char_to_ix[w] for w in re.sub(r'(\\n|\\)','',str(line)).split() if w in char_to_ix])
    nb_classes = len(authors)
    print(inputs_train[0])
    maxlen = np.max([len(x) for x in inputs_train+inputs_test])

    X_train = sequence.pad_sequences(np.array(inputs_train), maxlen=maxlen)
    X_test = sequence.pad_sequences(np.array(inputs_test), maxlen=maxlen)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    if random:
        from sklearn.utils import shuffle
        X_train, y_train, train = shuffle(X_train, y_train, train, random_state=0)
        X_test, y_test, test = shuffle(X_test, y_test, test, random_state=0)
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, X_test, y_train, y_test, maxlen, nb_classes, vocab_size+1, char_to_ix, ix_to_char, train, test
