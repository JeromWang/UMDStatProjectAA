from __future__ import absolute_import
from __future__ import print_function

import glob
import numpy as np
import sys
sys.setrecursionlimit(3000)
import codecs
import h5py
import yaml
import json
import itertools
import nltk.data
import os
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils.np_utils import accuracy, to_categorical, categorical_probas_to_classes
from keras.models import Sequential, Graph, model_from_json
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import ShuffleSplit
from six.moves import cPickle
from keras.callbacks import EarlyStopping, ModelCheckpoint

from model import get_main_model
from dataset_twitter import get_dataset_partitions
from miscelaneous import *
np.random.seed(0)


#vartrain_sampling_size = int(sys.argv[1]) if len(sys.argv) > 1 else None
conf_file = sys.argv[1] if len(sys.argv) > 1 else None

def main_run(X_train, X_test, y_train, y_test, maxlen, embedding_dims, max_features, filter_tuple, hidden_dims, dropout, activation, nb_classes, learning_rate, batch_size, nb_epochs, filename, pkl_file):
    
    #Declaration of some names for saving the json file, the weights file, the history pkl file
    h5file = filename + '_weights.hdf5'
    json_file = filename + '.json'

    #Callbacks declaration
    checkpointer = ModelCheckpoint(filepath=h5file, monitor='val_acc', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    history = LossHistory()
    dict_callbacks = {'history':history, 'checkpointer':checkpointer, 'early':early_stopping}
    
    #Get Keras Model
    model=get_main_model(maxlen, embedding_dims, max_features, filter_tuple, hidden_dims, nb_classes, dropout=dropout, activation_value=activation)
    adam = Adam(lr=learning_rate)
    
    #Compile Keras Model 
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epochs, validation_data=(X_test, y_test), callbacks=list(dict_callbacks.values()))
    
    #Save Model specification to a JSON
    json_string = model.to_json()
    open(json_file, 'w').write(json_string)
    
    #Evaluate model on the best saved model based on the accuracy
    score, acc = evaluate_model(json_file, h5file, X_test, y_test, adam, batch_size=batch_size)
    print_to_pkl(dict_callbacks['history'], pkl_file)
    return score, acc



with open(conf_file) as f:
    locals().update(json.load(f))

#In order to explore every testing digit, we'd have to create an additional loop 
#for testing_digit in range(10):
#print("For testing digit: ", testing_digit)
for setup in setups:
    if training_mod == "var_author":
        print("Varying number of authors setup")
        filename = head_file.format(setup=setup[0], conf=conf_file[-7:-5])
        pkl_file = head_histories.format(setup=setup[0], conf=conf_file[-7:-5])
        X_train, X_test, y_train, y_test, maxlen, nb_classes, vocab_size, char_to_ix, ix_to_char, train, test = get_dataset_partitions(-1, None, None, dataset_dir, setup[1], 0, n_examples=n_examples, oversampling=ovs, ngram=2)
        score, acc = main_run(X_train, X_test, y_train, y_test, maxlen, embedding_dims, vocab_size, filter_tuple, hidden_dims, dropout, activation, nb_classes, learning_rate, batch_size, nb_epochs, filename, pkl_file)
        print('Accuracy:', acc)
    else:
        print("Varying number of training instances setup")
        test_accuracies = []
        #ranges must be defined in the JSON file
        for xauthor, xauthor2 in ranges:
            spc_file = "{x}_{y}_{z}".format(x=setup, y=xauthor, z=xauthor2)
            filename = head_file.format(setup=spc_file, conf=conf_file[-7:-5])
            pkl_file = head_histories.format(setup=spc_file, conf=conf_file[-7:-5])
            
            X_train, X_test, y_train, y_test, maxlen, nb_classes, vocab_size, char_to_ix, ix_to_char, train, test = get_dataset_partitions(xauthor, xauthor2, setup, dataset_dir, authors_file, 0, n_examples=n_examples, oversampling=ovs, ngram=2)
            score, acc = main_run(X_train, X_test, y_train, y_test, maxlen, embedding_dims, vocab_size, filter_tuple, hidden_dims, dropout, activation, nb_classes, learning_rate, batch_size, nb_epochs, filename, pkl_file)
            test_accuracies.append(acc)
            print('Accuracy:', acc)
        avg_acc = np.mean(test_accuracies)
        print('Average accuracy:', avg_acc)
