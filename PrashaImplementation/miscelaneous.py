from __future__ import absolute_import
from __future__ import print_function

import glob
import numpy as np
import sys
sys.setrecursionlimit(3000)
import codecs
import json
import itertools
import os
from keras.utils import np_utils
from keras.utils.np_utils import accuracy, to_categorical, categorical_probas_to_classes
from keras.models import model_from_json
from keras import backend as K
from keras.callbacks import Callback
from sklearn.metrics import accuracy_score
from six.moves import cPickle
np.random.seed(0)


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accs = []
        self.val_losses = []
        self.val_accs = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accs.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_accs.append(logs.get('val_acc'))
        
        
def evaluate_model(json_file, h5file, X_test, y_test, optimizer, batch_size=32):
    new_model = model_from_json(open(json_file).read())
    new_model.load_weights(h5file)
    new_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    score, acc = new_model.evaluate(X_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
    return score, acc

def print_to_pkl(pkl_object, output_file):
    pkl_file = open(output_file, 'wb')
    sys.setrecursionlimit(3000)
    cPickle.dump(pkl_object, pkl_file, protocol=cPickle.HIGHEST_PROTOCOL)
    pkl_file.close()
    return True

def get_histories(hist_source):
    #hist_source must be in glob format
    with open(hist_source, 'rb') as f:
        loaded_obj = cPickle.load(f)
    return loaded_obj.losses, loaded_obj.accs, loaded_obj.val_losses, loaded_obj.val_accs
    