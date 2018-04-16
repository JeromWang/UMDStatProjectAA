from keras.preprocessing import sequence
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation, Flatten, TimeDistributedDense
from keras.layers import Input, Embedding, LSTM, Dense, merge, Lambda
from keras.models import Model
from keras.layers.convolutional import Convolution1D
from keras import backend as K


def max_1d(X):
    return K.max(X, axis=1)

def get_main_model(maxlen, embedding_dims, max_features, filter_tuple, hidden_dims, nb_classes, dropout=0.25, activation_value='relu'):
    #activation_value must be relu or tanh
    main_input = Input(shape=(maxlen,), dtype='int32', name='main_input')
    x = Embedding(max_features, embedding_dims, input_length=maxlen)(main_input)
    if dropout > 0:
        x = Dropout(dropout)(x)
    list_cnn_nodes = get_conv_layer(x, filter_tuple, activation_value)
    if len(list_cnn_nodes)>1:
        list_cnn_nodes = merge(list_cnn_nodes, mode='concat')
    else:
        list_cnn_nodes = list_cnn_nodes[0] #Fix this horrible trick
    if dropout > 0:
        list_cnn_nodes = Dropout(dropout)(list_cnn_nodes)
    fully_connected = Dense(hidden_dims, activation='relu')(list_cnn_nodes)
    main_loss = Dense(nb_classes, activation='softmax', name='main_output')(fully_connected)
    model = Model(input=main_input, output=main_loss)
    return model

def get_conv_layer(node_input, filter_tuple, activation_value):
    n_layers = len(filter_tuple)
    cnn_nodes = []
    for i in range(n_layers):
        cnn =  Convolution1D(nb_filter=filter_tuple[i][0], filter_length=filter_tuple[i][1], border_mode='valid', activation=activation_value, subsample_length=1)(node_input)
        cnn = Lambda(max_1d, output_shape=(filter_tuple[i][0],))(cnn)
        cnn_nodes.append(cnn)
    return cnn_nodes