from keras.layers import Input, Dense, Embedding
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from keras.preprocessing.text import one_hot
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from collections import Counter
from keras.utils import np_utils
import nltk
import itertools
import csv
import numpy as np
import re
import os

set_num = 100
def baseline(docs,labels):
	tokenizer = nltk.word_tokenize
	vect = CountVectorizer(ngram_range=(1, 2),tokenizer=tokenizer,min_df=5,strip_accents ='unicode')
	vect.fit(docs)
	docs_bow = vect.transform(docs)
	lrclf = LogisticRegression(C=1,solver = 'lbfgs',random_state=0,penalty = 'l2' ,n_jobs=len(set(labels)))
	print("BOWLR baseline cross-validation:")
	print(max(cross_val_score(lrclf,docs_bow,labels,cv=10)))

def load_data():
	# define documents
	labels = []
	docs = []
	for line in open('tweets','r'):  
		docs.append(line) 
	for line in open('labels','r'):  	
		labels.append(line) 	

	category_num = len(set(labels))
	labels = np_utils.to_categorical(labels, category_num)

	print (docs[0])
	
	vocab = []
	for x in docs:
		vocab += re.sub(r'(\\n|\\)','',str(x)).split()

	vocab_size = min(len(set(vocab)))
	
	w2id = {w: i for i, w in enumerate(vocab, 1)}
	id2w = {i: w for i, w in enumerate(vocab, 1)}
	
	embedded_docs = []
	for line in docs:
		embedded_docs.append([w2id[w] for w in re.sub(r'(\\n|\\)','',str(line)).split() if w in w2id])
	#embedded_docs = [one_hot(doc, vocab_size) for doc in docs]
	print (embedded_docs[0])

	max_length = max(len(x) for x in embedded_docs)

        # pad documents to max length
	padded_docs = pad_sequences(embedded_docs, maxlen=max_length, padding='post')
	print (padded_docs[0])

	print(vocab_size,max_length)
	return padded_docs, labels, max_length, vocab_size, category_num


print('Loading data')
#x, y, vocabulary, vocabulary_inv = load_data()

X,y, sequence_length, vocabulary_size, category_num = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=30)


#sequence_length = x.shape[1] # 56
#vocabulary_size = len(vocabulary_inv) # 18765
embedding_dim = 256
filter_sizes = [3,4,5]
num_filters = 500
drop = 0.25

epochs = 100
batch_size = 32

# this returns a tensor
print("Creating Model...")
inputs = Input(shape=(sequence_length,), dtype='int32')
embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
dropout_0 = Dropout(drop)(embedding)
reshape = Reshape((sequence_length,embedding_dim,1))(dropout_0)

conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)


concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = Flatten()(concatenated_tensor)
dropout_1 = Dropout(drop)(flatten)
hidden = Dense(units=512, activation='relu')(dropout_1)
output = Dense(units=category_num, activation='softmax')(hidden)


# this creates a model that includes
model = Model(inputs=inputs, outputs=output)

checkpoint = ModelCheckpoint('AA-W.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
print("Traning Model...")
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, callbacks=[checkpoint], validation_data=(X_test, y_test))  # starts training


