'''
Training recurrent convolutional network on IMDB sentiment classification 
'''

from __future__ import print_function 
import numpy as np 
np.random.seed(1234)

from keras.preprocessing import sequence
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Activation 
from keras.layers import Embedding 
from keras.layers import LSTM 
from keras.layers import Convolution1D, MaxPooling1D
from keras.datasets import imdb 

# Embedding 
max_features = 20000
maxlen = 100
embedding_size = 128 

# Convolution 
filter_length = 5 
nb_filter = 64 
pool_length = 4 

# LSTM
lstm_output_size = 70 

# Training 
batch_size = 30 
nb_epoch = 2


print('Loading data ...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model ...')


model = Sequential()

model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Dropout(0.25))
model.add(Convolution1D(nb_filter=nb_filter,
	filter_length=filter_length,
	border_mode='valid',
	activation='relu',
	subsample_length=1))
model.add(MaxPooling1D(pool_length=pool_length))
model.add(LSTM(lstm_output_size))
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.summary()

model.compile(loss='binary_crossentropy',
	optimizer='adam',
	metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
	validation_data=(X_test, y_test))

score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print('\n')
print('Test score:', score)
print('Test accuracy:', acc)