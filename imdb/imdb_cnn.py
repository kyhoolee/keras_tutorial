from __future__ import print_function
import numpy as np 
np.random.seed(1337)

from keras.preprocessing import sequence
from keras.model import Sequential
from keras.layers import Dense, Dropout, Activation 
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.layers import Embedding
from keras.datasets import imdb


# set parameters
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
nb_filters = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 2


print('Loading data ...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words = max_features)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (sample x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test,max_len=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model ...')


model = Sequential()

# Efficient embedding layer
# map vocab indices into embedding_dims dimensions
model.add(Embedding(max_features, embedding_dims, 
	input_length=maxlen, dropout=0.2))

# Convolution1D
# learn nb_filter word group filters of size filter_length
model.add(Convolution1D(nb_filter=nb_filter,
	filter_length=filter_length,
	border_mode='valid',
	activation='relu',
	subsample_length=1))

# max_pooling
model.add(GlobalMaxPooling1D())

# Vanila hidden layer
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# Final classification layer
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
	optimizer='adam',
	metrics=['accuracy'])

model.fit(X_train, y_train,
	batch_size=batch_size,
	nb_epoch=nb_epoch,
	validation_data=(X_test, y_test))



