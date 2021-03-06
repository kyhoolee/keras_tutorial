from __future__ import print_function
import numpy as np 
import datetime

np.random.seed(1223)

from keras.datasets import mnist 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K 


now = datetime.datetime.now 

batch_size = 128 
nb_classes = 5
nb_epoch = 5

img_rows, img_cols = 28, 28
nb_filters = 32
pool_size = 2 
kernel_size = 3


if K.image_dim_ordering() == 'th':
	input_shape = (1,img_rows, img_cols)
else:
	input_shape = (img_rows, img_cols, 1)

def train_model(model, train, test, nb_classes):
	X_train = train[0].reshape((train[0].shape[0],) + input_shape)
	X_test = test[0].reshape((test[0].shape[0],) + input_shape)
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255

	print('X_train shape: ', X_train.shape)
	print(X_train.shape[0], ' train samples')
	print(X_test.shape[0], ' test samples')

	Y_train = np_utils.to_categorical(train[1], nb_classes)
	Y_test = np_utils.to_categorical(test[1], nb_classes)

	model.compile(loss='categorical_crossentropy', 
		optimizer='adadelta', metrics=['accuracy'])

	t = now()
	model.fit(X_train, Y_train, 
		batch_size=batch_size,
		nb_epoch=nb_epoch,
		verbose=1,
		validation_data=(X_test, Y_test))

	print('Training time: %s' % (now() - t))
	score = model.evaluate(X_test, Y_test, verbose=0)
	print('Test score: ', score[0])
	print('Test accuracy: ', score[1])

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train_lt5 = X_train[y_train < 5]
y_train_lt5 = y_train[y_train < 5]
X_test_lt5 = X_test[y_test < 5]
y_test_lt5 = y_test[y_test < 5]

X_train_gte5 = X_train[y_train >= 5]
y_train_gte5 = y_train[y_train >= 5] - 5
X_test_gte5 = X_test[y_test >= 5]
y_test_gte5 = y_test[y_test >= 5] - 5


# define 2 groups of layers: feature (convolutions) and classification (dense)

feature_layers = [
	Convolution2D(nb_filters, kernel_size, kernel_size,
		border_mode='valid', input_shape=input_shape),
	Activation('relu'),
	Convolution2D(nb_filters, kernel_size, kernel_size),
	Activation('relu'),
	MaxPooling2D(pool_size=(pool_size, pool_size)),
	Dropout(0.25),
	Flatten()
]

classification_layers = [
	Dense(128),
	Activation('relu'),
	Dropout(0.5),
	Dense(nb_classes),
	Activation('softmax')
]

model = Sequential(feature_layers + classification_layers)

train_model(model, 
	(X_train_lt5, y_train_lt5), (X_test_lt5, y_test_lt5), 
	nb_classes)

# freeze feature layers and rebuild model
for l in feature_layers:
	l.trainable = False

# transfer: train dense layers for new classification task
train_model(model, 
	(X_train_gte5, y_train_gte5),
	(X_test_gte5, y_test_gte5),
	nb_classes)
