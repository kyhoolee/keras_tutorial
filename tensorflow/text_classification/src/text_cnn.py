import tensorflow as tf
import numpy as np

class TextCNN(object):
    '''
    A CNN for text classification
    Embbeding -> convolutional -> max-pooling -> softmax
    '''
    def __init__(self, sequence_length,
                 num_classes, vocab_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='W')
        self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
        self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create convolution + maxpool layer for each filter size
        pooled_outputs = []

        for i, filter_size in enumerate(filter_sizes) :
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                # Convolution layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1,1,1,1],
                    padding='VALID',
                    name='conv'
                )

                # Non-linear
                h = tf.nn.relu(tf.nn.bias_add(conv,b), name='relu')

                # Maxpooling
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1,1,1,1],
                    padding='VALID',
                    name='pool'
                )
                pooled_outputs.append(pooled)


        # Combine all the pooled features


