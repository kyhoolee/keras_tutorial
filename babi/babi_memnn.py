'''
Train memory network on the bAbI dataset
- End-To-End memory network
'''

from __future__ import print_function

from keras.models import Sequential 
from keras.layers.embeddings import Embedding 
from keras.layers import Activation, Dense, Merge, Permute, Dropout
from keras.layers import LSTM
from keras.utils.data_utils import get_file 
from keras.preprocessing.sequence import pad_sequences
from functools import reduce 
import tarfile 
import numpy as np 
import re 


def tokenize(sent):
	return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

def parse_stories(lines, only_supporting=False):
	data = [] 
	story = []
	for line in lines:
		line = line.decode('utf-8').strip()
		nid, line = line.split(' ', 1)
		nid = int(nid)
		if nid == 1:
			story = []
		if '\t' in line:
			q,a,supporting = line.split('\t')
			q = tokenize(q)
			substory = None
			if only_supporting:
				supporting = map(int, supporting.split())
				substory = [story[i-1] for i in supporting]
			else:
				substory = [x for x in story if x]
			data.append((substory, q, a))
			story.append('')
		else:
			sent = tokenize(line)
			story.append(sent)

	return data 


def get_stories(f, only_supporting=False, max_length=None):
	data = parse_stories(f.readlines(), only_supporting=only_supporting)
	print(len(data))
	flatten = lambda data: reduce(lambda x, y: x+y, data)
	data = [(flatten(story), q, answer) 
		for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
	return data 

def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
	X = []
	Xq = []
	Y = []

	for story, query, answer in data:
		x = [word_idx[w] for w in story]
		xq = [word_idx[w] for w in query]
		y = np.zeros(len(word_idx) + 1)
		y[word_idx[answer]] = 1
		X.append(x)
		Xq.append(xq)
		Y.append(y)

	return (pad_sequences(X, maxlen=story_maxlen),
		pad_sequences(Xq, maxlen=query_maxlen), 
		np.array(Y))


try:
	path = get_file('babi-tasks-v1-2.tar.gz', 
		origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
except:
	print('Error downloading')
	raise 

tar = tarfile.open(path)

challenges = {
	'single_supporting_fact_10k':'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
	'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
}

challenge_type = 'single_supporting_fact_10k'
challenge = challenges[challenge_type]

print('Extracting stories for the challenge:', challenge_type)
train_stories = get_stories(tar.extractfile(challenge.format('train')))
test_stories = get_stories(tar.extractfile(challenge.format('test')))

vocab = sorted(reduce(lambda x, y : x | y, (set(story + q + [answer])
	for story, q, answer in train_stories + test_stories)))

vocab_size = len(vocab) + 1
story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

print('-')
print('Vocab size:', vocab_size, 'unique words')
print('Story max length:', story_maxlen, 'words')
print('Query max length:', query_maxlen, 'words')
print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))
print('-')
print('Here\'s what a "story" tuple looks like (input, query, answer):')
print(train_stories[0])
print('-')
print('Vectorizing the word sequences ...')

word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
inputs_train, queries_train, answers_train = vectorize_stories(
	train_stories, word_idx, story_maxlen, query_maxlen)
inputs_test, queries_test, answers_test = vectorize_stories(
	test_stories, word_idx, story_maxlen, query_maxlen)

print('-')
print('inputs: integer tensor of shape (samples, max_length)')
print('inputs_train shape:', inputs_train.shape)
print('inputs_test shape:', inputs_test.shape)
print('-')
print('queries: integer tensor of shape (samples, max_length)')
print('queries_train shape:', queries_train.shape)
print('queries_test shape:', queries_test.shape)
print('-')
print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
print('answers_train shape:', answers_train.shape)
print('answers_test shape:', answers_test.shape)
print('-')
print('Compiling...')




# Model
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size,
	output_dim=64, input_length=story_maxlen))
input_encoder_m.add(Dropout(0.3))

question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_size,
	output_dim=64, input_length=query_maxlen))
question_encoder.add(Dropout(0.3))

match = Sequential()
match.add(Merge([input_encoder_m, question_encoder],
	mode='dot', dot_axes=[2,2]))
match.add(Activation('softmax'))


input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size, 
	output_dim=query_maxlen, input_length=story_maxlen))
input_encoder_c.add(Dropout(0.3))

response = Sequential()
response.add(Merge([match, input_encoder_c], mode='sum'))
response.add(Permute((2,1)))


answer = Sequential()
answer.add(Merge([response, question_encoder], mode='concat', concat_axis=-1))

answer.add(LSTM(32))
answer.add(Dropout(0.3))
answer.add(Dense(vocab_size))
answer.add(Activation('softmax'))

answer.summary()

answer.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
answer.fit([inputs_train, queries_train, inputs_train],
	answers_train,
	batch_size=32,
	validation_data=([inputs_test, queries_test, inputs_test], answers_test))


loss, acc = answer.evaluate([inputs_test, queries_test, inputs_test], answers_test, batch_size=32)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))


