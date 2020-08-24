import os
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


cwd = os.getcwd()
print('Working Directory: ', cwd)
path = cwd + '/Fake News/data'

x_train = np.load(path + '/processed/xtr_shuffled.npy', allow_pickle=True)
x_test = np.load(path + '/processed/xte_shuffled.npy', allow_pickle=True)
y_train = np.load(path + '/processed/ytr_shuffled.npy', allow_pickle=True).astype('int')
y_test = np.load(path + '/processed/yte_shuffled.npy', allow_pickle=True).astype('int')
unlabelled = np.load(path + '/processed/xun_shuffled.npy', allow_pickle=True)


MAX_VOCAB_SIZE = 1000000
MAX_DOC_LENGTH = 100

corpus = np.concatenate((x_train, x_test, unlabelled))
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token='UNK')
tokenizer.fit_on_texts(corpus)
word_index = tokenizer.word_index
print('Vocabulary size :', len(word_index))


def encode(data):
    data = tokenizer.texts_to_sequences(data)
    data = pad_sequences(data, padding='post', maxlen=MAX_DOC_LENGTH)
    return data

x_train = encode(x_train)
x_test = encode(x_test)
unlabelled = encode(unlabelled)

encoded_docs = np.concatenate((x_train, x_test, unlabelled))

len_list = [len(row) for row in unlabelled]

print('Mean length of corpus in terms of words: ', np.mean(len_list))
print('Max length of corpus in terms of words: ', np.max(len_list))
print('Min length of corpus in terms of words: ', np.min(len_list))
print('Median length of corpus in terms of words: ', np.median(len_list))


pickle_out = open(path + '/meta/word_index.pickle','wb')
pickle.dump(word_index, pickle_out)
pickle_out.close()

print('Word Index saved locally')

np.save(path + '/temp/x_train.npy', x_train)
np.save(path + '/temp/x_test.npy', x_test)
np.save(path + '/temp/unlabelled.npy', unlabelled)
np.save(path + '/temp/y_train.npy', y_train)
np.save(path + '/temp/y_test.npy', y_test)

print('Features and labels saved locally in npy format')
