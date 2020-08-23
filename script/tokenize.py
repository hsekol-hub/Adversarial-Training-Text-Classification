import os
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


cwd = os.getcwd()
print('Working Directory: ', cwd)
df = pd.read_csv(cwd + '/Imdb/data/processed/imdb_spacy.csv')

def encode_labels(row):
    """
    Feature encoding for labels
    :param row: each dataset row
    :return: label
    """
    if row == 'positive':
        return 1
    elif row == 'negative':
        return 0
    else:
        return -1

features = df.review
labels = np.array(df.sentiment.apply(encode_labels))

MAX_VOCAB_SIZE = 1000000


tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token='UNK')
tokenizer.fit_on_texts(features)


encoded_docs = tokenizer.texts_to_sequences(features)
word_index = tokenizer.word_index
print('Vocabulary size :', len(word_index))

len_list = [len(row) for row in encoded_docs]
print('Mean length of corpus in terms of words: ', np.mean(len_list))
print('Max length of corpus in terms of words: ', np.max(len_list))
print('Min length of corpus in terms of words: ', np.min(len_list))
print('Median length of corpus in terms of words: ', np.median(len_list))

MAX_DOC_LENGTH = 100

features = pad_sequences(encoded_docs, padding='post', maxlen=MAX_DOC_LENGTH)

pickle_out = open(cwd + '/Imdb/data/meta/word_index.pickle','wb')
pickle.dump(word_index, pickle_out)
pickle_out.close()

print('Word Index saved locally')
np.save(cwd + '/Imdb/data/feature_tokens.npy', features)
np.save(cwd + '/Imdb/data/label_tokens.npy', labels)
print('Features and labels saved locally in npy format')


