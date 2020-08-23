import os
import random
import numpy as np

cwd = os.getcwd()
print('Working Directory: ', cwd)
path = cwd + '/Imdb/data'

EMBEDDING_DIM = 300
MAX_DOC_LENGTH = 100

def load_data():
  """

  :return: features, labels from local directory (tokens in npy format)
  """
  features = np.load(path + '/feature_tokens.npy')
  labels = np.load(path + '/label_tokens.npy')
  return features, labels

def train_test_split(split_ratio = 0.3):
  """

  :param split_ratio: train test split ratio
  :return: Saves training and testing set in temporary folders and is overwritten for the next dataset
  """
  features, labels = load_data()
  inds = np.arange(features.shape[0])
  random.Random(1).shuffle(inds)
  features = features[inds]
  labels = labels[inds]
  features = features
  labels = labels

  num_test_samples = int(split_ratio * features.shape[0])
  print('Split ratio {}/{}:'.format(num_test_samples, features.shape[0]))

  x_train = features[:-num_test_samples]
  y_train = labels[:-num_test_samples]
  x_test = features[-num_test_samples:]
  y_test = labels[-num_test_samples:]
  print("Training size:", x_train.shape, y_train.shape)
  print("Testing size:", x_test.shape, y_test.shape)

  np.save(path + '/temp/50k/x_train.npy', x_train)
  np.save(path + '/temp/50k/y_train.npy', y_train)
  np.save(path + '/temp/50k/x_test.npy', x_test)
  np.save(path + '/temp/50k/y_test.npy', y_test)
  print('Done')



train_test_split()