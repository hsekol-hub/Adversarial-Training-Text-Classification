import os
import re
import time
import en_core_web_lg
import contractions
import pandas as pd


cwd = os.getcwd()
print('Working Directory: ', cwd)

df = pd.read_csv(cwd + '/Imdb/data/raw/IMDB Dataset.csv')
df.dropna(inplace=True)
print('Unique lables:', df.groupby('sentiment').size())

nlp = en_core_web_lg.load()

def clean_corpus(row):
    # add your custom rules here; also do not change the order
    row = contractions.fix(row)
    row = row.lower()
    row = re.sub(r'<br />', '', row) # only for imdb dataset
    row = re.sub(r'[^a-z]+', ' ', row) # keeps only alphabets
    row = re.sub(r"\b[a-zA-Z]\b", "", row) # removes single characters
    row = re.sub(' +', ' ', row) # removes extra spaces
    return row

def stopwords(tokens):
    if tokens.is_stop == False:
        return tokens.lemma_

def spacy_tokenize(row):
    tokens = pd.Series(nlp(row))
    tokens = tokens.apply(stopwords)
    tokens = tokens[~tokens.isnull()].T
    tokens = tokens.str.cat(sep = ' ')
    return tokens

features = df['review']
start = time.time()
print('Cleaning dataset...')
features = features.apply(clean_corpus)
features = features.apply(spacy_tokenize)
stop = time.time()
print('Time elapsed: {} s'.format(round((stop-start),1)))

df = pd.DataFrame([features, df.sentiment]).T
df.to_csv(cwd + '/Imdb/data/processed/imdb_spacy.csv', index=False)


# Takes 42 minutes