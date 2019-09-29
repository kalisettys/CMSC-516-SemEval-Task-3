import csv
import pandas as pd
import spacy
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tqdm import tqdm
import re
import time
import pickle

from tqdm import tqdm
import re
import time
import pickle



######## Input Data

data=pd.read_csv('data2.csv')

######### Make Sure it is a dataframe structure

data2=pd.DataFrame(data)

data2['word1'].value_counts()

print(data2.head())


punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'

data2['clean_context1'] = data2['context1'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))

data2['clean_context2'] = data2['context2'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))


data2['clean_context1'] = data2['clean_context1'].str.lower()
data2['clean_context2'] = data2['clean_context2'].str.lower()

print(data2['clean_context1'])

# remove numbers
data2['clean_context1'] = data2['clean_context1'].str.replace("[0-9]", " ")
data2['clean_context2'] = data2['clean_context2'].str.replace("[0-9]", " ")


# remove whitespaces


data2['clean_context1'] = data2['clean_context1'].apply(lambda x:' '.join(x.split()))

data2['clean_context2'] = data2['clean_context2'].apply(lambda x:' '.join(x.split()))


print(data2['clean_context1'])

print(data2['clean_context2'])


# import spaCy's language model



# import spaCy's language model
nlp = spacy.load('en', disable=['parser', 'ner'])


# function to lemmatize text
def lemmatization(texts):
    output = []
    for i in texts:
        s = [token.lemma_ for token in nlp(i)]
        output.append(' '.join(s))
    return output



data2['clean_context1'] = lemmatization(data2['clean_context1'])

data2['clean_context2'] = lemmatization(data2['clean_context2'])


import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
print(data2.sample(3))
def elmo_vectors(x):
  embeddings = elmo(x.tolist(), signature="default", as_dict=True)["elmo"]

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    # return average of ELMo features
    return sess.run(tf.reduce_mean(embeddings,1))

list_train = [data2[i:i+100] for i in range(0,data2.shape[0],100)]

elmo_train = [elmo_vectors(x['clean_context1']) for x in list_train]

elmo_train2 = [elmo_vectors(x['clean_context2']) for x in list_train]

elmo_train_new = np.concatenate(elmo_train, axis = 0)
elmo_train_new2 = np.concatenate(elmo_train2, axis = 0)

pickle_out = open("elmo_train_03032019.pickle","wb")
pickle.dump(elmo_train_new, pickle_out)
pickle_out.close()


pickle_out2 = open("elmo_train2_03032019.pickle","wb")
pickle.dump(elmo_train_new2, pickle_out2)
pickle_out.close()

pickle_in = open("elmo_train_03032019.pickle", "rb")
elmo_train_new = pickle.load(pickle_in)

# load elmo_train_new
pickle_in2 = open("elmo_train2_03032019.pickle", "rb")
elmo_test_new2 = pickle.load(pickle_in2)
