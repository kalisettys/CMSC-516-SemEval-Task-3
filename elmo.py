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
from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm
import re
import time
import pickle

#####python -m spacy download en

######## Input Data

read_data=pd.read_csv('trial_data.csv')

######### Make Sure it is a dataframe structure

data_to_df=pd.DataFrame(read_data)

data_to_df['word1'].value_counts()

#print(data_to_df.head())



punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~,.'

data_to_df['clean_context1'] = data_to_df['context1'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))

data_to_df['clean_context2'] = data_to_df['context2'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))
data_to_df['clean_word1'] = data_to_df['word1'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))
data_to_df['clean_word2'] = data_to_df['word2'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))


data_to_df['clean_context1'] = data_to_df['clean_context1'].str.lower()
data_to_df['clean_context2'] = data_to_df['clean_context2'].str.lower()
data_to_df['clean_word1word1'] = data_to_df['clean_word1'].str.lower()
data_to_df['clean_word2'] = data_to_df['clean_word2'].str.lower()

#print(data_to_df['clean_context1'])

# remove numbers
data_to_df['clean_context1'] = data_to_df['clean_context1'].str.replace("[0-9]", " ")
data_to_df['clean_context2'] = data_to_df['clean_context2'].str.replace("[0-9]", " ")
data_to_df['clean_word1'] = data_to_df['clean_word1'].str.replace("[0-9]", " ")
data_to_df['clean_word2'] = data_to_df['clean_word2'].str.replace("[0-9]", " ")


# remove whitespaces


data_to_df['clean_context1'] = data_to_df['clean_context1'].apply(lambda x:' '.join(x.split()))
data_to_df['clean_context2'] = data_to_df['clean_context2'].apply(lambda x:' '.join(x.split()))
data_to_df['clean_word1'] = data_to_df['clean_word1'].apply(lambda x:' '.join(x.split()))
data_to_df['clean_word2'] = data_to_df['clean_word2'].apply(lambda x:' '.join(x.split()))


#print(data_to_df['clean_context1'])
#print(data_to_df['clean_context2'])

# import spaCy's language model
nlp = spacy.load('en', disable=['parser', 'ner'])


# function to lemmatize text
def lemmatization(texts):
    output = []
    for i in texts:
        s = [token.lemma_ for token in nlp(i)]
        output.append(' '.join(s))
    return output

data_to_df['clean_context1'] = lemmatization(data_to_df['clean_context1'])
data_to_df['clean_context2'] = lemmatization(data_to_df['clean_context2'])
data_to_df['clean_word1'] = lemmatization(data_to_df['clean_word1'])
data_to_df['clean_word2'] = lemmatization(data_to_df['clean_word2'])

####fixing an ssl error
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
        getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
#print(data_to_df.sample(3))
def elmo_vectors(x):
    embeddings = elmo_model(x.tolist(), signature="default", as_dict=True)["elmo"]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        # return average of ELMo features
        return sess.run(tf.reduce_mean(embeddings,1))

list_train = [data_to_df[i:i+100] for i in range(0,data_to_df.shape[0],100)]

elmo_context1 = [elmo_vectors(x['clean_context1']) for x in list_train]

elmo_context2 = [elmo_vectors(x['clean_context2']) for x in list_train]
elmo_word1 = [elmo_vectors(x['clean_word1']) for x in list_train]
elmo_word2 = [elmo_vectors(x['clean_word2']) for x in list_train]

context1_vecRep = np.concatenate(elmo_context1, axis = 0)
context2_vecRep = np.concatenate(elmo_context2, axis = 0)
word1_vecRep = np.concatenate(elmo_word1, axis = 0)
word2_vecRep = np.concatenate(elmo_word2, axis = 0)


##save vector to file
#save_vector_pickle_out = open("elmo_train_03032019.pickle","wb")
#pickle.dump(context1_vecRep, save_vector_pickle_out)
#save_vector_pickle_out.close()


#pickle_out2 = open("elmo_train2_03032019.pickle","wb")
#pickle.dump(context2_vecRep, pickle_out2)
#save_vector_pickle_out.close()

#grab_vector_pickle_in = open("elmo_train_03032019.pickle", "rb")
#elmo_train_new = pickle.load(grab_vector_pickle_in)

# load elmo_train_new
#final_grab = open("elmo_train2_03032019.pickle", "rb")
#elmo_test_new2 = pickle.load(final_grab)


print("word1 : ",word1_vecRep)
print("word2 : ",word2_vecRep)

print("Context 1:", context1_vecRep)
#print("Context 2:", context2_vecRep[0][0])

print("cosine similarity is :",cosine_similarity(word1_vecRep[0],context1_vecRep[0]))

