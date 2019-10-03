
import csv
import pandas as pd
import spacy
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm
import re
import time
import pickle

from tqdm import tqdm
import re
import time
import pickle
from allennlp.commands.elmo import ElmoEmbedder
import scipy



######## Input Data

data=pd.read_csv('data2.csv')

######### Make Sure it is a dataframe structure

data2=pd.DataFrame(data)

data2['word1'].value_counts()




punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'

data2['clean_context1'] = data2['context1'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))

data2['clean_context2'] = data2['context2'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))


data2['clean_word1'] = data2['word1'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))

data2['clean_word2'] = data2['word2'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))

data2['clean_context1'] = data2['clean_context1'].str.lower()
data2['clean_context2'] = data2['clean_context2'].str.lower()



data2['clean_word1'] = data2['clean_word1'].str.lower()
data2['clean_word2'] = data2['clean_word2'].str.lower()

# remove numbers
data2['clean_context1'] = data2['clean_context1'].str.replace("[0-9]", " ")
data2['clean_context2'] = data2['clean_context2'].str.replace("[0-9]", " ")


# remove numbers
data2['clean_word1'] = data2['clean_word1'].str.replace("[0-9]", " ")
data2['clean_word2'] = data2['clean_word2'].str.replace("[0-9]", " ")


# remove whitespaces


data2['clean_context1'] = data2['clean_context1'].apply(lambda x:' '.join(x.split()))

data2['clean_context2'] = data2['clean_context2'].apply(lambda x:' '.join(x.split()))


data2['clean_word1'] = data2['clean_word1'].apply(lambda x:' '.join(x.split()))

data2['clean_word2'] = data2['clean_word2'].apply(lambda x:' '.join(x.split()))


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


data2['clean_word1'] = lemmatization(data2['clean_word1'])

data2['clean_word2'] = lemmatization(data2['clean_word2'])


n = 8
words = [[] for _ in range(n)]


words2 = [[] for _ in range(n)]

pos=[0]*8
pos2=[0]*8
pos3=[0]*8
pos4=[0]*8
sim1=[0]*8

sim2=[0]*8

diff=[0]*8

for i in range (8):
    words[i]= data2['clean_context1'][i].split(' ')
    words2[i]= data2['clean_context2'][i].split(' ')
    if data2['clean_word1'][i] in words[i]:
        pos[i] = words[i].index(data2['clean_word1'][i])
    if data2['clean_word2'][i] in words:
        pos2[i] = words[i].index(data2['clean_word2'][i])

    if data2['clean_word1'][i] in words2[i]:
        pos3[i] = words2[i].index(data2['clean_word1'][i])

    if data2['clean_word2'][i] in words2[i]:
        pos4[i] = words2[i].index(data2['clean_word2'][i])

    elmo = ElmoEmbedder()
    tokens = words[i]
    vectors = elmo.embed_sentence(tokens)

    assert (len(vectors) == 3)  # one for each layer in the ELMo output
    assert (len(vectors[0]) == len(tokens))

    sim1[i]=1 - scipy.spatial.distance.cosine(vectors[2][pos[i]], vectors[2][pos2[i]])

    tokens2 = words2[i]
    vectors2 = elmo.embed_sentence(tokens2)

    assert (len(vectors2) == 3)  # one for each layer in the ELMo output
    assert (len(vectors2[0]) == len(tokens2))
    sim2[i]=1 - scipy.spatial.distance.cosine(vectors2[2][pos3[i]], vectors2[2][pos4[i]])
    diff[i]=sim1[i]-sim2[i]

print(diff)

'''print(sim1)
print(sim2)'''







'''words = data2['clean_context1'][0].split(' ')
words2 = data2['clean_context2'][0].split(' ')

if 'way' in words:
    pos = words.index('way')

if 'manner' in words:
    pos2 = words.index('manner')

print(pos)
print(pos2)
print(words)


if 'way' in words2:
    pos3 = words2.index('way')

if 'manner' in words2:
    pos4 = words2.index('manner')

print(pos3)
print(pos4)
print(words2)



print(words2[14],words2[35])



from allennlp.commands.elmo import ElmoEmbedder
import scipy
elmo = ElmoEmbedder()
tokens =words
vectors = elmo.embed_sentence(tokens)

assert(len(vectors) == 3) # one for each layer in the ELMo output
assert(len(vectors[0]) == len(tokens))

print(1-scipy.spatial.distance.cosine(vectors[2][54], vectors[2][8]))

tokens2 =words2
vectors2 = elmo.embed_sentence(tokens2)

assert(len(vectors2) == 3) # one for each layer in the ELMo output
assert(len(vectors2[0]) == len(tokens2))
print(1-scipy.spatial.distance.cosine(vectors2[2][14], vectors2[2][35]))'''


'''start_index = data2['clean_context1'][0].find('')
end_index = start_index + len('way')

print(data2['clean_context1'][0].index('way'))


print(data2['clean_context1'][0].index('manner'))

print(start_index)


from allennlp.commands.elmo import ElmoEmbedder
import scipy
elmo = ElmoEmbedder()
tokens =data2['clean_context1'][0]
vectors = elmo.embed_sentence(tokens)

assert(len(vectors) == 3) # one for each layer in the ELMo output
assert(len(vectors[0]) == len(tokens)) # the vector elements correspond with the input tokens

print(data2['clean_context1'][0][283:286])
print(data2['clean_context1'][0][46:52])


print(1-scipy.spatial.distance.cosine(vectors[2][283:286], vectors[2][46:52]))'''

'''#####0.65113586


from allennlp.commands.elmo import ElmoEmbedder
import scipy
elmo = ElmoEmbedder()
tokens2 =data2['clean_context2'][0]
vectors2= elmo.embed_sentence(tokens2)

assert(len(vectors) == 3) # one for each layer in the ELMo output
assert(len(vectors[0]) == len(tokens)) # the vector elements correspond with the input tokens

print(1-scipy.spatial.distance.cosine(vectors2[2][13], vectors2[2][33]))'''



# import spaCy's language model












'''import os, ssl
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


elmo_word1=[elmo_vectors(x['word1']) for x in list_train]



elmo_train_new = np.concatenate(elmo_train, axis = 0)
elmo_train_new2 = np.concatenate(elmo_train2, axis = 0)


elmo_word1_new=np.concatenate(elmo_word1, axis = 0)

print("train ",elmo_train[0].shape)
print("word ",elmo_word1[0].shape)



print("data2 shape",data2.shape)






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
elmo_test_new2 = pickle.load(pickle_in2)'''
