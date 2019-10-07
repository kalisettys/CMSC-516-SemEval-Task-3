import pandas as pd
import spacy
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from scipy.stats import pearsonr

from tqdm import tqdm
import re
import time
import pickle
import os, ssl


from allennlp.commands.elmo import ElmoEmbedder
import scipy



######## Input Data
initial_data = pd.read_csv('trial_data.csv')

###converting data to dataframe using pandas

data_df = pd.DataFrame(initial_data)
data_df['word1'].value_counts()

#### removing punctuation
punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~.,'
data_df['clean_context1'] = data_df['context1'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))
data_df['clean_context2'] = data_df['context2'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))

data_df['clean_word1'] = data_df['word1'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))
data_df['clean_word2'] = data_df['word2'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))

#####converting everything to lower case
data_df['clean_context1'] = data_df['clean_context1'].str.lower()
data_df['clean_context2'] = data_df['clean_context2'].str.lower()

data_df['clean_word1'] = data_df['clean_word1'].str.lower()
data_df['clean_word2'] = data_df['clean_word2'].str.lower()

######removing numbers
data_df['clean_context1'] = data_df['clean_context1'].str.replace("[0-9]", " ")
data_df['clean_context2'] = data_df['clean_context2'].str.replace("[0-9]", " ")

data_df['clean_word1'] = data_df['clean_word1'].str.replace("[0-9]", " ")
data_df['clean_word2'] = data_df['clean_word2'].str.replace("[0-9]", " ")


########removing whitespaces
data_df['clean_context1'] = data_df['clean_context1'].apply(lambda x:' '.join(x.split()))
data_df['clean_context2'] = data_df['clean_context2'].apply(lambda x:' '.join(x.split()))


data_df['clean_word1'] = data_df['clean_word1'].apply(lambda x:' '.join(x.split()))
data_df['clean_word2'] = data_df['clean_word2'].apply(lambda x:' '.join(x.split()))


# importing spacy's language model to parse the data
spacy_nlp_parser = spacy.load('en', disable=['parser', 'ner'])

#### text lemmatization = a form of normalization in text
def lemmatizeText(textual_data):
    output = []
    for i in textual_data:
        s = [token.lemma_ for token in spacy_nlp_parser(i)]
        output.append(' '.join(s))
    return output

data_df['clean_context1'] = lemmatizeText(data_df['clean_context1'])
data_df['clean_context2'] = lemmatizeText(data_df['clean_context2'])

data_df['clean_word1'] = lemmatizeText(data_df['clean_word1'])
data_df['clean_word2'] = lemmatizeText(data_df['clean_word2'])


total_samples = 8
words_context1 = [[] for _ in range(total_samples)]
words_context2 = [[] for _ in range(total_samples)]

word1_context1 = [0] * 8
word2_context1 = [0] * 8
word1_context2 = [0] * 8
word2_context2 = [0] * 8

similarityScore_context1 = [0] * 8
similarityScore_context2 = [0] * 8

difference = [0] * 8

for i in range (8):

    words_context1[i] = data_df['clean_context1'][i].split(' ')
    words_context2[i] = data_df['clean_context2'][i].split(' ')

    if data_df['clean_word1'][i] in words_context1[i]:
        word1_context1[i] = words_context1[i].index(data_df['clean_word1'][i])

    if data_df['clean_word2'][i] in words_context1[i]:
        word2_context1[i] = words_context1[i].index(data_df['clean_word2'][i])

    if data_df['clean_word1'][i] in words_context2[i]:
        word1_context2[i] = words_context2[i].index(data_df['clean_word1'][i])

    if data_df['clean_word2'][i] in words_context2[i]:
        word2_context2[i] = words_context2[i].index(data_df['clean_word2'][i])

    elmo_embeddingModel = ElmoEmbedder()
    tokens = words_context1[i]
    vectors = elmo_embeddingModel.embed_sentence(tokens)

    assert (len(vectors) == 3)  # one for each layer in the ELMo output
    assert (len(vectors[0]) == len(tokens))

    similarityScore_context1[i] = 1 - scipy.spatial.distance.cosine(vectors[2][word1_context1[i]], vectors[2][word2_context1[i]])

    tokens2 = words_context2[i]
    vectors2 = elmo_embeddingModel.embed_sentence(tokens2)

    ##### 3 corresponds to the total number of layers in the elmo model (1 for each layer)
    assert (len(vectors2) == 3)
    assert (len(vectors2[0]) == len(tokens2))

    similarityScore_context2[i] = 1 - scipy.spatial.distance.cosine(vectors2[2][word1_context2[i]], vectors2[2][word2_context2[i]])



    difference[i] = similarityScore_context1[i] - similarityScore_context2[i]

print("-------------Difference between the effect of Context1 vs Context2 on the two pair of words------------")
print(difference)

print("----- Similarity of Word1 & Word2 in Context1 ------")
print(similarityScore_context1)

print("----- Similarity of Word1 & Word2 in Context1 ------")
print(similarityScore_context2)


### Spearman's Correlation
gold_data = pd.read_csv('gold_en.csv')
gold_df = pd.DataFrame(gold_data)
#print(gold_df["diff"])

gold_standard_scores = gold_df["diff"]
#print(gold_standard_scores)

spearman_value, p_value = spearmanr(difference, gold_standard_scores)

pearson_value, p_value = pearsonr(difference, gold_standard_scores)


print("-------- Spearman Correlation ------")
print(spearman_value)


print("-------- Pearson Correlation ------")
print(pearson_value)

alpha = 0.05
if p_value > alpha:
    print('The two samples are uncorrelated' % p_value)
else:
    print('The two samples are correlated' % p_value)

print("-------- P Value --------")
print(p_value)



'''
words = data_df['clean_context1'][0].split(' ')
words2 = data_df['clean_context2'][0].split(' ')

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
print(1-scipy.spatial.distance.cosine(vectors2[2][14], vectors2[2][35]))


start_index = data_df['clean_context1'][0].find('')
end_index = start_index + len('way')

print(data_df['clean_context1'][0].index('way'))
print(data_df['clean_context1'][0].index('manner'))
print(start_index)



elmo = ElmoEmbedder()
tokens = data_df['clean_context1'][0]
vectors = elmo.embed_sentence(tokens)

assert(len(vectors) == 3) # one for each layer in the ELMo output
assert(len(vectors[0]) == len(tokens)) # the vector elements correspond with the input tokens
print(data_df['clean_context1'][0][283:286])


print(data_df['clean_context1'][0][46:52])
print(1-scipy.spatial.distance.cosine(vectors[2][283:286], vectors[2][46:52]))

#####0.65113586
from allennlp.commands.elmo import ElmoEmbedder
import scipy
elmo = ElmoEmbedder()
tokens2 = data_df['clean_context2'][0]
vectors2= elmo.embed_sentence(tokens2)
assert(len(vectors) == 3) # one for each layer in the ELMo output
assert(len(vectors[0]) == len(tokens)) # the vector elements correspond with the input tokens
print(1-scipy.spatial.distance.cosine(vectors2[2][13], vectors2[2][33]))


if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context


elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
print(data_df.sample(3))


def elmo_vectors(x):
  embeddings = elmo(x.tolist(), signature="default", as_dict=True)["elmo"]
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    # return average of ELMo features
    return sess.run(tf.reduce_mean(embeddings,1))


list_train = [data_df[i:i+100] for i in range(0, data_df.shape[0],100)]

elmo_train = [elmo_vectors(x['clean_context1']) for x in list_train]
elmo_train2 = [elmo_vectors(x['clean_context2']) for x in list_train]

elmo_word1=[elmo_vectors(x['word1']) for x in list_train]
elmo_train_new = np.concatenate(elmo_train, axis = 0)

elmo_train_new2 = np.concatenate(elmo_train2, axis = 0)
elmo_word1_new=np.concatenate(elmo_word1, axis = 0)

print("train ",elmo_train[0].shape)
print("word ",elmo_word1[0].shape)
print("data2 shape",data_df.shape)


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
