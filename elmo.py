import pandas as pd
import spacy
import scipy
from scipy.stats import spearmanr
import sys
from matplotlib.pylab import plt

spacy_nlp_parser = spacy.load('en', disable=['parser', 'ner'])

from allennlp.commands.elmo import ElmoEmbedder

def main():
    # Input Data
    file_name=sys.argv[1]
    initial_data = pd.read_csv(file_name)

    ##converting data to dataframe using pandas
    data_df = pd.DataFrame(initial_data)
    total_samples = data_df['word1'].count()

    data_df = cleaning_data(data_df)
    data_df = Lemmatization(data_df)
    diff, sim_context1, sim_context2 = elmo_Model(total_samples, data_df)

    print("Similarity of Word1 & Word2 in Context1:", sim_context1)
    print("Similarity of Word1 & Word2 in Context2:", sim_context2)

    print("-------------Difference between the effect of Context1 vs Context2 on the two pair of words------------")
    print(diff)

    spearman_value, p_value = correlationMethod(diff)
    print("-------- Spearman Correlation ------")
    print(spearman_value)
    print("-------- P Value --------")
    print(p_value)
    plotting(sim_context1,sim_context2,diff,data_df,total_samples)

def cleaning_data(data_df):
    #Removing punctuation
    punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~.,'
    data_df['clean_context1'] = data_df['context1'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))
    data_df['clean_context2'] = data_df['context2'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))
    data_df['clean_word1'] = data_df['word1'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))
    data_df['clean_word2'] = data_df['word2'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))

    #Converting to lower case
    data_df['clean_context1'] = data_df['clean_context1'].str.lower()
    data_df['clean_context2'] = data_df['clean_context2'].str.lower()
    data_df['clean_word1'] = data_df['clean_word1'].str.lower()
    data_df['clean_word2'] = data_df['clean_word2'].str.lower()

    #Removing numbers
    data_df['clean_context1'] = data_df['clean_context1'].str.replace("[0-9]", " ")
    data_df['clean_context2'] = data_df['clean_context2'].str.replace("[0-9]", " ")
    data_df['clean_word1'] = data_df['clean_word1'].str.replace("[0-9]", " ")
    data_df['clean_word2'] = data_df['clean_word2'].str.replace("[0-9]", " ")


    #Removing whitespaces
    data_df['clean_context1'] = data_df['clean_context1'].apply(lambda x:' '.join(x.split()))
    data_df['clean_context2'] = data_df['clean_context2'].apply(lambda x:' '.join(x.split()))
    data_df['clean_word1'] = data_df['clean_word1'].apply(lambda x:' '.join(x.split()))
    data_df['clean_word2'] = data_df['clean_word2'].apply(lambda x:' '.join(x.split()))

    return data_df

#### Text lemmatization = a form of normalization in text
def lemmatizeText(textual_data):
    output = []
    for i in textual_data:
        s = [token.lemma_ for token in spacy_nlp_parser(i)]
        output.append(' '.join(s))
    return output


def Lemmatization(data_df):
    data_df['clean_context1'] = lemmatizeText(data_df['clean_context1'])
    data_df['clean_context2'] = lemmatizeText(data_df['clean_context2'])
    data_df['clean_word1'] = lemmatizeText(data_df['clean_word1'])
    data_df['clean_word2'] = lemmatizeText(data_df['clean_word2'])
    return data_df

def elmo_Model(total_samples, data_df):

    words_context1 = [[] for _ in range(total_samples)]
    words_context2 = [[] for _ in range(total_samples)]

    word1_context1 = [0] * total_samples
    word2_context1 = [0] * total_samples
    word1_context2 = [0] * total_samples
    word2_context2 = [0] * total_samples

    similarityScore_context1 = [0] * total_samples
    similarityScore_context2 = [0] * total_samples

    difference = [0] * total_samples

    for i in range(total_samples):

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

        # 3 corresponds to the total number of layers in the elmo model (1 for each layer)
        assert (len(vectors2) == 3)
        assert (len(vectors2[0]) == len(tokens2))

        similarityScore_context2[i] = 1 - scipy.spatial.distance.cosine(vectors2[2][word1_context2[i]], vectors2[2][word2_context2[i]])

        difference[i] = similarityScore_context1[i] - similarityScore_context2[i]

    return difference, similarityScore_context1, similarityScore_context2


def correlationMethod(diff):
    gold_data = pd.read_csv(sys.argv[2])
    gold_df = pd.DataFrame(gold_data)
    gold_standard_scores = gold_df["diff"]
    spearman_value, p_value = spearmanr(diff, gold_standard_scores)
    return spearman_value, p_value

def plotting(sim_context1,sim_context2,diff,data_df,total_samples):

    plt.plot(sim_context1,label="Context 1")
    plt.plot(sim_context2,label="Context 2")


    x_labels_word1 = data_df["word1"]
    x_labels_word2 = data_df["word2"]

    xlabels = [0] * total_samples
    xticks_x = [0] * total_samples


    for wp in range (total_samples):
        xlabels[wp] = x_labels_word1[wp]+ "\n"+x_labels_word2[wp]
        xticks_x[wp] = wp+1
    
    plt.plot(diff,label="Difference")

    plt.legend(loc='center right')

    # Add title and x, y labels
    plt.title("Elmo Embedding Model Results", fontsize=16, fontweight='bold')

    plt.xlabel("Word")
    plt.ylabel("Similarity")

    plt.xticks(xticks_x, xlabels)
    plt.show()


if __name__ == '__main__':
    main()
