##use MLBioMedLAT-780-Questions.csv

import os
import pandas as pd
import numpy as np
from pandas import DataFrame,Series
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize,pos_tag
from nltk.corpus import wordnet
inputfile = "MLBioMedLAT-780-Questions.csv"
data = pd.read_csv(inputfile, encoding='utf-8',header=0,sep = None,engine='python')
dataQuestion = data['Question']
def get_word_net_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB;
    elif tag.startswith('N'):
        return wordnet.NOUN;
    elif tag.startswith('R'):
        return wordnet.ADV;
    else:
        return None

def dataProcess():
    wnl = WordNetLemmatizer()
    dataQuestionDealt = []
    #lemmatize sentence
    for sentence in dataQuestion:
        tokens = word_tokenize(sentence)
        tagged_sent =  pos_tag(tokens)
        lemmas_sent = []
        for tag in tagged_sent:
            word_net_pos = get_word_net_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(tag[0],pos = word_net_pos))
        dataQuestionDealt.append(lemmas_sent)

    dataQuestionDealt = DataFrame(dataQuestionDealt)
    dataY = DataFrame(data['Labels'])
    #print(dataQuestionDealt)
    #print(dataY)
    dataDealt = pd.concat([dataQuestionDealt, dataY], axis=1)
    print(dataDealt)
    return dataDealt
#dataDealt为词形还原后数据






















