import os
import pandas as pd
import numpy as np
from pandas import DataFrame,Series
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize,pos_tag
from nltk.corpus import wordnet
from preprocess import dataProcess


def dataCopyTransfer():
    data = dataProcess()
    dataCopy = DataFrame()
    dataLabels = data['Labels']
    #i represents lineIndex
    for i in range(len(data)):
        str = dataLabels[i]
        labels = str.split(",")
        # print(d)
        # print(labels)
        D = DataFrame()
        newData = DataFrame()
        D = data[i:i+1];
        newData = pd.concat([D]*len(labels),ignore_index=True)

        for index in range(len(labels)):
            newData.loc[index,'Labels']=labels[index]
        #print(newData)
        dataCopy = pd.concat([dataCopy,newData],ignore_index=True)


    print("dataCopy**********:\n"+dataCopy)

def dataLabelPowerTransfer():
    data = dataProcess()
    dataLabelPower = DataFrame()
    print("dataLabelPower##########:\n" + dataLabelPower)


dataCopyTransfer()
dataLabelPowerTransfer()

