#ELMo文本特征提取
import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm
import re
import time
import pickle

import tensorflow_hub as hub
import tensorflow as tf

train = pd.read_csv("train_2kmZucJ.csv",encoding='utf_8')
test = pd.read_csv("test_oJQbWVk.csv",encoding='utf_8')
elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
# x = ["I'm a Graduated student"]
# embeddings = elmo(x, signature="default", as_dict=True)["elmo"]
# print(embeddings.shape)

#preprocess
# remove URL's from train and test
train['clean_tweet'] = train['tweet'].apply(lambda x: re.sub(r'httpS+', '', x))
test['clean_tweet'] = test['tweet'].apply(lambda x: re.sub(r'httpS+', '', x))
#remove punctuation
punctuation = '!"#$%&()*+-/:;< =>?@[\]^_`{|}~'
train['clean_tweet'] = train['clean_tweet'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))
test['clean_tweet'] = test['clean_tweet'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))



#lowercase
train['clean_tweet'] = train['clean_tweet'].str.lower()
test['clean_tweet'] = test['clean_tweet'].str.lower()

train['clean_tweet'] = train['clean_tweet'].str.replace("[0-9]", " ")
test['clean_tweet'] = test['clean_tweet'].str.replace("[0-9]", " ")

train['clean_tweet'] = train['clean_tweet'].apply(lambda x:' '.join(x.split()))
test['clean_tweet'] = test['clean_tweet'].apply(lambda x: ' '.join(x.split()))



nlp = spacy.load('en_core_web_sm',disable = ['parser','ner'])

# function to lemmatize text

def lemmatization(texts):
    result = []
    for i in texts:
        s = [token.lemma_ for token in nlp(i)]
        result.append(' '.join(s))
    return result

train['clean_tweet'] = lemmatization(train['clean_tweet'])
test['clean_tweet'] = lemmatization(test['clean_tweet'])

print(train.sample(10)['clean_tweet'])

def elmo_vectors(x):
    embeddings = elmo(x.tolist(), signature="default", as_dict=True)["elmo"]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        return sess.run(tf.reduce_mean(embeddings, 1))

list_train = [train[i:i+100] for i in range(0,train.shape[0],100)]
list_test = [test[i:i+100] for i in range(0,test.shape[0],100)]

elmo_train = [elmo_vectors(x['clean_tweet']) for x in list_train]
elmo_test = [elmo_vectors(x['clean_tweet']) for x in list_test]

print('words2vectors finish')

elmo_train_new = np.concatenate(elmo_train, axis = 0)
elmo_test_new = np.concatenate(elmo_test, axis = 0)

pickle_out = open("elmo_train_03032019.pickle","wb")
pickle.dump(elmo_train_new, pickle_out)
pickle_out.close()

print("store elmo successfully")