import os
import pandas as pd
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()
# #lemmatize nouns
print(wnl.lemmatize('cars','n'))
print(wnl.lemmatize('men','n'))
#lemmatize verbs
print(wnl.lemmatize('running','v'))
print(wnl.lemmatize('ate','v'))

#lemmatize adjectives
print(wnl.lemmatize('saddest','a'))
print(wnl.lemmatize('fancier','a'))