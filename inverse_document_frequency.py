import numpy as np
from collections import Counter

mydoclist = ['Julie loves me more than Linda loves me',
'Jane likes me more than Julie loves me',
'He likes basketball more than baseball']


def freq(term, document):
  return document.split().count(term)

def numDocsContaining(word, doclist):
    doccount = 0
    for doc in doclist:
        if freq(word, doc) > 0:
            doccount +=1
    return doccount

def build_lexicon(corpus):
    lexicon = set()
    for doc in corpus:
        lexicon.update([word for word in doc.split()])
    return lexicon

# vocabulary = build_lexicon(mydoclist)


def idf(word, doclist):
    n_samples = len(doclist)
    df = numDocsContaining(word, doclist)
    ratio = n_samples / (1+df)
    return np.log(ratio)

# idf_value =idf('noth', mydoclist)
# print('idf value: ', idf_value)
