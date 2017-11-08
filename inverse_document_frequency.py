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

vocabulary = build_lexicon(mydoclist)


def idf(word, doclist):
    n_samples = len(doclist)
    df = numDocsContaining(word, doclist)
    return np.log(n_samples / 1+df)

def build_idf_matrix(idf_vector):
    idf_mat = np.zeros((len(idf_vector), len(idf_vector)))
    np.fill_diagonal(idf_mat, idf_vector)
    return idf_mat

my_idf_vector = [idf(word, mydoclist) for word in vocabulary]
print ('Our vocabulary vector is [' + ', '.join(list(vocabulary)) + ']')
print ('The inverse document frequency vector is [' + ', '.join(format(freq, 'f') for freq in my_idf_vector) + ']')

# my_idf_matrix = build_idf_matrix(my_idf_vector)
# print(my_idf_matrix)