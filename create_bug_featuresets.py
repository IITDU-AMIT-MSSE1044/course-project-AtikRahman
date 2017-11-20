import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import random
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import csv
from sklearn import svm
from inverse_document_frequency import idf


from Preprocessing import clean_str

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def create_lexicon(bug, non_bug):
    lexicon = []
    with open(bug, encoding='utf8') as f:
        for line in f:
            all_words = clean_str(line)
            all_words = word_tokenize(all_words)
            for w in all_words:
                if w not in stop_words:
                    lexicon.append(w)
            # lexicon += list(all_words)

    with open(non_bug, encoding='utf8') as f:
        for line in f:
            all_words = clean_str(line)
            all_words = word_tokenize(all_words)
            for w in all_words:
                if w not in stop_words:
                    lexicon.append(w)
            # lexicon += list(all_words)

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)
    l2 = []
    for w in w_counts:
        # print(w_counts[w])
        if w_counts[w] > 5:
            l2.append(w)
    print(len(l2))
    return l2

def create_full_document(bug_file, nonBug_file):
    positive_examples = list(open(bug_file, encoding="utf8").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(nonBug_file, encoding="utf8").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]

    return x_text


def sample_handling(sample, lexicon, classification):
    featureset = []
    full_documents = create_full_document('bug.txt', 'nonBug.txt')

    with open(sample, encoding='utf8') as f:
        for line in f:
            current_words = word_tokenize(line.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word in lexicon:
                    idf_value = idf(word, full_documents)
                    index_value = lexicon.index(word.lower())
                    features[index_value] = idf_value

            features = list(features)
            featureset.append([features, classification])
            # print(featureset)

    return featureset

def create_features(sample, lexicon, classification):
    featureset = []
    with open(sample, encoding='utf8') as f:
        for line in f:
            current_words = word_tokenize(line.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1

            features = list(features)
            featureset.append([features, classification])
            # print(featureset)

    return featureset

def create_feature_sets_and_labels(bug, non_bug, test_size=0.1):
    lexicon = create_lexicon(bug, non_bug)
    features = []
    features += sample_handling(bug, lexicon, [1])
    features += sample_handling(non_bug, lexicon, [0])
    random.shuffle(features)
    features = np.array(features)

    testing_size = int(test_size * len(features))

    train_x = list(features[:, 0][:-testing_size])
    train_y = list(features[:, 1][:-testing_size])
    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])

    return train_x, train_y, test_x, test_y

def create_feature_sets_and_labels_for_nn(bug, non_bug, test_size=0.1):
    lexicon = create_lexicon(bug, non_bug)
    features = []
    features += create_features(bug, lexicon, [1, 0])
    features += create_features(non_bug, lexicon, [0, 1])
    random.shuffle(features)
    features = np.array(features)

    testing_size = int(test_size * len(features))

    train_x = list(features[:, 0][:-testing_size])
    train_y = list(features[:, 1][:-testing_size])
    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])

    return train_x, train_y, test_x, test_y




