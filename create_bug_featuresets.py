import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import random
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def create_lexicon(bug, non_bug):
    lexicon = []
    with open(bug, encoding='utf8') as f:
        for line in f:
            all_words = word_tokenize(line)
            lexicon += list(all_words)

    with open(non_bug, encoding='utf8') as f:
        for line in f:
            all_words = word_tokenize(line)
            lexicon += list(all_words)

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)
    l2 = []
    for w in w_counts:
        # print(w_counts[w])
        if 1000 > w_counts[w] > 50:
            l2.append(w)
    print(len(l2))
    return l2


def sample_handling(sample, lexicon, classification):
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
    features += sample_handling(bug, lexicon, [1, 0])
    features += sample_handling(non_bug, lexicon, [0, 1])
    random.shuffle(features)
    features = np.array(features)

    testing_size = int(test_size * len(features))

    train_x = list(features[:, 0][:-testing_size])
    train_y = list(features[:, 1][:-testing_size])
    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])

    return train_x, train_y, test_x, test_y


tr_x, tr_y, t_x, t_y = create_feature_sets_and_labels('bug.txt', 'nonBug.txt')
print('ok...')