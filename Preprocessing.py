import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels(bug_data_file, nonBug_data_file):

    bug_examples = list(open(bug_data_file, "r").readlines())
    bug_examples = [s.strip() for s in bug_examples]
    nonBug_examples = list(open(nonBug_data_file, "r").readlines())
    nonBug_examples = [s.strip() for s in nonBug_examples]

    x_text = bug_examples + nonBug_examples
    x_text = [clean_str(sent) for sent in x_text]

    bug_labels = [[1, 0] for _ in bug_examples]
    nonBug_labels = [[0, 1] for _ in nonBug_examples]
    y = np.concatenate([bug_labels, nonBug_labels], 0)
    return [x_text, y]