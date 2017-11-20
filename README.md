## Bug or Not? Bug Report Classification using N-Gram IDF.

# Description: 
This project aims to classify bug reports whether it is bug or not bug. It performs a bug reports classification with N-gram Inverse Document Frequency (IDF). The inverse document frequency is a measure of how much information the word provides, that is, how important a term is. This project builds classification models with popular machine learning classification techniques Logistic Regression and Random Forest using features from N-gram IDF.

## My Project
The steps of this project can be divided into following layers:

**Layer 1:** Data Input Layer -  
Two pre-annotated bug and non-bug datasets are used as the input.

**Layer 2:** Data Preprocessing Layer - 
Before extracting features from the training set, full dataset have to be preprocessed. Removing URLs, punctuation, Remove stop words. These are removed because they do not contribute to any sentiment. After tokenizing all words, lemmatization is performed to reduce inflectional forms of a word to a common base form.

**Layer 3:** Classification Layer - 
After preprocessing of the bug reports, it is extracted features for classifying the bug reports. To extract feature, a lexicon has been created and then calculated inverse document frequency for each word. Finally a feature matrix has been built from lexicon with idf for every document. This feature matrix split with training and testing sets. After that, Logistic Regression and Random Forest classifiers are used to classify bug and non-bug reports.

## Extension
I have extended this base project. In addition I have used neural network technique to train bug report and non-bug report. Neural network is a set of methods to let this system try to learn which one is bug and which one not but from thousands of bug report examples. In this training, count vector is used instead of inverse document frequency. In my neural network, I have used three hidden layer where 500 hundred nodes exist in each layer. Rectified linear unit (ReLU) is used for activation function. To optimize cost, I use the AdamOptimizer. 

## Dataset

[Jira Duraspace](https://jira.duraspace.org/browse/DS-2732?jql=project%20%3D%20DS%20AND%20resolution%20%3D%20Unresolved%20ORDER%20BY%20priority%20DESC)

## Instructions to use
**Prerequisite:** Python

* Clone the project
* Build and run 
* run command: python Neural_Network.py

## Reference
[1] Terdchanakul, Pannavat, et al. "Bug or Not? Bug Report Classification Using N-Gram   IDF." Proceeding of 33rd IEEE International Conference on Software Maintenance and Evolution (ICSME), 2017.

[2] Lam, Savio LY, and Dik Lun Lee. "Feature reduction for neural network based text categorization." Database Systems for Advanced Applications, 1999. Proceedings., 6th International Conference on. IEEE, 1999.


