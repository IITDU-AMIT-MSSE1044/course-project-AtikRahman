# from sklearn.feature_extraction.text import TfidfTransformer
# from create_bug_featuresets import create_feature_sets_and_labels
# train_x, train_y, test_x, test_y = create_feature_sets_and_labels('bug.txt', 'nonBug.txt')
# train_x_counts = len(train_x)
#
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # vectorizer = TfidfVectorizer()
# # vectorizer.fit_transform(corpus)
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#
# tf_transformer = TfidfTransformer(use_idf=False).fit(train_x_counts)
# X_train_tf = tf_transformer.transform(train_x_counts)
# print(X_train_tf.shape)


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(subset='train',
    categories=categories, shuffle=True, random_state=42)

print(len(twenty_train.data))
print(twenty_train.data[1])
datas = twenty_train.data

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape


tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape

