import create_bug_featuresets
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

train_x, train_y, test_x, test_y = create_bug_featuresets.create_feature_sets_and_labels_for_nn('bug.txt', 'nonBug.txt')
x_train, y_train, x_test, y_test = create_bug_featuresets.create_feature_sets_and_labels('bug.txt', 'nonBug.txt')
print('length of trainset', len(train_x))
print('length of testset', len(test_x))

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2
batch_size = 64
hm_epochs = 30

x = tf.placeholder('float')
y = tf.placeholder('float')

hidden_1_layer = {'weight': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'weight': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'bias': tf.Variable(tf.random_normal([n_classes])), }


# Nothing changes
def neural_network_model(data):
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weight']), hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weight']) + output_layer['bias']

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    print('prediction is ', prediction)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # print('Initial weights: ', sess.run(hidden_1_layer['weight']))


        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                epoch_loss += c
                i += batch_size
                # print('next weights: ', sess.run(hidden_1_layer['weight']))

            print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))
        # print('final weights: ', sess.run(hidden_1_layer['weight']))
        nn_accuracy = accuracy.eval({x: test_x, y: test_y})
        return nn_accuracy

def logistic_regression():
    log_reg = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1,
                                 penalty='l2', random_state=None, tol=0.0001)
    log_reg.fit(x_train, y_train)
    accuracy_lr = log_reg.score(x_test, y_test)
    # print('accuracy of logistic regression: ', accuracy_lr)
    return accuracy_lr

def random_forest():
    random_forest = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    random_forest.fit(x_train, y_train)
    accuracy_rf = random_forest.score(x_test, y_test)
    # print('accuracy of random forest: ', accuracy_rf)
    return accuracy_rf

neural_network_accuracy = train_neural_network(x)
logistic_reg_accuracy = logistic_regression()
random_forest_accuracy = random_forest()

print('accuracy from logistic regression: ', logistic_reg_accuracy)
print('accuracy from random forest', random_forest_accuracy)

print('accuracy from neural network: ', neural_network_accuracy)


