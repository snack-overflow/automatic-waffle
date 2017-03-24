from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score as accuracy
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA



#################################################
# solver    activation      score (most of the times)
#
# lbfgs     tanh            0.5
# lbfgs     identity        0.5
# lbfgs     relu            0.5
# lbfgs     logistic        0.5
#
# sgd       relu            0.5
# sgd       tanh            0.5
# sgd       identity        0.5
# sgd       logistic        0.5
#
# adam      relu            0.5
# adam      tanh            0.5
# adam      identity        0.5
# adam      logistic        0.5

import numpy as np
from numpy import *
import random




left = pd.read_csv('left-signals.csv',header=None)
right = pd.read_csv('right-signals.csv',header=None)
print "Finally done loading files :P"
# left = left.transpose()
# right = right.transpose()

labels = []

for i in range(len(left)-1375*5):
    labels.append(0)

for i in range(len(right)-1375*5):
    labels.append(1)

data = left.as_matrix()[:-(1375*5)][:]
data = vstack((data,right.as_matrix()[:-(1375*5)][:]))


data = scale(data)



# pca = PCA(n_components=8)
# train_x = pca.fit_transform(train_x)

pca = PCA(n_components=15)
data = pca.fit_transform(data)

clf = MLPClassifier(solver='adam',activation='logistic',
 alpha=1e-5, hidden_layer_sizes=(32), random_state=1,verbose=True)

clf.fit(data,labels)



test_labels = [0 for i in range(1375*5)]
test_labels += [1 for i in range(1375*5)]
test_data = left.as_matrix()[-(1375*5):][:]
test_data = vstack((test_data, right.as_matrix()[-(1375*5):][:]))

test_data = scale(test_data)

test_data = pca.fit_transform(test_data)

ypred = clf.predict(test_data)
score = accuracy(test_labels,ypred)
print score
