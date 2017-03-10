from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score as accuracy
clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32), random_state=1,verbose=True)
import pandas as pd
# import scipy.io as spio
#
# left_signals = spio.loadmat('./left_signals')['left_signals']
#
#
import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, Activation,Dropout,LSTM
# import pandas as pd
from numpy import *
import random
# # import pdb; pdb.set_trace()
# model = Sequential([
#     Dense(64, input_dim=132),
#     Activation('relu'),
#     Dense(64),
#     Activation('relu'),
#     Dropout(0.5),
#     Dense(1,activation = 'softmax'),
# ])
#
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])


left = pd.read_csv('W_into_left.csv',header=None)
right = pd.read_csv('W_into_right.csv',header=None)
left = left.transpose()
right = right.transpose()
labels = []
for i in range(len(left)-(1375*5)):
    labels.append(0)

for i in range(len(right)-(1375*5)):
    labels.append(1)
data = left.as_matrix()[:-(1375*5)][:]
data = vstack((data,right.as_matrix()[:-(1375*5)][:]))


all_data = c_[data,labels]
random.shuffle(all_data)
labels = all_data[:,-1:]

data = all_data[:,:-1]


test_labels = [0 for i in range(1375*5)]
test_labels += [1 for i in range(1375*5)]
test_data = left.as_matrix()[-(1375*5):][:]
test_data = vstack((test_data, right.as_matrix()[-(1375*5):][:]))

labels = list(labels)
clf.fit(data,labels)
ypred = clf.predict(test_data)
score = accuracy(test_labels,ypred)
# data = np.random.random((1000, 784))
# labels = np.random.randint(2, size=(1000, 1))
# test_data = np.random.random((200, 784))
# test_labels = np.random.randint(2, size=(200, 1))
# model.fit(data, labels, nb_epoch=10, batch_size=32)
# score = model.evaluate(test_data, test_labels, batch_size=16)
print score
