from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score as accuracy
import numpy as np
from numpy import *
import random
import pandas as pd
from sklearn.preprocessing import scale

clf = MLPClassifier(solver='adam',activation='relu', alpha=0.005, hidden_layer_sizes=(10,5),verbose=True
,warm_start=True,tol=0.000001, learning_rate='invscaling')


# import pdb; pdb.set_trace()
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
#
#
# from keras.models import Sequential
# from keras.layers import Dense, Activation,Dropout,LSTM
# from keras.models import load_model

# model = Sequential()
# model.add(LSTM(64,input_shape=(1375,132),return_sequences=True))
# model.add(LSTM(32, return_sequences=True, inner_activation='sigmoid', activation='hard_sigmoid'))
# model.add(LSTM(32, inner_activation='sigmoid', activation='hard_sigmoid'))
# model.add(Dense(1,activation = 'softmax'))
#
#
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
#
# model.save('my_model.h5')
#
# model = load_model('my_model.h5')

left = pd.read_csv('W_into_left.csv',header=None)
right = pd.read_csv('W_into_right.csv',header=None)
print "done"
left = left.transpose()
right = right.transpose()
labels = []
for i in range(len(left)-1375*5):
    labels.append(0)

for i in range(len(right)-1375*5):
    labels.append(1)

data = left.as_matrix()[:-(1375*5),:]
data = vstack((data,right.as_matrix()[:-(1375*5),:]))

# data = c_[left.as_matrix()[:-5,-10:] + left.as_matrix()[:-5,:10]]
# data = vstack((data,c_[right.as_matrix()[:-(5),-10:] + right.as_matrix()[:-5,:10]]))

all_data = []
all_data = c_[data,labels]
random.shuffle(all_data)
labels = all_data[:,-1:]

data = all_data[:,:-1]
data = scale(data)
# data = reshape(data,(119,1375,132))

for i in range(200):
    print "**",i
    clf.fit(data,labels.ravel())


# model.fit(data, labels, nb_epoch=10,batch_size=1)

test_labels = [0 for i in range(1375*5)]
test_labels += [1 for i in range(1375*5)]
test_data = left.as_matrix()[-(1375*5):,:]
test_data = vstack((test_data, right.as_matrix()[-(1375*5):,:]))
test_data = scale(test_data)
# test_labels = [0 for i in range(5)]
# test_labels += [1 for i in range(5)]
# test_data = c_[left.as_matrix()[-5:,-10:] + left.as_matrix()[-5:,:10]]
# test_data = vstack((test_data,c_[right.as_matrix()[-(5):,-10:] + right.as_matrix()[-5:,:10]]))


ypred = clf.predict(test_data)
score = accuracy(test_labels,ypred)
print score
# score,acc = model.evaluate(test_data, test_labels, batch_size=16,show_accuracy=True)
# print score + "\n" + acc + "\n"
