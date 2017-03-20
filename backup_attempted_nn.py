from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score as accuracy
# clf = MLPClassifier(solver='adam',activation='logistic', alpha=1e-5, hidden_layer_sizes=(64, 64), random_state=1,verbose=True)
import pandas as pd
from sklearn.preprocessing import scale

def dense_to_one_hot(labels_dense,num_classes=2):
    num_labels = labels_dense.shape[0]
    labels_one_hot = zeros((num_labels,num_classes))
    index = 0
    for i in range(num_labels):
        if labels_dense[i] == 0:
            index = 0
        else:
            index=1
        labels_one_hot[i][index] = 1
    return labels_one_hot


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
# import scipy.io as spio
#
# left_signals = spio.loadmat('./left_signals')['left_signals']
#
#
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout,LSTM
from keras.models import load_model
# import pandas as pd
from numpy import *
import random


# # import pdb; pdb.set_trace()
model = Sequential()
model.add(LSTM(64,input_shape=(1375,132),return_sequences=True))
model.add(LSTM(32, return_sequences=True, inner_activation='sigmoid', activation='hard_sigmoid'))
model.add(LSTM(32, inner_activation='sigmoid', activation='hard_sigmoid'))
model.add(Dense(1,activation = 'softmax'))


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#
# model.save('my_model.h5')
#
# model = load_model('my_model.h5')

left = pd.read_csv('W_into_left.csv',header=None)
right = pd.read_csv('W_into_right.csv',header=None)
print "Finally done loading files :P"
left = left.transpose()
right = right.transpose()
# left = pd.DataFrame(np.random.rand(1375*10,132))
# right = pd.DataFrame(np.random.rand(1375*10,132))

labels = []
# for i in range(len(left)-1375*5):
#     labels.append(0)
#
# for i in range(len(right)-1375*5):
#     labels.append(1)

for i in range(len(left)/1375-5):
    labels.append(0)

for i in range(len(right)/1375-5):
    labels.append(1)



data = left.as_matrix()[:-(1375*5)][:]
data = vstack((data,right.as_matrix()[:-(1375*5)][:]))
data = scale(data)
# data = left.as_matrix()[:-5][:]
# data = vstack((data,right.as_matrix()[:-(5)][:]))


# all_data = c_[data,labels]
# random.shuffle(all_data)
# labels = dense_to_one_hot(all_data[:,-1:],2)

# data = all_data[:,:-1]
data = reshape(data,(119,1375,132))


# clf.fit(data,labels.ravel())



model.fit(data, labels, nb_epoch=10)

# test_labels = [0 for i in range(1375*5)]
# test_labels += [1 for i in range(1375*5)]
# test_data = left.as_matrix()[-(1375*5):][:]
# test_data = vstack((test_data, right.as_matrix()[-(1375*5):][:]))

test_labels = [0 for i in range(5)]
test_labels += [1 for i in range(5)]
test_data = left.as_matrix()[-(1375*5):][:]
test_data = vstack((test_data, right.as_matrix()[-(1375*5):][:]))
test_data = scale(test_data)
test_data = reshape(test_data,(10,1375,132))

# ypred = clf.predict(test_data)
# score = accuracy(test_labels,ypred)
# print score
model.save('my_model.h5')

# model = load_model('my_model.h5')

score,acc = model.evaluate(test_data, test_labels)
print score
print acc
